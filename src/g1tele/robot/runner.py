r"""The robot-host process: operator frames in, joint targets to a backend, feedback out.

    OperatorFrame --(link)--> Retargeter --> RobotBackend (kinematic / MuJoCo / Unitree DDS)
                                   \-> EpisodeRecorder
It reacts to a new frame immediately (no fixed-rate polling in the way) and keeps the backend
ticking at its own rate in between.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from g1tele import protocol
from g1tele.config import Config
from g1tele.link import RobotLink
from g1tele.record import EpisodeRecorder
from g1tele.retarget.retargeter import Retargeter
from g1tele.robot.base import RobotBackend

log = logging.getLogger('g1tele.robot')


def make_backend(cfg: Config) -> RobotBackend:
    name = cfg.robot.backend
    if name == 'kinematic':
        from g1tele.robot.kinematic import KinematicBackend
        return KinematicBackend(cfg)
    if name == 'mujoco':
        from g1tele.robot.mujoco_backend import MujocoBackend
        return MujocoBackend(cfg)
    if name == 'dds':
        from g1tele.robot.unitree_dds import UnitreeDDSBackend
        return UnitreeDDSBackend(cfg)
    raise ValueError(f"robot.backend must be kinematic, mujoco or dds, got {name!r}")


class RobotRunner:
    def __init__(self, cfg: Config, backend: Optional[RobotBackend] = None,
                 link: Optional[RobotLink] = None):
        self.cfg = cfg
        self.retargeter = Retargeter(cfg)
        self.backend = backend or make_backend(cfg)
        self.link = link or RobotLink(cfg.link)
        self.recorder: Optional[EpisodeRecorder] = None
        self.last_frame_time = 0.0
        self.latest_targets = None
        self.stats = {'frames': 0, 'apply_ms': 0.0, 'latency_ms': 0.0}
        self.fps_window = []
        self._last_feedback = 0.0
        self._jpeg: Optional[bytes] = None
        self._jpeg_id = 0
        self._sent_jpeg_id = 0
        self._last_picture_sent = 0.0
        self._recv_wall = 0.0
        self._echo = {}

    # -------------------------------------------------------------- events
    def handle_event(self, ev: dict):
        kind = ev.get('cmd')
        if kind == 'record_start' and self.cfg.record.enabled:
            self.recorder = EpisodeRecorder(self.cfg.record.root, ev.get('task', 'task'),
                                            ev.get('goal', ''), self.cfg.record.fps,
                                            raw=self.cfg.record.raw_keypoints)
            self.recorder.start()
            log.info('recording started: %s', self.recorder.dir)
        elif kind == 'record_stop' and self.recorder is not None:
            path = self.recorder.stop(save=ev.get('save', True))
            log.info('recording %s', f'saved to {path}' if path else 'discarded')
            self.recorder = None
        elif kind == 'reset':
            self.backend.reset_scene()
        else:
            log.debug('ignored event %s', ev)

    # -------------------------------------------------------------- one frame
    def on_frame(self, frame: protocol.OperatorFrame):
        t_recv = time.time()
        t0 = time.perf_counter()
        targets = self.retargeter.step(frame)
        self.backend.apply(targets)
        apply_ms = (time.perf_counter() - t0) * 1e3
        self.latest_targets = targets
        self.last_frame_time = time.monotonic()
        latency = (time.time() - frame.t_capture) * 1e3     # includes any clock offset between hosts
        self.stats.update(apply_ms=apply_ms, latency_ms=latency)
        self.stats['frames'] += 1
        self._recv_wall = t_recv
        self._echo = {'seq': frame.seq, 't_send': frame.t_send, 't_recv': t_recv}
        now = time.monotonic()
        self.fps_window = [t for t in self.fps_window if now - t < 1.0] + [now]
        if self.recorder is not None and frame.mode == protocol.TRACKING:
            self.recorder.add(frame.to_bytes(), targets, self.backend.observe(), self._last_jpeg_image,
                              latency)

    _last_jpeg_image = None

    # -------------------------------------------------------------- main loop
    def run(self, stop=None, max_seconds: Optional[float] = None):
        period = 1.0 / self.backend.rate_hz
        next_tick = time.monotonic()
        t_start = next_tick
        render_stop = threading.Event()
        self._render_stop = render_stop
        if self.cfg.robot.render:
            threading.Thread(target=self._render_loop, args=(render_stop,), daemon=True).start()
        log.info('robot runner up: backend=%s rate=%.0f Hz', self.cfg.robot.backend, self.backend.rate_hz)
        while not (stop is not None and stop.is_set()):
            now = time.monotonic()
            if max_seconds is not None and now - t_start > max_seconds:
                break
            timeout = max(0.0, next_tick - now)
            frame, events = self.link.poll(int(timeout * 1000))
            for ev in events:
                self.handle_event(ev)
            if frame is not None:
                self.on_frame(frame)
            now = time.monotonic()
            if (self.last_frame_time and now - self.last_frame_time > self.cfg.robot.watchdog
                    and self.latest_targets is not None and self.latest_targets.mode != protocol.IDLE):
                log.warning('no operator data for %.1f s: easing to rest', now - self.last_frame_time)
                self.latest_targets = self.retargeter.step(None)
                self.backend.apply(self.latest_targets)
            if now >= next_tick:
                self.backend.step()
                next_tick += period
                if next_tick < now - 5 * period:      # fell far behind: do not try to catch up
                    next_tick = now + period
                self._feedback(now)
        self.shutdown()

    def _render_loop(self, stop):
        """Draws and JPEG-encodes the simulator view at robot.render_fps in its own thread, so a slow GPU
        or a big image never delays the control loop (that was visible as the arms stuttering)."""
        period = 1.0 / self.cfg.robot.render_fps
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                image = self.backend.render()
            except Exception as e:     # no GPU/EGL here: keep teleoperating, without the video
                log.warning('rendering is not available (%s: %s); continuing without the simulator video',
                            type(e).__name__, str(e)[:120])
                return
            if image is not None:
                self._last_jpeg_image = image
                ok, buf = cv2.imencode('.jpg', image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self._jpeg, self._jpeg_id = buf.tobytes(), self._jpeg_id + 1
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _feedback(self, now: float):
        """A picture goes out the moment it exists (at up to robot.render_fps); status alone goes out at 30 Hz
        when there is no video. Never status in between pictures: the link keeps only the newest message, so
        a status-only message would overwrite a picture the operator has not drawn yet."""
        new_picture = self._jpeg_id != self._sent_jpeg_id
        if not new_picture and (now - self._last_feedback < 1.0 / 30.0
                                or now - self._last_picture_sent < 0.05):
            return
        self._last_feedback = now
        t = self.latest_targets
        status = {'t_robot': time.time(), 'echo': self._echo, 'mode': t.mode if t else protocol.IDLE,
                  'arms': t.status if t else {}, 'fps_in': len(self.fps_window),
                  'apply_ms': round(self.stats['apply_ms'], 2),
                  'latency_ms': round(self.stats['latency_ms'], 1),
                  'ik_error_mm': {s: round(1000 * e, 2) for s, e in (t.ik_error.items() if t else [])},
                  'recording': self.recorder is not None and self.recorder.active,
                  'scene': self.backend.scene_status()}
        jpeg = self._jpeg if new_picture else None
        if new_picture:
            self._last_picture_sent = now
        self._sent_jpeg_id = self._jpeg_id
        self.link.send_feedback(status, jpeg)

    def shutdown(self):
        if hasattr(self, '_render_stop'):
            self._render_stop.set()
            time.sleep(0.1)
        if self.recorder is not None:
            self.recorder.stop()
        self.backend.close()
        self.link.close()
