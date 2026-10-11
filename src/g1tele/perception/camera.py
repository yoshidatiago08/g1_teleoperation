"""Camera sources. Every source yields RGBDFrame objects: colour, depth aligned to colour, intrinsics.

    RealSenseCamera   the real device (pyrealsense2), with depth post-processing and the IMU
    ReplayCamera      frames recorded by RGBDRecorder (benchmarks and offline tuning)
    LatestFrame       wraps a source in a thread that always holds the NEWEST frame: processing
                      never works through a backlog, so latency stays one frame
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from g1tele.config import CameraCfg


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class RGBDFrame:
    color: np.ndarray                 # HxWx3 uint8, BGR
    depth: np.ndarray                 # HxW float32, metres, 0 = no reading
    intrinsics: Intrinsics
    t_capture: float                  # wall clock, s
    index: int = 0
    gravity: Optional[np.ndarray] = None   # unit vector pointing DOWN, camera frame (from the IMU)


class RealSenseCamera:
    def __init__(self, cfg: CameraCfg):
        import pyrealsense2 as rs
        self.rs = rs
        self.cfg = cfg
        self.pipeline = rs.pipeline()
        config = rs.config()
        if cfg.serial:
            config.enable_device(cfg.serial)
        config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
        config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
        self.has_imu = False
        if cfg.imu:
            try:
                config.enable_stream(rs.stream.accel)
                self.has_imu = True
            except Exception:
                self.has_imu = False
        try:
            profile = self.pipeline.start(config)
        except RuntimeError:
            if not self.has_imu:
                raise
            # no motion module on this camera: run without the IMU
            config = rs.config()
            if cfg.serial:
                config.enable_device(cfg.serial)
            config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
            config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
            profile = self.pipeline.start(config)
            self.has_imu = False
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self._configure_depth(depth_sensor)
        self.align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height)
        self.filters = []
        if cfg.spatial_filter:
            self.filters.append(rs.spatial_filter())
        if cfg.temporal_filter:
            self.filters.append(rs.temporal_filter())
        if cfg.hole_filling:
            self.filters.append(rs.hole_filling_filter())
        self.index = 0
        self._gravity = None
        # accel -> colour sensor rotation, to express gravity in the colour camera frame
        self._accel_to_color = np.eye(3)
        if self.has_imu:
            try:
                ex = profile.get_stream(rs.stream.accel).get_extrinsics_to(profile.get_stream(rs.stream.color))
                self._accel_to_color = np.array(ex.rotation).reshape(3, 3).T
            except Exception:
                self.has_imu = False

    def _configure_depth(self, sensor):
        rs = self.rs
        try:
            preset = self.cfg.depth_preset
            if sensor.supports(rs.option.visual_preset) and preset:
                names = {sensor.get_option_value_description(rs.option.visual_preset, float(i)): float(i)
                         for i in range(int(sensor.get_option_range(rs.option.visual_preset).max) + 1)}
                if preset in names:
                    sensor.set_option(rs.option.visual_preset, names[preset])
            if self.cfg.laser_power >= 0 and sensor.supports(rs.option.laser_power):
                sensor.set_option(rs.option.laser_power, self.cfg.laser_power)
        except Exception:
            pass   # option sets are best effort: they differ between camera models

    def read(self, timeout_ms: int = 2000) -> Optional[RGBDFrame]:
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None
        t = time.time()
        if self.has_imu:
            accel = frames.first_or_default(self.rs.stream.accel)
            if accel:
                a = accel.as_motion_frame().get_motion_data()
                v = np.array([a.x, a.y, a.z])
                n = np.linalg.norm(v)
                if n > 1e-6:
                    # An accelerometer at rest measures the reaction to gravity: it points UP.
                    self._gravity = -(self._accel_to_color @ (v / n))
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        for f in self.filters:
            depth = f.process(depth)
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        self.index += 1
        return RGBDFrame(np.asanyarray(color.get_data()).copy(), depth_m, self.intrinsics, t,
                         self.index, self._gravity)

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class RGBDRecorder:
    """Writes a session to a folder so it can be replayed exactly: colour JPEG, depth 16-bit PNG (mm), the
    calibration the tracker was using, and what the tracker sent live (`live.msgpack`).

    Writing happens in a thread: the camera loop never waits for the disk."""

    def __init__(self, path: str, config: Optional[dict] = None):
        import queue
        self.path = path
        os.makedirs(os.path.join(path, 'color'), exist_ok=True)
        os.makedirs(os.path.join(path, 'depth'), exist_ok=True)
        self.times, self.gravity, self.intrinsics = [], [], None
        self.config, self.profile_start = config, None
        self.n = 0
        self.bytes = 0
        self._q = queue.Queue()
        self._live = open(os.path.join(path, 'live.msgpack'), 'wb')
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_start_profile(self, profile_dict):
        if self.profile_start is None:
            self.profile_start = profile_dict

    def write(self, f: RGBDFrame, live: Optional[bytes] = None):
        self._q.put((self.n, f.color.copy(), f.depth.copy(), live))
        self.times.append(f.t_capture)
        self.gravity.append(None if f.gravity is None else [float(x) for x in f.gravity])
        self.intrinsics = f.intrinsics.to_dict()
        self.n += 1

    def _run(self):
        import msgpack
        while True:
            item = self._q.get()
            if item is None:
                return
            i, color, depth, live = item
            cp = os.path.join(self.path, 'color', f'{i:06d}.jpg')
            dp = os.path.join(self.path, 'depth', f'{i:06d}.png')
            cv2.imwrite(cp, color, [cv2.IMWRITE_JPEG_QUALITY, 92])
            # 2 mm steps: below the sensor's own noise, and much smaller files than millimetres
            cv2.imwrite(dp, np.clip(np.round(depth * 500.0), 0, 65535).astype(np.uint16))
            self.bytes += os.path.getsize(cp) + os.path.getsize(dp)
            if live is not None:
                self._live.write(msgpack.packb(live, use_bin_type=True))

    def pending(self) -> int:
        return self._q.qsize()

    def close(self, profile_end: Optional[dict] = None):
        self._q.put(None)
        self._thread.join()
        self._live.close()
        with open(os.path.join(self.path, 'meta.json'), 'w') as fh:
            json.dump({'intrinsics': self.intrinsics, 't': self.times, 'gravity': self.gravity,
                       'depth_unit': 0.002, 'config': self.config, 'profile_start': self.profile_start,
                       'profile_end': profile_end}, fh)


class ReplayCamera:
    """Plays a folder written by RGBDRecorder. realtime=False returns frames as fast as asked."""

    def __init__(self, path: str, realtime: bool = True, loop: bool = False):
        with open(os.path.join(path, 'meta.json')) as fh:
            meta = json.load(fh)
        self.path, self.realtime, self.loop = path, realtime, loop
        self.intrinsics = Intrinsics(**meta['intrinsics'])
        self.t, self.gravity = meta['t'], meta['gravity']
        self.meta = meta
        self.depth_unit = meta.get('depth_unit', 0.001)        # older recordings: millimetres
        self.i = 0
        self.start = None

    def __len__(self):
        return len(self.t)

    def read(self, timeout_ms: int = 0) -> Optional[RGBDFrame]:
        if self.i >= len(self.t):
            if not self.loop:
                return None
            self.i, self.start = 0, None
        if self.realtime:
            if self.start is None:
                self.start = time.time() - self.t[self.i]
            delay = (self.start + self.t[self.i]) - time.time()
            if delay > 0:
                time.sleep(delay)
        color = cv2.imread(os.path.join(self.path, 'color', f'{self.i:06d}.jpg'))
        depth = cv2.imread(os.path.join(self.path, 'depth', f'{self.i:06d}.png'), cv2.IMREAD_UNCHANGED)
        g = self.gravity[self.i]
        t = time.time() if self.realtime else self.t[self.i]
        frame = RGBDFrame(color, depth.astype(np.float32) * self.depth_unit, self.intrinsics, t, self.i,
                          None if g is None else np.array(g))
        self.i += 1
        return frame

    def close(self):
        pass


class LatestFrame:
    """Runs `source.read()` in a thread; `get()` returns the newest unseen frame."""

    def __init__(self, source):
        self.source = source
        self._frame = None
        self._seen = True
        self._cond = threading.Condition()
        self._stop = False
        self._done = False
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop:
            frame = self.source.read()
            if frame is None:
                if getattr(self.source, 'finite', False) or isinstance(self.source, ReplayCamera):
                    break
                continue
            with self._cond:
                if not self._seen:
                    self.dropped += 1
                self._frame, self._seen = frame, False
                self._cond.notify_all()
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def get(self, timeout: float = 1.0) -> Optional[RGBDFrame]:
        with self._cond:
            if not self._cond.wait_for(lambda: not self._seen or self._done, timeout):
                return None
            if self._seen:
                return None
            self._seen = True
            return self._frame

    def close(self):
        self._stop = True
        self.thread.join(timeout=2.0)
        self.source.close()
