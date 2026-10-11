"""ZeroMQ link between the operator's machine and the robot host.

Three channels, each with the right delivery rule:
    stream    operator state, 30-60 Hz: only the LATEST value matters (CONFLATE), an old frame is
              worse than a dropped one
    control   events (calibrate, record start/stop): reliable and ordered
    feedback  status and simulator video back to the operator: latest value only
The operator's machine connects out to the robot host, which binds the ports.
"""
from __future__ import annotations

import json
from typing import Optional

import zmq

from g1tele.config import LinkCfg
from g1tele.protocol import OperatorFrame


class OperatorLink:
    """Operator side: sends frames and events, receives feedback."""

    def __init__(self, cfg: LinkCfg):
        self.ctx = zmq.Context.instance()
        addr = lambda port: f'tcp://{cfg.host}:{port}'
        self.stream = self.ctx.socket(zmq.PUB)
        self.stream.setsockopt(zmq.SNDHWM, 2)
        self.stream.setsockopt(zmq.LINGER, 0)
        self.stream.connect(addr(cfg.stream_port))
        self.control = self.ctx.socket(zmq.PUSH)
        self.control.setsockopt(zmq.LINGER, 200)
        self.control.connect(addr(cfg.control_port))
        self.feedback = self.ctx.socket(zmq.SUB)
        self.feedback.setsockopt(zmq.CONFLATE, 1)
        self.feedback.setsockopt(zmq.SUBSCRIBE, b'')
        self.feedback.connect(addr(cfg.feedback_port))

    def send_frame(self, frame: OperatorFrame):
        self.stream.send(frame.to_bytes(), zmq.NOBLOCK)

    def send_event(self, **event):
        self.control.send(json.dumps(event).encode())

    def poll_feedback(self) -> Optional[dict]:
        """Latest feedback message, or None. {'status': {...}, 'jpeg': bytes | None}"""
        try:
            raw = self.feedback.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None
        head, _, jpeg = raw.partition(b'\n')
        return {'status': json.loads(head), 'jpeg': jpeg or None}

    def close(self):
        for s in (self.stream, self.control, self.feedback):
            s.close(0)


class RobotLink:
    """Robot-host side: binds the ports, receives frames and events, sends feedback."""

    def __init__(self, cfg: LinkCfg, bind: str = '*'):
        self.ctx = zmq.Context.instance()
        self.stream = self.ctx.socket(zmq.SUB)
        self.stream.setsockopt(zmq.CONFLATE, 1)
        self.stream.setsockopt(zmq.SUBSCRIBE, b'')
        self.stream.bind(f'tcp://{bind}:{cfg.stream_port}')
        self.control = self.ctx.socket(zmq.PULL)
        self.control.bind(f'tcp://{bind}:{cfg.control_port}')
        self.feedback = self.ctx.socket(zmq.PUB)
        self.feedback.setsockopt(zmq.SNDHWM, 2)
        self.feedback.setsockopt(zmq.LINGER, 0)
        self.feedback.bind(f'tcp://{bind}:{cfg.feedback_port}')
        self.poller = zmq.Poller()
        self.poller.register(self.stream, zmq.POLLIN)
        self.poller.register(self.control, zmq.POLLIN)

    def poll(self, timeout_ms: int = 0):
        """Wait up to timeout_ms. Returns (frame | None, [events])."""
        events, frame = [], None
        ready = dict(self.poller.poll(timeout_ms))
        if self.control in ready:
            while True:
                try:
                    events.append(json.loads(self.control.recv(zmq.NOBLOCK)))
                except zmq.Again:
                    break
        if self.stream in ready:
            try:
                frame = OperatorFrame.from_bytes(self.stream.recv(zmq.NOBLOCK))
            except zmq.Again:
                pass
        return frame, events

    def send_feedback(self, status: dict, jpeg: Optional[bytes] = None):
        try:
            self.feedback.send(json.dumps(status).encode() + b'\n' + (jpeg or b''), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def close(self):
        for s in (self.stream, self.control, self.feedback):
            s.close(0)
