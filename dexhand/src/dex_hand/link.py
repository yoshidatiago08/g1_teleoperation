"""The network between the notebook (camera, display) and the VM (MediaPipe, dex-retargeting, MuJoCo).

Three ZeroMQ sockets, msgpack on the wire; the VM binds, the notebook connects:

    frames  notebook -> VM   PUB/SUB   JPEG colour (+ small 16-bit depth), the newest frame wins
    state   VM -> notebook   PUB/SUB   hand command, scene state and two preview JPEGs, the newest wins
    events  notebook -> VM   PUSH/PULL keys: clutch, scale, reset (reliable, none dropped)

Ports are base, base+1, base+2 (default 5601-5603). Nothing here knows about hands.
"""
import zmq
import msgpack

DEFAULT_PORT = 5601


class Channel:
    def __init__(self, kind, port, host=None, bind=False, newest_only=False, hwm=2):
        self.sock = zmq.Context.instance().socket(kind)
        self.sock.setsockopt(zmq.LINGER, 0)
        if kind == zmq.PUB:
            self.sock.setsockopt(zmq.SNDHWM, hwm)            # a slow link drops frames instead of queueing latency
        if kind == zmq.SUB:
            self.sock.setsockopt(zmq.SUBSCRIBE, b'')
            if newest_only:
                self.sock.setsockopt(zmq.CONFLATE, 1)
        if bind:
            self.sock.bind(f'tcp://*:{port}')
        else:
            self.sock.connect(f'tcp://{host}:{port}')

    def send(self, message):
        try:
            self.sock.send(msgpack.packb(message, use_bin_type=True), zmq.NOBLOCK)
        except zmq.Again:
            pass

    def recv(self, timeout_ms=0):
        """The next message, or None if nothing arrives within `timeout_ms`."""
        if self.sock.poll(timeout_ms) == 0:
            return None
        return msgpack.unpackb(self.sock.recv(), raw=False)

    def close(self):
        self.sock.close(0)


def vm_side(port=DEFAULT_PORT):
    """(frames in, state out, events in), bound on this machine."""
    return (Channel(zmq.SUB, port, bind=True, newest_only=True), Channel(zmq.PUB, port + 1, bind=True),
            Channel(zmq.PULL, port + 2, bind=True))


def notebook_side(host, port=DEFAULT_PORT):
    """(frames out, state in, events out), connected to the VM."""
    return (Channel(zmq.PUB, port, host), Channel(zmq.SUB, port + 1, host, newest_only=True),
            Channel(zmq.PUSH, port + 2, host))
