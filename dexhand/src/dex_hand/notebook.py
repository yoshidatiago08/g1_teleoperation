"""Notebook side: only light work. Reads the camera, sends compressed frames to the VM, shows what comes back.

    python -m dex_hand.notebook --vm-host 100.64.0.7 --source webcam

Window: your camera with the tracked hand (drawn by the VM) next to the simulator view (rendered by the VM).
Keys: space = clutch, + / - = motion scale x1.25 / x0.8, o = put the cylinder back, q = quit.
RViz (if ROS 2 is installed) shows the hand, table and cylinder from the same state.
"""
import argparse
import os
import time

import cv2
import numpy as np

from dex_hand import capture, link

WINDOW = 'dexhand'
PLACEHOLDER = np.zeros((360, 480, 3), np.uint8)


def decode(jpeg):
    return PLACEHOLDER if jpeg is None else cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def compose(preview, sim, state):
    left, right = decode(preview), decode(sim)
    height = 360
    tiles = [cv2.resize(t, (int(t.shape[1] * height / t.shape[0]), height)) for t in (left, right)]
    image = np.hstack(tiles)
    lines = ['waiting for the VM...']
    if state:
        latency = f"   latency {1000 * (time.time() - state['t_capture']):.0f} ms" if state.get('t_capture') else ''
        scene = state.get('scene') or {}
        lines = [f"{'CLUTCH' if state['clutch'] else 'ENGAGED' if state.get('engaged') else 'HOLD'}"
                 f"   scale {state['scale']:.2f}{latency}",
                 f"cylinder -> target {scene.get('to_target_cm', '?')} cm   lifted {scene.get('lifted_cm', '?')} cm"
                 f"{'   ON TARGET' if scene.get('on_target') else ''}"]
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 340 - 22 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vm-host', default=os.environ.get('VM_HOST'), help='the VM running dex_hand.vm (or env VM_HOST)')
    parser.add_argument('--port', type=int, default=link.DEFAULT_PORT)
    parser.add_argument('--hand', default='right', choices=['right', 'left'])
    parser.add_argument('--source', default='webcam', choices=['webcam', 'realsense', 'none'],
                        help='none: send no frames (the VM runs with --source synthetic)')
    parser.add_argument('--device', default='0')
    parser.add_argument('--hfov', type=float, default=70.0, help='webcam horizontal field of view (degrees)')
    parser.add_argument('--no-rviz', action='store_true')
    parser.add_argument('--no-window', action='store_true')
    args = parser.parse_args(argv)
    if not args.vm_host:
        parser.error('set --vm-host (or VM_HOST in .env): the IP of the machine running dex_hand.vm')

    frames_out, state_in, events_out = link.notebook_side(args.vm_host, args.port)
    camera = capture.make_capture(args.source, args.device, args.hfov)
    window = not args.no_window and bool(os.environ.get('DISPLAY'))
    bridge = None
    if not args.no_rviz:
        try:
            from dex_hand.ros_bridge import RosBridge
            bridge = RosBridge(args.hand)
        except ImportError:
            print('ROS 2 not found: no RViz (use the notebook image, or --no-rviz)', flush=True)
    print(f'dexhand notebook: {args.source} -> {args.vm_host}:{args.port}', flush=True)

    last = {'state': None, 'preview': None, 'sim': None}
    try:
        while True:
            if camera is not None:
                frame = camera.read()
                if frame is not None:
                    frames_out.send(capture.encode(frame, time.time()))
            message = state_in.recv(0 if camera is not None else 15)
            if message is not None:
                last = message
                if bridge is not None:
                    bridge.publish(message['state'])
            if window:
                cv2.imshow(WINDOW, compose(last.get('preview'), last.get('sim'), last.get('state')))
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            if key == ord(' '):
                events_out.send({'cmd': 'clutch'})
            elif key in (ord('+'), ord('=')):
                events_out.send({'cmd': 'scale', 'factor': 1.25})
            elif key in (ord('-'), ord('_')):
                events_out.send({'cmd': 'scale', 'factor': 0.8})
            elif key == ord('o'):
                events_out.send({'cmd': 'reset'})
            elif key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None:
            camera.close()
        if bridge is not None:
            bridge.close()


if __name__ == '__main__':
    main()
