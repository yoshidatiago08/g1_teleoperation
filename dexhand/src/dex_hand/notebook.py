"""Notebook side: only light work. Reads the camera, sends compressed frames to the VM, shows what comes back.

    python -m dex_hand.notebook --vm-host 100.64.0.7 --source webcam

Window: your camera with the tracked hand (drawn by the VM) next to the simulator view (rendered by the VM).
Keys: c = calibrate (palm to the camera, fingers up, hold still; the robot starts imitating when it is done), space = pause/resume imitation (nothing
moves while paused: recentre your hand, resume, carry on; the robot keeps its orientation and only follows your rotation from there),
a = re-align the robot's orientation to your hand's absolute orientation, + / - = motion scale x1.25 / x0.8,
o = put the cylinder back, q = quit. The motion scale lives here (--motion-scale / MOTION_SCALE) and is sent to the VM.
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
        cal = state.get('calibration') or {'state': 'none'}
        cal_text = {'none': 'NOT CALIBRATED: press c, palm to the camera, fingers up',
                    'collecting': 'CALIBRATING: hold still...',
                    'done': f"calibrated (spread {cal.get('spread_deg')} deg){'; ' + cal['hand'] if cal.get('hand') else ''}",
                    'failed': f"CALIBRATION FAILED: {cal.get('why')}"}[cal['state']]
        cam = state.get('camera') or {}
        cam_text = (f"camera tilt from the IMU: pitch {cam['pitch_deg']} deg, roll {cam['roll_deg']} deg" if cam.get('imu')
                    else 'no camera IMU: calibrate with the camera level, or mount it level')
        status = state.get('status', '?')
        arm = state.get('arm')
        arm_text = f"arm IK: error {arm['error_mm']} mm" if arm else 'floating hand (no arm)'
        lines = [cam_text, cal_text,
                 f"{status}   scale {state['scale']:.2f}{latency}   {arm_text}",
                 '[c] calibrate  [space] pause/resume  [a] align  [+/-] scale  [o] reset cylinder',
                 f"cylinder -> target {scene.get('to_target_cm', '?')} cm   lifted {scene.get('lifted_cm', '?')} cm"
                 f"{'   ON TARGET' if scene.get('on_target') else ''}"]
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 352 - 20 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
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
    parser.add_argument('--motion-scale', type=float, default=float(os.environ.get('MOTION_SCALE') or 0.3),
                        help='robot hand motion per unit of your hand motion (sent to the VM)')
    parser.add_argument('--rotation-scale', type=float, default=float(os.environ.get('ROTATION_SCALE') or 1.0))
    parser.add_argument('--mode', default=os.environ.get('MODE') or 'arm', choices=['arm', 'floating'], help='must match the VM (RViz shows the G1 or the hand)')
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
            bridge = RosBridge(args.hand, mode=args.mode)
        except ImportError:
            print('ROS 2 not found: no RViz (use the notebook image, or --no-rviz)', flush=True)
    print(f'dexhand notebook: {args.source} -> {args.vm_host}:{args.port}', flush=True)

    last = {'state': None, 'preview': None, 'sim': None}
    want = {'motion': args.motion_scale, 'rotation': args.rotation_scale}     # this side is the source of truth
    synced = 0.0
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
            state = last.get('state')
            if state and time.time() - synced > 0.5 and (abs(state['scale'] - want['motion']) > 1e-6
                                                          or abs(state['rotation_scale'] - want['rotation']) > 1e-6):
                events_out.send({'cmd': 'set_scale', **want})       # also re-applies it after a VM restart
                synced = time.time()
            if window:
                cv2.imshow(WINDOW, compose(last.get('preview'), last.get('sim'), last.get('state')))
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            if key == ord(' '):
                events_out.send({'cmd': 'clutch'})
            elif key == ord('c'):
                events_out.send({'cmd': 'calibrate'})
            elif key == ord('a'):
                events_out.send({'cmd': 'align'})
            elif key in (ord('+'), ord('=')):
                want['motion'] = min(want['motion'] * 1.25, 3.0)
            elif key in (ord('-'), ord('_')):
                want['motion'] = max(want['motion'] * 0.8, 0.02)
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
