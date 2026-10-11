"""g1tele-doctor: check this machine, say what is missing and how to fix it.

    g1tele-doctor operator      # the machine with the camera
    g1tele-doctor robot         # the machine running the simulator
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys

OK, WARN, FAIL = '\033[32mOK  \033[0m', '\033[33mWARN\033[0m', '\033[31mFAIL\033[0m'
failures = 0


def report(level, what, hint=''):
    global failures
    failures += level == FAIL
    print(f'[{level}] {what}' + (f'\n        -> {hint}' if hint else ''))


def check_import(module, pip, required=True):
    try:
        mod = importlib.import_module(module)
        report(OK, f'{module} {getattr(mod, "__version__", "")}'.strip())
        return mod
    except Exception as e:
        report(FAIL if required else WARN, f'{module} not importable ({type(e).__name__})', f'pip install {pip}')
        return None


def common():
    v = sys.version_info
    report(OK if v >= (3, 10) else FAIL, f'python {v.major}.{v.minor}.{v.micro}',
           'g1tele needs Python 3.10+' if v < (3, 10) else '')
    for mod, pip in (('numpy', 'numpy<2'), ('scipy', 'scipy'), ('yaml', 'pyyaml'), ('zmq', 'pyzmq'),
                     ('msgpack', 'msgpack')):
        check_import(mod, pip)
    cv2 = check_import('cv2', 'opencv-python')
    return cv2


def operator():
    cv2 = common()
    import os
    if cv2 is not None and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
                                or sys.platform in ('darwin', 'win32')):
        report(WARN, 'no display on this machine', 'the live window needs a screen; use --no-gui over ssh')
    elif cv2 is not None:
        try:
            cv2.namedWindow('x', cv2.WINDOW_NORMAL)    # headless builds raise here
            cv2.destroyAllWindows()
            report(OK, 'OpenCV can open windows')
        except Exception:
            report(WARN, 'OpenCV has no window support (headless build)',
                   'pip uninstall opencv-python-headless && pip install opencv-python, or run with --no-gui')
    check_import('mediapipe', 'mediapipe==0.10.14')
    rs = check_import('pyrealsense2', 'pyrealsense2')
    if rs is not None:
        devices = list(rs.context().query_devices())
        if not devices:
            report(FAIL, 'no RealSense camera found', 'plug it into a USB 3 port; check `lsusb`')
        for d in devices:
            name = d.get_info(rs.camera_info.name)
            usb = d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else '?'
            report(OK if usb.startswith('3') else FAIL, f'{name} on USB {usb}',
                   'USB 2 drops frames and corrupts depth: use a USB 3 port and cable' if not usb.startswith('3') else '')
            has_imu = any('Motion' in s.get_info(rs.camera_info.name) for s in d.query_sensors())
            report(OK if has_imu else WARN, 'IMU (gravity from the camera)' if has_imu else 'no IMU',
                   '' if has_imu else 'without an IMU "up" comes from the floor/desk in view (tilt the camera down '
                                      '10-15 degrees) or from your arms in calibration (about 5 degrees less accurate)')


def robot():
    common()
    check_import('mujoco', 'mujoco', required=False)
    if shutil.which('nvidia-smi'):
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
        report(OK, f'GPU: {out}')
    else:
        report(WARN, 'no NVIDIA GPU found', 'Isaac Sim needs an RTX GPU; MuJoCo does not')
    check_import('cyclonedds', 'cyclonedds', required=False)
    check_import('unitree_sdk2py', 'unitree_sdk2_python (from github.com/unitreerobotics)', required=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('role', choices=['operator', 'robot'])
    args = ap.parse_args(argv)
    {'operator': operator, 'robot': robot}[args.role]()
    print('\nall required checks passed' if not failures else f'\n{failures} check(s) failed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
