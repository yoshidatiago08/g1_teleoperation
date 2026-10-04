import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'g1_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf', 'meshes'), glob('urdf/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='g1_teleoperation maintainers',
    maintainer_email='tiagoyuzoyoshida05@gmail.com',
    description='Camera-based teleoperation of the Unitree G1: tracking, IK and RViz visualization.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wrist_detector = g1_teleop.wrist_detector:main',
            'g1_arm_ik_node = g1_teleop.g1_arm_ik_node:main',
            'fake_camera = g1_teleop.fake_camera:main',
        ],
    },
)
