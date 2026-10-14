import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'dex_hand'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf', 'meshes'), glob('urdf/meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='g1_teleoperation maintainers',
    maintainer_email='tiagoyuzoyoshida05@gmail.com',
    description='Camera hand tracking retargeted to the Inspire hand with dex-retargeting.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['dex_hand = dex_hand.node:main']},
)
