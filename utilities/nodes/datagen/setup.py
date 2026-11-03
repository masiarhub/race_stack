from setuptools import setup
import os
from glob import glob

package_name = 'datagen'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'policies'), glob(os.path.join('policies', '*.yaml')) + glob(os.path.join('policies', '*.py'))),
        (os.path.join('share', package_name, 'specs'), glob(os.path.join('specs', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Masiar Etemadi',
    maintainer_email='metemadi@ethz.ch',
    description='Simulated head-to-head data generation for opponent prediction',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mine = datagen.cli:mine',
            'extract = datagen.cli:extract',
            'check = datagen.cli:check',
            'episode_manager = datagen.episode_manager:main',
            'policy_runner = datagen.policy_runner:main',
        ],
    },
)
