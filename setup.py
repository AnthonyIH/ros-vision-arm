from setuptools import find_packages, setup

package_name = 'amr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(where='src', exclude=['test']),
    package_dir={'': 'src'},  # ← tell setuptools to look in src/
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ll',
    maintainer_email='lukas.lindenroth@kcl.ac.uk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hardware_interface = amr.hardware_interface:main',
            'camera_publisher = amr.camera_publisher:main',
            'aruco_pose = amr.aruco_pose:main',
            'marker_tracking = amr.marker_tracking:main',
        ],
    },
)
