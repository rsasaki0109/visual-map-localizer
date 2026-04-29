from pathlib import Path

from setuptools import find_packages, setup

PACKAGE_NAME = "visual_map_localizer_ros"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch",
         [str(p) for p in Path("launch").glob("*.launch.py")]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ryohei Sasaki",
    maintainer_email="rsasaki0109@gmail.com",
    description="ROS2 wrapper for visual-map-localizer (single-image 6DoF VPS).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vps_node = visual_map_localizer_ros.vps_node:main",
        ],
    },
)
