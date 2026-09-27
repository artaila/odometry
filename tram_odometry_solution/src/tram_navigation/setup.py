import os
from glob import glob
from setuptools import setup

package_name = "tram_navigation"

data_files = [
    (os.path.join("share", "ament_index", "resource_index", "packages"), [os.path.join("resource", package_name)]),
    (os.path.join("share", package_name), ["package.xml"]),
]

if os.path.isdir("launch"):
    data_files.append(
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py"))
    )

if os.path.isdir("config"):
    data_files.append(
        (os.path.join("share", package_name, "config"), glob("config/*.yaml"))
    )

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tram Navigation Team",
    maintainer_email="dev@example.com",
    description="Отказоустойчивая резервная одометрия беспилотного трамвая",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "telemetry_synchronizer_node = tram_navigation.telemetry_synchronizer_node:main",
        ],
    },
)
