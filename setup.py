from setuptools import find_packages, setup

setup(
    name="paddlemetrics",
    packages=find_packages(exclude=["tests*", "docs*"]),
)
