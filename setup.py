from setuptools import setup, find_packages

setup(
    name="paddlemetrics",
    packages=find_packages(exclude=["tests*", "docs*"]),
)
