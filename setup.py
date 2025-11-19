# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
from setuptools import setup, find_packages

setup(
    name="diffuse",
    version="0.1.0",
    description="A package for diffusion models",
    long_description=open("README.md").read(),
    packages=find_packages(include=["diffuse", "diffuse.*"]),
    python_requires=">=3.12",
)
