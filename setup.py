from setuptools import setup, find_packages

setup(
    name="diffuse",
    version="0.1.0",
    description="A package for diffusion models",
    long_description=open("README.md").read(),
    packages=find_packages(include=["diffuse", "diffuse.*"]),
    python_requires=">=3.6",
)
