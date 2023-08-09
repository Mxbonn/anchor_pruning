from setuptools import find_packages, setup

setup(
    name="apr",
    packages=find_packages(),
    version="1.0.0",
    description="Anchor Pruning for Object Detection.",
    author="Maxim Bonnaerens",
    license="",
    install_requires=["torch", "torchprofile", "pycocotools"],
)
