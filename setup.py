
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="snd4onnx",
    version="0.0.1",
    description="Simple node deletion tool for onnx.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/snd4onnx",
    license="MIT License",
    packages=["snd4onnx"],
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "command_name=snd4onnx.onnx_remove_node:main"
        ]
    }
)