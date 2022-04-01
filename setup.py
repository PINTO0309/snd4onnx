
from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="snd4onnx",
    version="0.0.2",
    description="Simple node deletion tool for onnx.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/snd4onnx",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            "snd4onnx = snd4onnx.onnx_remove_node:main"
        ]
    }
)
