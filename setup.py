"""
(How to install GOSPEL)
>>> pip install -r requirements.txt
>>> python setup.py develop
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gospel",
    version="0.0.1",
    author="Jeheon Woo",
    author_email="woojh@kaist.ac.kr",
    description="Grid-based Open-Source Python Engine for Large systems",
    long_description=long_description,
    url="https://gitlab.com/jhwoo15/gospel",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: ??",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    # install_requires -> pip install -r requirements.txt
)
