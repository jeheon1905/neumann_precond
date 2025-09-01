import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neumann_precond",
    version="0.0.1",
    author="",
    author_email="",
    description="Preconditioner development using Neumann expansion",
    long_description=long_description,
    url="https://github.com/jeheon1905/neumann_precond",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    # install_requires -> pip install -r requirements.txt
)
