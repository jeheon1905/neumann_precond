import numpy
from setuptools import Extension, setup

from setuptools import find_packages, find_namespace_packages
from Cython.Build import cythonize

extensions = [
    Extension(name="gospel.Poisson_Solver.Poisson_ISF",
              sources=["gospel/Poisson_Solver/Poisson_ISF.pyx",
                       "gospel/Poisson_Solver/source/Faddeeva.cpp",
                       "gospel/Poisson_Solver/source/Poisson_Solver.cpp"],
              include_dirs=[numpy.get_include(), "./gospel/Poisson_Solver/source/"],
              extra_compile_args=["-std=c++11", "-mkl", "-no-multibyte-chars", "-qopenmp", "-xSSE4.2", "-O3"],
              #extra_compile_args=["-std=c++11", "-mkl", "-no-multibyte-chars", "-qopenmp", "-xSSE4.2", "-O3", "-xMIC-AVX512"],
              libraries = ["mkl_rt", "mkl_lapack95_lp64", "mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5", "pthread"],
              #libraries = ['iomp5', 'pthread']
              #libraries = ['mkl_lapack95_lp64', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
             )
    ]

setup(
    name="gospel",
    version="0.0.1",
    description='DFT package with FD method',
    install_requires=[
                      "setuptools",
                      "Cython",
                      "numpy",
                      "scipy",
                      "pylibxc",
                      "spglib",
                      "ase",
                      "dill"
    ],
    ext_modules=cythonize(extensions),
    #packages =["Diagonalize", "examples", "Poisson_Solver", "Pseudopotential", "XC"],
    packages = find_namespace_packages(), #[ "pyquantum" ],
    #py_modules= [ "FD_Operators",  "Grid",  "Mixing",  "Occupation",  "main",  "scf", "util", "__init__" ],
    language="c++",
    zip_safe=False,
    url='https://gitlab.com/jhwoo15/pyquantum',
    )


#setup(name='pyquantum',
#      version='0.0',
##      long_description=long_description(),
#      author='Woo & Choi',
#      author_email='',
#      license='MIT',
#      packages=find_packages(),
#      classifiers=[
#          'Programming Language :: Python :: 3',
#          ],
#      zip_safe=False)
