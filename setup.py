#!/usr/bin/env python3

try:
    from setuptools import setup, Command, find_packages
except ImportError:
    from distutils.core import setup, Command, find_packages

def readme():
    with open("README.md") as f:
        return f.read()

docs = ['sphinx', 'sphinx-book-theme', 'numpydoc']
tests = ['numpy', 'scipy', 'mpmath', 'pytest']
plots = ['pandas', 'num2tex', 'matplotlib', 'pyoscode']
all = docs + tests + plots

setup(name='riccati',
      version='1.0.1',
      description='adaptive Riccati defect correction solver',
      long_description=readme(),
      author='Fruzsina J Agocs and Alex H Barnett',
      author_email='',
      url='',
      packages=find_packages(),
      install_requires=['numpy'],
      setup_requires=['setuptools>=40.6.0', 'setuptools_scm', 'wheel', 'pytest-runner'],
      extras_require={
          'all': all,
          'docs': docs,
          'tests': tests,
          'plots': plots
          },
      include_package_data=True,
      license='Apache 2.0',
      classifiers=[],
      options={'bdist_wheel': {'universal': '1'}},
)
