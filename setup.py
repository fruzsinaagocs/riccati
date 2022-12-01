#!/usr/bin/env python3

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command

docs = ['sphinx', 'sphinx_rtd_theme', 'numpydoc']
tests = ['pytest', 'scipy', 'mpmath']
plots = ['pandas', 'num2tex', 'matplotlib']
all = docs + tests + plots

setup(name='riccati',
      version='0.0.5',
      description='adaptive Riccati defect correction solver',
      long_description='adaptive Riccati defect correction solver',
      author='Fruzsina J Agocs and Alex H Barnett',
      author_email='',
      url='',
      packages=['riccati'],
      install_requires=['numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      extras_require={
          'docs': all,
          'tests': tests,
          'plots': plots
          },
      include_package_data=True,
      license='MIT',
      classifiers=[],
)
