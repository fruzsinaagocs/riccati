#!/usr/bin/env python3

try:
    from setuptools import setup, Command, find_packages
except ImportError:
    from distutils.core import setup, Command


PACKAGES = find_packages(where='riccati')

docs = ['sphinx', 'sphinx-book-theme', 'numpydoc']
tests = ['scipy', 'mpmath', 'pytest', 'pytest[toml]']
plots = ['pandas', 'num2tex', 'matplotlib', 'pyoscode']
all = docs + tests + plots

setup(name='riccati',
      version='0.1.0',
      description='adaptive Riccati defect correction solver',
      long_description='adaptive Riccati defect correction solver',
      author='Fruzsina J Agocs and Alex H Barnett',
      author_email='',
      url='',
      packages=PACKAGES,
      packages_dir={"": "riccati"},
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
