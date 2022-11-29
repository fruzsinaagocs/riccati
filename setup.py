#!/usr/bin/env python3

try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command

setup(name='riccati',
      version='0.0.3',
      description='',
      long_description='',
      author='',
      author_email='',
      url='',
      packages=['riccati'],
      install_requires=['matplotlib', 'numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      extras_require={
          'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc']},
      tests_require=['pytest'],
      include_package_data=True,
      license='MIT',
      classifiers=[],
      )
