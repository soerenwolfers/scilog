#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='scilog',
    version='1.0.5',
    python_requires='>=3',
    long_description=open('README.rst').read(),
    description='Keep track of numerical experiments',
    author='Soeren Wolfers',
    author_email='soeren.wolfers@gmail.com',
    packages=find_packages(exclude=['*tests']),#,'examples*']),
    install_requires=['swutil','IPython'],
    entry_points={'console_scripts': ['scilog = scilog.scilog:main']}
)
