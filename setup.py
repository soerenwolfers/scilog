#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.command.install import install 
import os
import pathlib

def _patch_bashrc():
    bashrc_path=os.path.join(str(pathlib.Path.home()),'.bashrc')
    try:
        with open(bashrc_path,'r') as fp:
            bashrc = fp.readlines()
    except FileNotFoundError:
        bashrc=[]
    for line in bashrc:
        if 'eval "$(register-python-argcomplete scilog)"' in line:
            break
    else:
        with open(bashrc_path,'a+') as fp:
            fp.write('# added by scilog installer\neval "$(register-python-argcomplete scilog)"\n')    

class PostInstallCommand(install):
    def run(self):
        _patch_bashrc()
        install.run(self)
setup(
    name='scilog',
    version='1.7.1',
    url = 'https://github.com/soerenwolfers/scilog',
    python_requires='>=3.6',
    long_description=open('README.rst').read(),
    description='Record-keeping for computational experiments',
    author='Soeren Wolfers',
    author_email='soeren.wolfers@gmail.com',
    packages=find_packages(exclude=['*tests']),#,'examples*']),
    install_requires=['swutil','IPython','matplotlib','numpy','argcomplete'],
    scripts=['bin/gitdiffuntracked','bin/scilog'],
    #entry_points={'console_scripts': ['scilog = scilog.command_line:main']},
    cmdclass={
        'install': PostInstallCommand,
    },
)

