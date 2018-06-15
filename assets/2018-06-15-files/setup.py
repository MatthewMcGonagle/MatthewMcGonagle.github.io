'''
setup.py

The module name will be csort.
'''

from setuptools import Extension, setup

myModule = Extension('cSort', sources = ['moduleCSort.c'])
setup(name = 'myCSortPackage', ext_modules = [myModule])
