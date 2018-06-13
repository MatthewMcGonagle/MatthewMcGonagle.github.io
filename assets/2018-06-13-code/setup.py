from setuptools import setup, Extension

# Our module is an extension module since it is C code.

myModule = Extension('hello', sources = ['modulehello.c'])

# The package name doesn't affect the name of the module created.

setup(name = 'myPackageName', ext_modules = [myModule])
