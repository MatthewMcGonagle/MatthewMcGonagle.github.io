---
layout : post
date : 2018-06-13
title : Python Extension Modules, Hello World
---

## [Download `setup.py` Here]({{site . url}}/assets/2018-06-13-code/setup.py)
## [Download `modulehello.c` Here]({{site . url}}/assets/2018-06-13-code/modulehello.c)
## [Download `test.py` Here]({{site . url}}/assets/2018-06-13-code/test.py)

In this post we will look at how to extend Python by creating a C extension module, in particular I will be doing so
for Windows. I had some difficulty getting this to work and finding the right documentation to make everything
work. So I'm collecting what I've found to work in one place for my own ease of reference. Finally I should also
point out that creating an extension module using C code is really only meant to work with the CPython
implementation of the Python language (which I think is pretty standard).

We will keep our example as simple as possible; we will only look at making an extension module that uses C code
to print out "Hello World using printf". After building our extension module, using it will be as simple as
using the following Python code:
``` python
# test.py

import hello 

hello.sayHello()
```
Running this code with Python gives the output
```
Hello World using printf
```

Our information comes from the [official Python docs for extension modules](https://docs.python.org/3/extending/extending.html). However, I had some difficulty getting the directions of the official documents to work 
(as they are at the time of writing this post); in particular I found it better to use the 
`setuptools` module than the `distutils` module. The [official docs on the CPython API 
for C](https://docs.python.org/3/c-api/index.html) will also be helpful for doing anything more 
complicated than this example.

## Get Necessary Microsoft Visual Studio Tools

First, since this discussion is in particular focused on Windows, you need to make sure you have Microsoft Visual 
Studio C++ installed, for safe measure I also made sure I had the Python tools of Microsoft Visual Studio also
installed. 

## Use the `setuptools` Module

We will be using the `setuptools` module. A lot of the documentation on extension modules is centered around
the `distutils` module; however, as far as I can tell, the `setuptools` module is supposed to be a more modern
version of `distutils`. In fact, the documentation for `distutils` recommends using `setuptools`.

Furthermore, when I tried using `distutils`, my build process had trouble locating and using the batch file
`vcvarsall.bat`. I tried adding directories to my PATH variable and tried copying the batch file to many 
different locations. None of this worked, but luckily `setuptools` had no such problems. 

Now, to make our module `hello`, we will need to create two files: 
1. Our C source code `modulehello.c`; note the convention for the name: module<name>.c. 
2. A python file to setup the build process, `setup.py`.

## Making `setup.py`

Let us first take a look at `setup.py`.
``` python
# setup.py

from setuptools import setup, Extension

# Our module is an extension module since it is C code.

myModule = Extension('hello', sources = ['modulehello.c'])

# The package name doesn't affect the name of the module created.

setup(name = 'myPackageName', ext_modules = [myModule])
```
We tell the Python interpreter that we have an extension module named `hello` and that its source is
`modulehello.c`. Then we set up a package to hold the module. Now, we aren't really going to be making a package;
so I simply made the name `myPackageName` which you could replace with anything that is more related to your project.

## Making `modulehello.c`

First let's include the appropriate headers.
``` c
/* modulehello.c */

#include <Python.h>
#include <stdio.h>
```

Next, let's define the C function `hello_sayHello()` that will be responsible for printing our message.
``` c
/* The C definition of our function to print Hello World. By convention
the name is <module name>_<function name>, but this isn't necessary. */

static PyObject*
hello_sayHello(PyObject *self, PyObject *args) {

    printf("Hello World using printf");

    /* We wish for our function to return None, the Python
    equivalent of returning void. We have to do the following
    to return None. */

    Py_INCREF(Py_None);
    return Py_None; 
}
```
Note that we have made the function return a `None` python object. Now, we don't call this function directly from Python. We need to do some work to tell Python which 
functions are included in the module. 

First we create an array of the information for each of the methods in 
our module. The array ends in a sentinel value to indicate the end of the array.
``` c
/* This array lists all of the methods we are putting into our module. Take
note of the sentinel value at the end to indicate the ending of the array. */

static PyMethodDef HelloMethods[] = {
    {"sayHello", hello_sayHello, METH_VARARGS, "Print Hello World"},
    {NULL, NULL, 0, NULL} /* The sentinel value. */
};
```

Next, using the array of methods, we create a structure holding information on our module.
``` c
/* This declares set-up information for our module.*/

static struct PyModuleDef hellomodule = {

    PyModuleDef_HEAD_INIT,
    "hello",
    NULL, /*This is for documentation, which we won't use; so it is NULL. */
    -1,
    HelloMethods
};
```

Finally, we need to make a function for initializing our module using the module information structure.
``` c
/* Function to initialize the module. Note the necessary name format of
PyInit_<module name>. */

PyMODINIT_FUNC
PyInit_hello(void) {

    return PyModule_Create(&hellomodule);
}
``` 

## Building the Module

Now open a terminal in the current director and run the command
```
python setup.py build_ext --inplace
```
We use `build_ext` to tell Python to build the extension module, and we use `--inplace` to tell Python
to make a copy of the output in the current directory. This also creates a sub-directory `build` where
the original copy of the output of the build is stored. Using `--inplace` makes sure we don't have to
mess with this sub-directory structure.

## Using the Module

Now we need to make a python program to test our module. Create the file `test.py` simply containing the 
following
``` python
# test.py

import hello 

hello.sayHello()
```
Okay, then simply run the program from a terminal with
```
python test.py
```
Then we get our expected output. 
```
Hello World using printf
```

## [Download `setup.py` Here]({{site . url}}/assets/2018-06-13-code/setup.py)
## [Download `modulehello.c` Here]({{site . url}}/assets/2018-06-13-code/modulehello.c)
## [Download `test.py` Here]({{site . url}}/assets/2018-06-13-code/test.py)
