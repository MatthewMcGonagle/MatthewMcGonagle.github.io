---
layout: post
date: 2018-06-15
title: C Extension Module for Python, Quicksort
---

## [Download `moduleCSort.c` Here]({{site . url}}/assets/2018-06-15-files/moduleCSort.c)
## [Download `setup.py` Here]({{site . url}}/assets/2018-06-15-files/setup.py)
## [Download `benchmark.py` Here]({{site . url}}/assets/2018-06-15-files/benchmark.py)

In this post we will look at implementing quicksort in C as an extension module for Python. We will also
construct an implementation of quicksort in pure Python and get the benchmarks of both. The inputs for the algorithms
will be a Python list of Python `Int` quantities (these Python Int correspond to `long` in C).

The C version of quicksort will be accessible in Python by importing the module `cSort`, and the
pure Python version will be in the module `pySort`. The different versions of the quicksort functions are
`cSort.doQuickSort()` and `pySort.doQuickSort()`. Finally, we will note that we implement one of the most simple versions of quicksort. We simply choose as our pivot the 
element at the beginning of the array. 

Before we dive into the implementation details, let us use the next section to take a look at how to use the functions and obtain benchmarks
their performance.

## Benchmarking `cSort` and `pSort` Quicksort Functions.

The benchmarks are perfomed in `benchmark.py`; before we get started we need to import the appropriate modules and set up
the seed for our random number generator.

``` python
'''
benchmark.py

Get benchmarks for the C version of quicksort we have written vs the
pure Python version of quicksort we have written.
'''

import cSort # Has our C version of quicksort.
import pySort # Has our pure Python version of quicksort.
import random # For generating random test arrays.
import timeit # For getting our benchmark times.
import numpy as np # For numerical manipulation of our benchmark results.
import matplotlib.pyplot as plt # For plotting our results.

# Seed for consistency.

random.seed(20180614)
```

Before we benchmark our functions, let's try them out. Let's first generate a list
of 40 elements that includes two copies of every number from 0 to 19. The elements are put
into random order.
```python

# Set up a random list to test our sorting functions.

nItems = 20 
myList = [i for i in range(nItems)]
myList = myList + myList
myList = random.sample(myList, len(myList)) 
print(myList)

```
We get the following output:
```
[13, 3, 6, 16, 19, 19, 6, 14, 17, 5, 10, 2, 7, 18, 12, 10, 7, 15, 8, 16, 14, 3, 2, 17, 0, 12, 8, 4, 1, 4, 13, 9, 0, 11, 1, 18, 9, 5, 15, 11]
```

Now let's test both quicksort functions.
``` python
# Test the sorting functions.
   
cSortedList = cSort.doQuickSort(myList)
print('Result of cSort is\n', cSortedList)

pySortedList = pySort.doQuickSort(myList)
print('Results of pySort is\n', pySortedList)
```
We get the following output:
```
Result of cSort is
 [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19]
Results of pySort is
 [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19]
```
Great! So we see that the functions are working.

So now let us get the benchmarks; to do so we will be making use of the `timeit` module. To make this as painless as possible, let's 
make a function to handle using `timeit` to do so.
``` python
def getBenchmark(nItems, sortName, seed, nRepeat = 100, nArrays = 100): 
    '''
    Get the benchmark for a particular sorting algorithm, size of list, and random seed. The
    random seed is used to guarantee consistencey between sorting algorithms. The benchmark
    is found using timeit for different random arrays of size nItems. Then we take the
    average benchmark for these arrays.

    Parameters
    ----------
    nItems : Int
        The size of the list to get the benchmark for.
    
    sortName : String
        The name of the sorting function to use.

    seed : Int
        The seed to supply to random.seed().

    nRepeat : Int
        The number of times to repeat the sorting algorithm on single instance of an unsorted array.

    nArrays : Int
        The number of different random arrays to run the algorithm on.

    Returns
    -------
    Float
        The average milliseconds the sorting routine took.
    '''

    # The code to call the sorting algorithm.

    code = 'sortedList = ' + sortName + '(myList)' 
  
    # Set up the random seeds for each array for consitency. First
    # use the seed provided to generate new random seeds for each array.

    random.seed(seed) 
    seeds = np.random.randint(0, seed + nArrays, size = nArrays)

    # Now for each array, use its seed in the setup and use timeit
    # to benchmark the array.

    arrayResults = []

    for subSeed in seeds:
        setup = ( 'import cSort\n' +
                  'import pySort\n' + 
                  'import random\n' + 
                  'gc.enable()\n' +
                  'random.seed(' + str(subSeed) + ')\n' + 
                  'myList = range(' + str(nItems) + ')\n' +
                  'myList = random.sample(myList, ' + str(nItems) + ')' )

        result = timeit.timeit(code, setup, number = nRepeat)
        arrayResults.append(result)

    # Get the average benchmark for the different arrays.

    benchmark = np.array(arrayResults).mean() / nRepeat
    return benchmark
```
Note that to keep the random array generation consistent across the two different algorithms, we will give this function a seed. We will use different
seeds for different sizes of arrays however. Also note we take an average for random number of arrays for the given size.

Next let's run our benchmarks.
``` python
# Now let's set up our benchmarks.
 
benchmarkSizes = range(10, 200, 15)
benchmarks = {'cSort' : [], 'pySort' : []}

# Run the benchmarks.

for module in benchmarks.keys(): 
   
    sortName = module + '.doQuickSort' 
    print(sortName, 'Benchmarks')
    seeds = (20180614 + i for i in benchmarkSizes)

    for nItems, seed in zip(benchmarkSizes, seeds): 
        benchmark = getBenchmark(nItems = nItems, sortName = sortName, seed = seed) 
        benchmarks[module].append(benchmark)
        print('nItems = ', nItems, '\tbenchmark = ', benchmark)
```
We get the following output:
```
pySort.doQuickSort Benchmarks
nItems =  10    benchmark =  1.16245533823e-05
nItems =  25    benchmark =  4.17214226867e-05
nItems =  40    benchmark =  7.37319926875e-05
nItems =  55    benchmark =  0.000112821455175
nItems =  70    benchmark =  0.00015827261649
nItems =  85    benchmark =  0.000195776460431
nItems =  100   benchmark =  0.000244394713894
nItems =  115   benchmark =  0.000289817097458
nItems =  130   benchmark =  0.000371542436129
nItems =  145   benchmark =  0.000497422753056
nItems =  160   benchmark =  0.000431312916789
nItems =  175   benchmark =  0.000493551928445
nItems =  190   benchmark =  0.000608073165754
cSort.doQuickSort Benchmarks
nItems =  10    benchmark =  4.49035550588e-07
nItems =  25    benchmark =  7.61445866841e-07
nItems =  40    benchmark =  1.08439618239e-06
nItems =  55    benchmark =  1.41836020507e-06
nItems =  70    benchmark =  1.79329995535e-06
nItems =  85    benchmark =  2.0900779131e-06
nItems =  100   benchmark =  2.43525302116e-06
nItems =  115   benchmark =  2.81773262114e-06
nItems =  130   benchmark =  3.47776553594e-06
nItems =  145   benchmark =  4.69270770213e-06
nItems =  160   benchmark =  5.24520893871e-06
nItems =  175   benchmark =  5.71970628536e-06
nItems =  190   benchmark =  6.17000505683e-06
```
Let's get a plot of these benchmarks:

![Plot of benchmark times]({{site . url}}/assets/2018-06-15-files/times.svg)

Let's take a look at the ratios:

![Plot of ratios]({{site . url}}/assets/2018-06-15-files/ratios.svg)

We can see that the C quicksort is much faster than the pure Python quicksort.

## The `cSort` Module

Before we begin, make sure to include `Python.h`.
``` c
/* moduleCSort.c */

#include <Python.h>
```

First, we make the C source code for our module extension in the file `moduleCSort.c`. Our plan
is take the list of Python list of Python Int and turn it into a simple array of `long`. Then we will
sort this array and then convert the sorted array into a Python List. 

First let us make a function to help up convert the Python List into a C type array of `long`.
Note that a Python list is of C type `PyList` and a Pyton Int is of C type `PyLong`. 
However, the list is given to us as a pointer to a generic Python Object, that is `PyObject*`.
``` c 
/**
    \brief Converts a PyList of PyLong to a C array of long.
    
    Function takes a PyList of PyObject pointers to objects
    of type PyLong and converts it to a C array of long.
    
    @param list Pointer to the python list. The python list should be
                of type PyList. Check this before calling getArray. 
    @param size Pointer to a variable that will hold the size of the list. 
    
    @return Pointer to the start of the array.
*/
static
long* getArray(PyObject *list, Py_ssize_t *size) {

    Py_ssize_t i, nItems;
    PyObject *item;
    long *newarray;

    /* Get the size of the list */

    nItems = PyList_Size(list);
    if(nItems < 0)
        return NULL;
    *size = nItems;

    /* Allocate the array and get its contents from the list. */

    newarray = malloc(nItems * sizeof(long));

    for (i = 0; i < nItems; i++) {

        item = PyList_GetItem(list, i); 

        if(!PyLong_Check(item)) {
            free(newarray);
            PyErr_SetString(PyExc_TypeError, "Item NOT a PyLong!");
            return NULL;
        }

        newarray[i] = PyLong_AsLong(item);
    }

    return newarray;
}
```

Now, before we make our quicksort, it will be convenient to make a function to swap two 
elements in our array.
``` c
/**
    Swaps two positions in an array of long
    
    @param array Pointer to the beginning of the array.
    @param loc1 The first location to switch. 
    @param loc2 The second location to switch.

*/
static
void swap(long *array, Py_ssize_t loc1, Py_ssize_t loc2) {

    long temp = array[loc1];
    array[loc1] = array[loc2];
    array[loc2] = temp;

}
```

Next, let's make the function that performs quicksort on a C array.
``` c
/**
    Sort an array using quicksort.
    
    Use a simple version of quicksort to sort an array. We just use the
    first element as the pivot. We modify the array in place and use
    recursion.
    
    @param array Pointer to the beginning of the array
    @param size The size of the array.
    
*/
static
void doQuickSort(long *array, Py_ssize_t size) {

    long pivotVal;
    Py_ssize_t pivotLoc = 0, searchLoc = 1;

    /* First handle the trivial cases. */

    if(size < 2)
        return;

    else if(size == 2) {
        if(array[0] > array[1])
            swap(array, 0, 1);
        return;
    }

    /* Now do pivot. */

    pivotVal = array[pivotLoc];

    for(searchLoc = 1; searchLoc < size; searchLoc++) {

        if(array[searchLoc] <= pivotVal) {
    
            swap(array, searchLoc, pivotLoc + 1);
            swap(array, pivotLoc, pivotLoc + 1);
            pivotLoc++;
        }
    }

    /* Now recurse on the sub-arrays to the left and right of the pivot. */

    doQuickSort(array, pivotLoc);
    doQuickSort(&(array[pivotLoc+1]), size - pivotLoc - 1); 

}
```

Now we will need a function that takes a C array of `long` and builds a `PyList` of `PyLong`.
``` c
/**
    Build a PyList of PyLong from a C array of long.

    @param array Pointer to the beginning of the array.
    @param size The size of the array.
    
    @return Pointer to the PyList of PyLong that is constructed.
*/
static PyObject*
buildPyList(long* array, Py_ssize_t size) {

    PyObject *list, *value;
    Py_ssize_t i;
    int problem;

    /* Convert the array to a Python list of PyLong */

    /* First, allocate a new PyList */

    list = PyList_New(size);
    if(list == NULL) {
        PyErr_SetString(PyExc_TypeError, "Can't make new list\n");
        return NULL;
    }

    /* Now store the values in the array in the PyList */

    for(i = 0; i < size; i++) {
        value = PyLong_FromLong(array[i]);
        if(value == NULL) {
            PyErr_SetString(PyExc_TypeError, "Couldn't make PyLong");
            return NULL;
        }

        problem = PyList_SetItem(list, i, value);
        if(problem) {
            PyErr_SetString(PyExc_TypeError, "Couldn't set item in list.");
            return NULL;
        }
    } 

    return list;
 
}
```

Finally we construct the function that will be made callable from Python.
``` c
/**
    \brief Sort a PyList of PyLong using quicksort.
    
    Sorts a PyList of PyLong. First the list is converted to a C array of long.
    Then we do a quicksort on the array and convert the result into a PyList.

    @param self Pointer passed in by Python.
    @param args Pointer to arguments tuple passed in by Python. The arguments should simply
                be a single PyList of PyLong.

    @return Pointer to the new sorted PyList that is created.
*/
static PyObject*
csort_doQuickSort(PyObject *self, PyObject *args) {

    int parseOK; 
    Py_ssize_t size;
    PyObject *list;
    long *array;

    /* Get the list and check its type */

    parseOK = PyArg_ParseTuple(args, "O", &list); 
    if(!parseOK){
        PyErr_SetString(PyExc_TypeError, "Problem Parsing Arguments!\n");
        return NULL;
    }

    if(! PyList_Check(list)) {
        PyErr_SetString(PyExc_TypeError, "Object isn't a list!\n");
        return NULL;
    }

    /* Get list as an array. */

    array = getArray(list, &size);
    if(array == NULL) 
        return NULL;

    /* Now do a quick sort of the array. */

    doQuickSort(array, size);

    /* Convert the array to a Python list of PyLong */

    list = buildPyList(array, size);
    

    /* Garbage Collection */

    free(array);

    /* Return the sorted Python List */

    return list;
}
```

Lastly, we set up our Python module information.
``` c
/**
    List of the methods that will be callable from Python for this module.
*/
static PyMethodDef myCSortMethods[] = {

    /* First item is a function for debugging
       purposes. It is included in the source code, but we aren't discussing
       it in our post; so it is commented out.
       {"printList", csort_printList, METH_VARARGS, "Simply Print a List"}, */ 

    {"doQuickSort", csort_doQuickSort, METH_VARARGS, "Sort a list"},
    {NULL, NULL, 0, NULL} /* Sentinel value */
};

/**
    Module defintion information.
*/
static struct PyModuleDef cSortModule = {

    PyModuleDef_HEAD_INIT,
    "cSort",
    NULL,
    -1,
    myCSortMethods
    
};

/**
    Module initialization function.
*/
PyMODINIT_FUNC
PyInit_cSort(void) {

    return PyModule_Create(&cSortModule);
}
```

## Building the `cSort` Module

To build the `cSort` module from the C source code in `moduleCSort.c`, we first need to make `setup.py`.
``` python
'''
setup.py

The module name will be csort.
'''

from setuptools import Extension, setup

myModule = Extension('cSort', sources = ['moduleCSort.c'])
setup(name = 'myCSortPackage', ext_modules = [myModule])
```

Then, from the terminal, we run:
```
python setup.py build_ext --inplace
```

Now we are done making the `cSort` module; it should be accessible by simply importing it in Python using `import cSort`.

## The `pySort` Module

Now, let's make the pure Python implementation of quicksort. We aren't going to use `numpy`, because `numpy` is a C extension itself. 
We will use pure Python lists, hence why the performance isn't very good.

This implementation is more simple than the C version, so here is all of the code:
``` python
'''
pySort.py
'''

def doQuickSort(myList):
    ''' 
    Does a quicksort on a copy of the list.

    Parameters
    ----------
    myList : Python list of Int
        The list to sort.

    Returns
    -------
        A sorted copy of the list.
    '''
    newList = myList.copy()
    doQuickSortInPlace(newList, len(newList), offset = 0)
    return newList

def doQuickSortInPlace(myList, size, offset):
    '''
    Function to perform quicksort in place on a subrange of a list.

    Parameters
    ----------
    myList : Python list of Int
        The list we are sorting.

    size : Int
        The size of the sub-list we are sorting.

    offset : Int
        The index of the first element in the sub-list. So if
        offset is 5, then the first element of the sub-list is
        myList[5].
    '''

    # First handle trivial cases

    if size < 2:
        return

    if size == 2:
        if myList[offset] > myList[offset + 1]:
            swap(myList, offset, offset + 1)
        return

    # Now do pivoting.
    
    pivotVal = myList[offset] 
    pivotLoc = offset 
    for searchLoc in range(offset + 1, offset + size):
        if myList[searchLoc] <= pivotVal:
            swap(myList, searchLoc, pivotLoc + 1)
            swap(myList, pivotLoc, pivotLoc + 1) 
            pivotLoc += 1

    # Now recurse on list to the left of pivot and list to 
    # the right of the pivot.

    leftSize = pivotLoc - offset
    doQuickSortInPlace(myList, leftSize, offset)
    doQuickSortInPlace(myList, size - leftSize - 1, offset = pivotLoc + 1)

def swap(myList, loc1, loc2):
    '''
    Function to swap elements in a list of numbers (so that copy is unnecessary).

    Parameters
    ----------
    myList : List of Int
        The list holding the elements to swap.

    loc1 : Int
        The index of the first element to swap.

    loc2 : Int
        The index of the second element to swap.
    '''
    temp = myList[loc1]
    myList[loc1] = myList[loc2]
    myList[loc2] = temp
```

## [Download `moduleCSort.c` Here]({{site . url}}/assets/2018-06-15-files/moduleCSort.c)
## [Download `setup.py` Here]({{site . url}}/assets/2018-06-15-files/setup.py)
## [Download `benchmark.py` Here]({{site . url}}/assets/2018-06-15-files/benchmark.py)
