/* moduleCSort.c */

#include <Python.h>
#include <stdio.h>

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

/**
    \brief Print the contents of a PyList of PyLong.

    Prints the contents of a PyList of PyLong. Mostly used for testing purposes
    to make sure we are using the Python API correctly. First converts the
    PyList to a C array of long before using printf to print the contents.

    @param self Object pointer passed in by Python
    @param args Object pointer for arguments tuple passed in by python. Should be a single PyList of PyLong.

    @return A Python None object.
*/
static PyObject*
csort_printList(PyObject *self, PyObject *args) {

    int parseOK; 
    Py_ssize_t i, size;
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

    /* Now print out the contents of the array */

    printf("[");
    for(i = 0; i < size - 1; i++) {
       printf("%ld, ", array[i]); 
    }
    printf("%ld]", array[i]);

    /* Garbage Collection */

    free(array);
 
    /* Return None */

    Py_INCREF(Py_None);
    return Py_None;
}

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
