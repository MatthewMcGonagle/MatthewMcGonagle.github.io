---
layout : post 
title : "Selecting Sub-squares with Numpy Indexing"
date : 2017-10-13 
tags: [Python, Numpy]
---

## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

We will be looking at using advanced indexing in Numpy to select all possible sub-squares of some given side length `sidel` inside a two-dimensional numpy array `A` (i.e. `A` is a matrix). For example, consider the following 4x5 matrix; each 3x3 sub-square of this matrix has been color coded and labeled.
![Examples of Sub-squares of a Matrix]( {{ site . url }}/assets/2017-10-13-SubsquarePic.svg)

For the last two sections at the end of this post, we will get a benchmark of this method vs two methods of list comprehensions. For the last section, we will see that there are situations where the indexing method is more optimal for execution time.
## Constructing an Array Using Broadcasting and Indexing

First, we will import numpy by the standard convention of calling it `np`.
``` python
import numpy as np
```

Our matrix `A` will have dimensions given by
``` python
nRows = 5
nCols = 10
```
We use `np.arange` combined with Numpy broadcasting and indexing to create our matrix `A`.

``` python
row0 = np.arange(nCols)
col0 = np.arange(nRows)
A = row0 + nCols * col0[:, np.newaxis] 
```
This creates the following matrix:
```
A =
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]]
```

Let us briefly comment on how this works. Please consider the following figure for a visual description of what is bappening, but please also consider the actual description after the figure. 

![Visualization of the construction of the matrix A]({{ site . url }}/assets/2017-10-13-ArrayConstruction.svg)

Now, `row0` has shape `(10)`, and `col0` has shape `(5)`. For the computation of `A`, the expression `col0[:, np.newaxis]` changes the shape of `col0` from `(5)` to `(5, 1)`. Now for the calculation of `A`, the summands (expressions on the left and right of `+`) have dimensions that don't match. The left summand is initially one-dimensional and the right summand is two-dimensional. To fix this, numpy automatically adds an axis to row0 to make the left summand also two-dimensional.

However, the sizes of the dimensions for the summands still don't match. That is, they still don't have the same shape; the left summands has shape `(1, 10)`, the right summand has shape `(5, 1)`, and `(1, 10) != (5, 1)`. However, when dimension sizes don't match and for each axis only one size is greater than one, numpy can automatically fill in the rest to make all dimension sizes the same. For example, in our case the sizes of axis 0 are 1 and 5. These are different, but only one of them isn't 1. Therefore, numpy may make 5 copies of the array of shape `(1, 10)` along axis 0 to make an array of shape `(5, 10)`. Similarly, for axis 1, the array of shape `(5, 1)` may be copied along axis 1 to make a shape `(5, 10)`. 

Then the summands can be combined to give `A`.

A note on why we chose to use `col0[:, np.newaxis]` instead of `row0[:, np.newaxis]`. We see that `col0` represents the first column of `A`. That as we change the index in `col0`, this represents moving down a particular column of `A`. This is accomplished by changing `i` in `A[i, j]`. So we `col0[:, np.newaxis]` gives a shape `(5, 1)`. Indexing over this is similar to indexing over `i` in `A[i, j]`.

Similarly, `row0[np.newaxis, :]` is similar to indexing over `j` in `A[i, j]`.

## Selecting Sub-squares

Now, let use assume our sub-square side length `sidel` is valid, i.e. it's no more than `nRows` or `nCols`. For the purposes of an actual example, let's set 
``` python
sidel = 3
```
Now, let's compute an array holding all `sidel` by `sidel` sub-squares inside `A`. At first, we will index each sub-square with a two-dimensional index. Also, let us first find the index of the upper left corner of each sub-square.

We store the row index of each corner in an array `cornersRow` and the column index in an array `cornersCol`. For later use, we will require that the arrays have four dimensions, so we will set that up now.

``` python
# We remove sidel - 1 points from rows and cols, because these aren't corner pieces.
cornersRow = np.arange(nRows - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]
cornersCol = np.arange(nCols - sidel + 1)[np.newaxis, :, np.newaxis, np.newaxis]
corners = A[cornersRow, cornersCol]
```

The array `corners` has shape `(3, 8, 1, 1)`. In fact , `corners` may be written as 
```
corners = 
[[ [[ 0]] [[ 1]] [[ 2]] [[ 3]] [[ 4]] [[ 5]] [[ 6]] [[ 7]] ]
 [ [[10]] [[11]] [[12]] [[13]] [[14]] [[15]] [[16]] [[17]] ]
 [ [[20]] [[21]] [[22]] [[23]] [[24]] [[25]] [[26]] [[27]] ]]
```  
Note that the way we have written `corners` will differ from the output of the simple print statement `print(corners)`. We have written it this way, because it is more clear; Also the structure is the same.

Now, let's find a 2d array of sub-squares. Again, first let us find the rows and cols of each entry for each sub-square.

``` python
subsquareRow = cornersRow + np.arange(sidel)[:, np.newaxis]
subsquareCol = cornersCol + np.arange(sidel)
```
We see that `subsquareRow.shape` is `(3, 1, 3, 1)` and `subsquareCol.shape` is `(1, 8, 1, 3)`. These shapes will be broadcast into the full shape `(3, 8, 3, 3)` when we make use of them together. Now let's get the sub-squares of `A`.

```
subsquares = A[subsquareRow, subsquareCol]

```

We see that `subsquares.shape` is `(3, 8, 3, 3)`; this shape may be interpreted as 3x8 grid of 3x3 sub-squares. Showing `subsquares` in this form we have 
```
subsquares = 
[[ 0  1  2]   [[ 1  2  3]   [[ 2  3  4]   [[ 3  4  5]   [[ 4  5  6]   [[ 5  6  7]   [[ 6  7  8]   [[ 7  8  9]
 [10 11 12]    [11 12 13]    [12 13 14]    [13 14 15]    [14 15 16]    [15 16 17]    [16 17 18]    [17 18 19]
 [20 21 22]]   [21 22 23]]   [22 23 24]]   [23 24 25]]   [24 25 26]]   [25 26 27]]   [26 27 28]]   [27 28 29]]

[[10 11 12]   [[11 12 13]   [[12 13 14]   [[13 14 15]   [[14 15 16]   [[15 16 17]   [[16 17 18]   [[17 18 19]
 [20 21 22]    [21 22 23]    [22 23 24]    [23 24 25]    [24 25 26]    [25 26 27]    [26 27 28]    [27 28 29]
 [30 31 32]]   [31 32 33]]   [32 33 34]]   [33 34 35]]   [34 35 36]]   [35 36 37]]   [36 37 38]]   [37 38 39]]

[[20 21 22]   [[21 22 23]   [[22 23 24]   [[23 24 25]   [[24 25 26]   [[25 26 27]   [[26 27 28]   [[27 28 29]
 [30 31 32]    [31 32 33]    [32 33 34]    [33 34 35]    [34 35 36]    [35 36 37]    [36 37 38]    [37 38 39]
 [40 41 42]]   [41 42 43]]   [42 43 44]]   [43 44 45]]   [44 45 46]]   [45 46 47]]   [46 47 48]]   [47 48 49]]
```

So, `subsquares[i,j]` is the sub-square at the ith row and jth column in this grid, and `subsquares[i,j]` has shape `(3, 3)`!

## Flattening into an One-dimensional Array of Sub-squares

So, we now have a 3x8 grid of 3x3 sub-squares. How do we turn this into a one-dimensional array of 3x3 sub-squares? This part is much more simple than the previous parts. For this, we simply reshape `subsquares`.

``` python
# Now convert to single list of subsquares.
subsquareList = subsquares.reshape(-1, sidel, sidel)
```

We see that `subsquareList` has shape `(24, 3, 3)`. Let's take a look at the first four sub-squares in `subsquareList`, i.e. `subsquareList[:4, :, :]`. 
```
subsquareList[:4, :, :] = 
[[[ 0  1  2]
  [10 11 12]
  [20 21 22]]

 [[ 1  2  3]
  [11 12 13]
  [21 22 23]]

 [[ 2  3  4]
  [12 13 14]
  [22 23 24]]

 [[ 3  4  5]
  [13 14 15]
  [23 24 25]]]
```
So we see that the order the sub-squares are added to `subsquareList` is left to right and top to bottom as we would expect.

## Adding Over Sub-squares with Certain Weights

Suppose that for each sub-square, we wish to sum up the entries of the sub-square with certain weights given by a 3x3 array. For our example, we will use
``` python
weights = [[ 1, 0, -1],
           [ 0, 1,  0],
           [ 2, 0,  0]]
```
So for example, summing over the first sub-square 
```
subsquares[0,0] = 
[[ 0  1  2]
 [10 11 12]
 [20 21 22]]
```
with these weights gives `0 + 0 - 2 + 0 + 11 + 0 + 40 + 0 + 0 = 49`. To accomplish this with our multi-dimensional arrays, we will use `numpy.tensordot`. 

``` python
subsquareSums = np.tensordot(subsquares, weights, axes = [[2, 3], [0, 1]])
```

We see that `subsquareSums.shape` is `(3, 8)`. That is, it returns a number for each sub-square in the 3x8 grid of sub-squares. In fact, we have that `subsquareSums` is given by 
```
subsquareSums = 
[[ 49  52  55  58  61  64  67  70]
 [ 79  82  85  88  91  94  97 100]
 [109 112 115 118 121 124 127 130]]
```

## Timing the Weighted Sum

Now let's see how much faster these indexing methods are at computing weighted sums than using a reasonable list comprehension (with a little numpy).

First, we need to import the `timeit` and `random` modules:
``` python
import timeit
import random
```

Next, let's put our broadcasting methods into a single function:
``` python
def indexingMethod(X, weights, sidel):

    nRows, nCols = X.shape
    cornersRow =  np.arange(nRows - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    cornersCol = np.arange(nCols - sidel + 1)[np.newaxis, :, np.newaxis, np.newaxis]

    subsquareRow = cornersRow + np.arange(sidel)[:, np.newaxis]
    subsquareCol = cornersCol + np.arange(sidel)
    subsquares = X[subsquareRow, subsquareCol]
    subsquareSums = np.tensordot(subsquares, weights, axes = [[2, 3], [0, 1]])
    return subsquareSums

``` 

Next, let's construct a function that uses a list comprehension to index over our final grid of sums. The sums themselves will still be computed using `np.tensordot`.
``` python
def listComprehensionMethod(X, weights, sidel):
    nRows, nCols = X.shape
    subsquareSums = [[ np.tensordot(X[i : i + sidel, j : j + sidel], weights, [[0, 1], [0, 1]])
                       for j in np.arange(nCols - sidel + 1) ] 
                       for i in np.arange(nRows - sidel + 1) ]
    return np.array(subsquareSums)
```

There is also another way to use list comprehensions to do the weighted sum. The above method should be faster for larger sub-squares. The next method should be faster for smaller sub-squares.
``` python
def listComprehensionMethod2(X, weigths, sidel):
    nRows, nCols = weights.shape
    xRows, xCols = X.shape
    result = np.zeros((xRows - sidel + 1, xCols - sidel + 1))
    for i in range(nRows):
        for j in range(nCols):
            result += X[i : i + xRows - sidel + 1, j : j + xCols - sidel + 1] * weights[i, j]
    return result

```

Now let's time these functions. We will test them on a random matrix `B` of dimensions 100x100, and we will use a random weight matrix of with both dimensions having size sidel. We will test these for different values of the sub-square sidelength. 

``` python
# Get benchmarks for different methods.      
random.seed(1013)  
B = [[random.randint(0, 10) for j in range(100)]
        for i in range(100)]
B = np.array(B)

results = {'indexMethod' : [], 'list1' : [], 'list2' : []}
search = [3, 25, 50, 75, 97]
for sidel in search: 

    weights = [[random.randint(-2, 2) 
                    for j in range(sidel) ]
                    for i in range(sidel) ]
    weights = np.array(weights)
    
    print("\nbenchmarks for sidel = ", sidel)

    benchmark = timeit.timeit(lambda : indexingMethod(B, weights, sidel), number = 50)
    print("indexingMethod benchmark = ", benchmark)
    results['indexMethod'].append(benchmark)

    benchmark = timeit.timeit(lambda : listComprehensionMethod(B, weights, sidel), number = 50)
    print("listComprehensionMethod benchmark = ", benchmark)
    results['list1'].append(benchmark)

    benchmark = timeit.timeit(lambda : listComprehensionMethod2(B, weights, sidel), number = 50)
    print("listComprehensionMethod2 benchmark = ", benchmark)
    results['list2'].append(benchmark)

    # Double check that the methods are giving the same result.
    print('Check index method == list comprehension 1 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod(B, weights, sidel)))
    print('Check index method == list comprehension 2 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod2(B, weights, sidel)))


```
We get the following results:
```
sidel   index   list1   list2
3        0.034   4.796   0.014
25       1.018   3.107   0.639
50       1.893   1.720   1.370
75       1.064   0.537   1.491
97       0.037   0.016   1.460
```
A graph of these results (made with matplotlib) is:

![Graph of Benchmarks for Constant Weights]({{ site . url }}/assets/2017-10-13-Benchmarks.svg)

So we see that for the case of weights that are independent of the sub-square, the index method does worse than one of the other list comprehensions.

## Timing for Weights Varying by the Sub-square

Now let's try benchmarking for the case when the weights are depending on the sub-square. So now, `weights.shape` is `(100 - sidel + 1, 100 - sidel + 1, sidel, sidel)`. First, we have to redefine our functions to benchmark to use this new shape:
``` python
# Now look at benchmarks for weights that depend on sub-square.

def indexingMethod(X, weights, sidel):

    nRows, nCols = X.shape
    cornersRow =  np.arange(nRows - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    cornersCol = np.arange(nCols - sidel + 1)[np.newaxis, :, np.newaxis, np.newaxis]

    subsquareRow = cornersRow + np.arange(sidel)[:, np.newaxis]
    subsquareCol = cornersCol + np.arange(sidel)
    subsquares = X[subsquareRow, subsquareCol]
    subsquareSums = np.sum(subsquares * weights, axis = (2, 3)) 
    return subsquareSums

def listComprehensionMethod(X, weights, sidel):
    nRows, nCols = X.shape
    subsquareSums = [[ np.tensordot(X[i : i + sidel, j : j + sidel], weights[i, j], [[0, 1], [0, 1]])
                       for j in np.arange(nCols - sidel + 1) ] 
                       for i in np.arange(nRows - sidel + 1) ]
    return np.array(subsquareSums)

def listComprehensionMethod2(X, weigths, sidel):
    sqRos, sqCols, nRows, nCols = weights.shape
    xRows, xCols = X.shape
    result = np.zeros((xRows - sidel + 1, xCols - sidel + 1))
    for i in range(nRows):
        for j in range(nCols):
            result += X[i : i + xRows - sidel + 1, j : j + xCols - sidel + 1] * weights[:, :, i, j]
    return result
```

Again, let's get some benchmarks for different sidelengths of the sub-squares.

``` python
results = {'indexMethod' : [], 'list1' : [], 'list2' : []}
search = [3, 25, 35, 50, 75, 97]
for sidel in search: 

    weights = [[random.randint(-2, 2) 
                    for j in range(sidel) ]
                    for i in range(sidel) ]
    weights = np.array(weights)
    weights = weights + np.arange(100 - sidel + 1)[:, np.newaxis, np.newaxis]
    weights = weights - np.arange(100 - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]

    
    print("\nbenchmarks for sidel = ", sidel)

    benchmark = timeit.timeit(lambda : indexingMethod(B, weights, sidel), number = 50)
    print("indexingMethod benchmark = ", benchmark)
    results['indexMethod'].append(benchmark)

    benchmark = timeit.timeit(lambda : listComprehensionMethod(B, weights, sidel), number = 50)
    print("listComprehensionMethod benchmark = ", benchmark)
    results['list1'].append(benchmark)

    benchmark = timeit.timeit(lambda : listComprehensionMethod2(B, weights, sidel), number = 50)
    print("listComprehensionMethod2 benchmark = ", benchmark)
    results['list2'].append(benchmark)

    print('Check index method == list comprehension 1 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod(B, weights, sidel)))
    print('Check index method == list comprehension 2 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod2(B, weights, sidel)))

```

The results of this are:
```
sidel   index   list1   list2
3        0.038   4.929   0.015
25       1.394   3.235   1.433
35       2.065   2.672   2.906
50       2.543   1.790   4.298
75       1.445   0.563   1.724
97       0.053   0.016   1.331
```
A graph of these results (made with matplotlib) is:

![Graph of Benchmarks for Variable Weights]({{site . url}}/assets/2017-10-13-VarWeightsBenchmarks.svg)

Now we see that for the variable weights case, there is a range of sidelength where the numpy indexing method beats both of the list comprehensions! So there are cases where it is more optimal relative to time.

## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

