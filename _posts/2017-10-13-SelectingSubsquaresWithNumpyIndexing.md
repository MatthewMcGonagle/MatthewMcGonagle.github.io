---
layout : default
title : "Selecting Sub-squares with Numpy Indexing"
date : 2017-10-13 
---

## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

We will be looking at using advanced indexing in Numpy to select all possible sub-squares of some given side length `sidel` inside a two-dimensional numpy array `A` (i.e. `A` is a matrix).

## Constructing Array Using Broadcasting and Indexing

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

Let us briefly comment on how this works. Now, `row0` has shape `(10)`, and `col0` has shape `(5)`. For the computation of `A`, the expression `col0[:, np.newaxis]` changes the shape of `col0` from `(5)` to `(5, 1)`. Now for the calculation of `A`, the summands (expressions on the left and right of `+`) have dimensions that don't match. The left summand is initially one-dimensional and the right summand is two-dimensional. To fix this, numpy automatically adds an axis to row0 to make the left summand also two-dimensional.

However, the sizes of the dimensions for the summands still don't match. That is, they still don't have the same shape; the left summands has shape `(1, 10)`, the right summand has shape `(5, 1)`, and `(1, 10) != (5, 1)`. However, when dimension sizes don't match and for each axis only one size is greater than one, numpy can automatically fill in the rest to make all dimension sizes the same. For example, in our case the sizes of axis 0 are 1 and 5. These are different, but only one of them isn't 1. Therefore, numpy may make 5 copies of the array of shape `(1, 10)` along axis 0 to make an array of shape `(5, 10)`. Similarly, for axis 1, the array of shape `(5, 1)` may be copied along axis 1 to make a shape `(5, 10)`. 

Then the summands can be combined to give `A`.

A note on why we chose to use `col0[:, np.newaxis]` instead of `row0[:, np.newaxis]`. We see that `col0` represents the first column of `A`. That as we change the index in `col0`, this represents moving down a particular column of `A`. This is accomplished by changing `i` in `A[i, j]`. So we `col0[:, np.newaxis]` gives a shape `(5, 1)`. Indexing over this is similar to indexing over `i` in `A[i, j]`.

Similarly, `row0[np.newaxis, :]` is similar to indexing over `j` in `A[i, j]`.

## Selecting Sub-squares

Now, let use assume our sub-square side length `sidel` is valid, i.e. it's no more than `nRows` or `nCols`. For the purposes of an actual example, let's set 
``` python
sidel = 3
```
Now, let's compute an array holding all `sidel` by `sidel` sub-squares inside `A`. At first, we will index each sub-square with a two-dimensional index. 
## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

