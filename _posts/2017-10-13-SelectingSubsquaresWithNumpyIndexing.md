---
layout : post 
title : "Selecting Sub-squares with Numpy Indexing"
date : 2017-10-13 
---

## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

We will be looking at using advanced indexing in Numpy to select all possible sub-squares of some given side length `sidel` inside a two-dimensional numpy array `A` (i.e. `A` is a matrix). For example, consider the following 4x5 matrix; each 3x3 sub-square of this matrix has been color coded and labeled.
![Examples of Sub-squares of a Matrix]( {{ site . url }}/assets/2017-10-13-SubsquarePic.svg)

## Constructing an Array Using Broadcasting and Indexing

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

## [Download the Source Code for this Post]( {{ site . url }}/assets/2017-10-13-SubsquaresNumpy.py)

