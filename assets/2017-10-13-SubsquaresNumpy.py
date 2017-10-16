import numpy as np
import timeit 
import random

# First construct the array we will select sub-squares from.

nRows = 5
nCols = 10
row0 = np.arange(nCols)
col0 = np.arange(nRows)
A = row0 + nCols * col0[:, np.newaxis] 
print("A = ")
print(A)

sidel = 3

# We remove sidel - 1 points from rows and cols, because these aren't corner pieces.
cornersRow = np.arange(nRows - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]
cornersCol = np.arange(nCols - sidel + 1)[np.newaxis, :, np.newaxis, np.newaxis]
corners = A[cornersRow, cornersCol]
print("corners.shape = ", corners.shape)
print("\ncorners[:, :, 0, 0] = ")
print(corners[:, :, 0, 0])

subsquareRow = cornersRow + np.arange(sidel)[:, np.newaxis]
print('subsquareRow.shape = ', subsquareRow.shape)
subsquareCol = cornersCol + np.arange(sidel)
print('subsquareCol.shape = ', subsquareCol.shape)
subsquares = A[subsquareRow, subsquareCol]
print('subsquares.shape = ', subsquares.shape)
print(subsquares[0, 0], subsquares[0, 1])

nSqRows, nSqCols, sqSide1, sqSide2  = subsquares.shape
for i in range(nSqRows):
    for k in range(sidel):
        for j in range(nSqCols): # As we go across row, we are going across a row of subsquares.
            print(subsquares[i, j, k, :], ' ', end = "") 
        print()
    print()

# Now convert to single list of subsquares.
subsquareList = subsquares.reshape(-1, sidel, sidel)
print('subsquareList.shape = ', subsquareList.shape)
print('subsquareList[:4, :, :].shape = ', subsquareList[:4, :, :].shape)
print('subsquareList[:4, :, :] = ')
print(subsquareList[:4, :, :])

# Now let's add up entries in each sub-square with certain weights. The answer will be a 2d array of values.
weights = [[ 1, 0, -1],
           [ 0, 1,  0],
           [ 2, 0, 0]]
weights = np.array(weights)

subsquareSums = np.tensordot(subsquares, weights, axes = [[2, 3], [0, 1]])
print(subsquareSums.shape)
print(subsquareSums)

def indexingMethod(X, weights, sidel):

    nRows, nCols = X.shape
    cornersRow =  np.arange(nRows - sidel + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    cornersCol = np.arange(nCols - sidel + 1)[np.newaxis, :, np.newaxis, np.newaxis]

    subsquareRow = cornersRow + np.arange(sidel)[:, np.newaxis]
    subsquareCol = cornersCol + np.arange(sidel)
    subsquares = X[subsquareRow, subsquareCol]
    subsquareSums = np.tensordot(subsquares, weights, axes = [[2, 3], [0, 1]])
    return subsquareSums

def listComprehensionMethod(X, weights, sidel):
    nRows, nCols = X.shape
    subsquareSums = [[ np.tensordot(X[i : i + sidel, j : j + sidel], weights, [[0, 1], [0, 1]])
                       for j in np.arange(nCols - sidel + 1) ] 
                       for i in np.arange(nRows - sidel + 1) ]
    return np.array(subsquareSums)

B = [[random.randint(0, 10) for j in range(100)]
        for i in range(100)]
B = np.array(B)
sidel = 3
weights = [[random.randint(-2, 2) 
                for j in range(sidel) ]
                for i in range(sidel) ]

print("\nbenchmarks for sidel = ", sidel)
print("indexingMethod benchmark = ")
print(timeit.timeit(lambda : indexingMethod(B, weights, sidel), number = 50))
print("listComprehensionMethod benchmark = ")
print(timeit.timeit(lambda : listComprehensionMethod(B, weights, sidel), number = 50))

sidel = 75
weights = [[random.randint(-2, 2) 
                for j in range(sidel) ]
                for i in range(sidel) ]

print("\nbenchmarks for sidel = ", sidel)
print("indexingMethod benchmark = ")
print(timeit.timeit(lambda : indexingMethod(B, weights, sidel), number = 50))
print("listComprehensionMethod benchmark = ")
print(timeit.timeit(lambda : listComprehensionMethod(B, weights, sidel), number = 50))

