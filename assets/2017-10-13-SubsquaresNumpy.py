import numpy as np

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

subsquareRow = np.full((nSqRows, nSqCols), np.arange(nSqRows)[:, np.newaxis], dtype = 'int32').reshape(-1)
subsquareCol = np.full((nSqRows, nSqCols), np.arange(nSqCols)[np.newaxis, :], dtype = 'int32').reshape(-1) 
subsquareList = subsquares[subsquareRow, subsquareCol, :, :]
print(subsquareList.shape)
print(subsquareList)


