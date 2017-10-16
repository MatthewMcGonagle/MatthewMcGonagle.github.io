import numpy as np
import timeit 
import random
import matplotlib.pyplot as plt

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


# Functions for benchmarks of different methods.

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

def listComprehensionMethod2(X, weigths, sidel):
    nRows, nCols = weights.shape
    xRows, xCols = X.shape
    result = np.zeros((xRows - sidel + 1, xCols - sidel + 1))
    for i in range(nRows):
        for j in range(nCols):
            result += X[i : i + xRows - sidel + 1, j : j + xCols - sidel + 1] * weights[i, j]
    return result

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

    print('Check index method == list comprehension 1 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod(B, weights, sidel)))
    print('Check index method == list comprehension 2 : ', end = '')
    print(np.array_equal(indexingMethod(B, weights, sidel), listComprehensionMethod2(B, weights, sidel)))

print('\nsidel\tindex\tlist1\tlist2')
for s,i,l1,l2 in zip(search, results['indexMethod'], results['list1'], results['list2']):
    print(s, '\t', '%.3f' % i, '\t', '%.3f' % l1, '\t', '%.3f' % l2)
    
for key in results:
    results[key] = np.array(results[key])

plt.plot(search, results['indexMethod'])
plt.plot(search, results['list1'])
plt.plot(search, results['list2'])
plt.legend(['index', 'listComprehension1', 'listComprehension2'])
plt.title('Benchmarks vs Different Sub-square Side Lengths')
plt.savefig('2017-10-13-Benchmarks.svg')

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

print('\nsidel\tindex\tlist1\tlist2')
for s,i,l1,l2 in zip(search, results['indexMethod'], results['list1'], results['list2']):
    print(s, '\t', '%.3f' % i, '\t', '%.3f' % l1, '\t', '%.3f' % l2)

plt.clf()
plt.plot(search, results['indexMethod'])
plt.plot(search, results['list1'])
plt.plot(search, results['list2'])
plt.legend(['index', 'listComprehension1', 'listComprehension2'])
plt.title('Variable Weights Benchmarks vs Different Sub-square Side Lengths')
plt.savefig('2017-10-13-VarWeightsBenchmarks.svg')

