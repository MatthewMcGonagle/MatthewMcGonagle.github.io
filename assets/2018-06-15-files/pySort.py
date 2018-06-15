def doQuickSort(myList):
    newList = myList.copy()
    doQuickSortInPlace(newList, len(newList), offset = 0)
    return newList

def doQuickSortInPlace(myList, size, offset):

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
        temp = myList[loc1]
        myList[loc1] = myList[loc2]
        myList[loc2] = temp
