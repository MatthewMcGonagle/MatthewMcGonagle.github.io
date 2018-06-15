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
