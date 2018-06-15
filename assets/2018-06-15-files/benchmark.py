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

def main():
    '''
    Main execution.
    '''

    # Set up a random list to test our sorting functions.

    nItems = 20 
    myList = [i for i in range(nItems)]
    myList = myList + myList
    myList = random.sample(myList, len(myList)) 
    print(myList)

    # Test the sorting functions.
   
    cSortedList = cSort.doQuickSort(myList)
    print('Result of cSort is\n', cSortedList)

    pySortedList = pySort.doQuickSort(myList)
    print('Results of pySort is\n', pySortedList)
   
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
   
    # Plot our benchmark times.

    for sortName in benchmarks.keys(): 
        plt.plot(benchmarkSizes, benchmarks[sortName])
    plt.title('Benchmarks for Sorting Algorithms')
    plt.xlabel('List Size')
    plt.ylabel('Average Time (ms)')
    plt.legend(benchmarks.keys())
    plt.savefig('times.svg')
    plt.show()

    # Plot the ratios of the benchmark times.

    ratios = [pyResult / cResult for pyResult, cResult in 
                zip(benchmarks['pySort'], benchmarks['cSort'])]
    plt.plot(benchmarkSizes, ratios)
    plt.title('Ratio of pySort.doQuickSort to cSort.doQuickSort')
    plt.xlabel('List Size')
    plt.ylabel('Ratio pySort / cSort') 
    plt.savefig('ratios.svg')
    plt.show()
                     
 
# Do the main execution.

main()
