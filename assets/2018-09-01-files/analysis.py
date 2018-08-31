'''
analysis.py

Graphs the outcomes of the run-time statistics.
'''

import pandas as pd
import matplotlib.pyplot as plt

stats = pd.read_csv('Analysis\\stats.csv', index_col = 'name')

# Remove the prefix 'stderror' from all of the names.
nSkip = len("stderror")
index = [name[nSkip:] for name in stats.index]

# Change the no test case to have the name 'None' + some padding that will let it be correctly
# processed with the other names later when we remove the dependence state from the names in
# the index..

noTestName = "stderrorNOTEST"[nSkip:]
index[index == noTestName] = "None123" # Pad the end by 3 characters that will be removed later. 

# Let's check up on the indices we have made.
print(index)

# Give stats the new indices.
stats.index = index

# The state of the dependence is given by the last three characters, except in the no test case.
# Also remove the state of dependence from the index names, this is where the padding for the
# no test index was needed.

stats['dependence'] = [name[-3:] for name in stats.index]
stats.index = [old[:-3] for old in stats.index] 

# Manually change the no test case.

stats.loc['None', 'dependence'] = 'None'

# Now convert the heap allocations from being in bytes to being in Megabytes, appropriately
# change the names of the columns too.

byteCols = ["totalHeap", "gcHeap", "maxResidency", "maxSlop"]
rename = {}
for col in byteCols:
    oldName = col + ' (b)'
    rename[oldName] = col + " (Mb)"
    stats[oldName] = stats[oldName] * 1e-6 

stats = stats.rename(index = str, columns = rename)

print(stats)

# We will be splitting our graphs in to two cases. Graph all of the dependent and no test case together.
# Second case is graphing all of the independent and no test case together.

indMask = (stats.dependence == "ind") | (stats.dependence == "None") 
depMask = (stats.dependence == "dep") | (stats.dependence == "None") 

stats = { 'ind' : stats.loc[indMask, :]
        , 'dep' : stats.loc[depMask, :]
        }

# Now make the graphs.

directory = "Analysis\\"
for split, name in zip(['ind', 'dep'], ['Independent', 'Dependent']):

    stats[split].plot.bar(y = "maxResidency (Mb)")
    plt.title('Max Residency in Heap for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Size (Megabytes)')
    plt.savefig(directory + split + 'Residency.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = ["totalHeap (Mb)", "gcHeap (Mb)"])
    plt.title('Heap Allocations for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Size (Megabytes)')
    plt.savefig(directory + split + 'Allocations.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = ["user Prod (%)", 'total Prod (%)'])
    plt.title('Productivities for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Productivity by Time (%)')
    plt.savefig(directory + split + 'Productivities.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = 'Total (s)' )
    plt.title('Total Times for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Times (s)')
    plt.savefig(directory + split + 'Times.svg', bbox_inches = 'tight')
    plt.show()

