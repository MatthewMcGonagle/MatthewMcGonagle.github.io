'''
heatmapBins.py
Example of using pandas and seaborn to make a heat map of mean values for 2d histogram binning. 
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use a seed to have reproducible results.

np.random.seed(20190121)

nSamples = 1000 
nCut = 10

def zFunction(X):
    # z = x - y
    return X[:, 0] - X[:, 1]

print('Generating Data')
data = np.random.normal(size = (nSamples, 2))
data = pd.DataFrame(data)
data['z'] = zFunction(data.values)
print(data.info())

plt.clf()
plt.title('Feature Data')
plt.scatter(data.loc[:, 0], data.loc[:, 1])
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.tight_layout()
plt.savefig('graphs/features.svg')

plt.clf()
sns.jointplot(data[0], data[1], kind = 'kde')
plt.gcf().suptitle('Density of Features')
plt.tight_layout()
plt.savefig('graphs/density.svg')

cuts = pd.DataFrame({str(feature) + 'Bin' : pd.cut(data[feature], nCut) for feature in [0, 1]})
print('at first cuts are pandas intervalindex.')
print(cuts.head())
print(cuts.info())

print(data.join(cuts).head())

means = data.join(cuts).groupby( list(cuts) ).mean()
means = means.unstack(level = 0) # Use level 0 to put 0Bin as columns.

# Reverse the order of the rows as the heatmap will print from top to bottom.
means = means.iloc[::-1]
print(means.head())
print(means['z'])

plt.clf()
sns.heatmap(means['z']) 
plt.title('Means of z vs Features 0 and 1')
plt.tight_layout()
plt.savefig('graphs/means1.svg')

plt.clf()
sns.heatmap(means['z'], xticklabels = means['z'].columns.map(lambda x : x.left),
                        yticklabels = means['z'].index.map(lambda x : x.left))
plt.title('Means of z vs Features 0 and 1')
plt.tight_layout()
plt.savefig('graphs/means2.svg')
