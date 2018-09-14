'''
    Module: example.py
    Author: Matthew McGonagle

    Create some simulated data for a classification problem. Use cross-validation to train a 
    classification tree and do cost-complexity pruning to prevent overfitting. We use the
    classes defined in `prune.py` to do the pruning.

'''
import prune 

from sklearn.tree import (DecisionTreeClassifier, export_graphviz)
from sklearn.model_selection import cross_val_score

from graphviz import Digraph

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(20180904)

def simulateData(nPoints, errorRate):
    '''
    Create simulated data in unit square. Points with x + y > 1 are in class 1 and otherwise is class
    0. However, a certain random portion of the points have their classes randomly switched, the 
    proportion indicated by the error rate.

    Parameters
    ----------
    nPoints : Int
        The number of points in the simulated data.
    
    errorRate : Float
        The proportion of the data that will be randomly selected to have their classes flipped.

    Returns
    -------
    Numpy Array of Float of Shape (nPoints, 2)
        The xy-coordinates of the data.
    
    Numpy Array of Int of Shape (nPoints, 1)
        The classes of each data point.
    '''

    x = np.random.rand(nPoints, 2)

    y = np.zeros(nPoints)
    upperRight = x[:, 0] + x[:, 1] > 1
    y[upperRight] = 1

    errors = np.random.choice(np.arange(nPoints), int(nPoints * errorRate))
    y[errors] = 1 - y[errors] 

    return x, y.reshape(-1, 1)

###########################################
############# Main ########################
###########################################

# Get the simulated data.

nPoints = 500
errorRate = 0.1
x, y = simulateData(nPoints, errorRate)

# Graph the data.

plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'winter', s = 100)
plt.title('Simulated Data, Error Rate = ' + str(errorRate * 100) + "%")
plt.savefig('graphs/data.svg')
plt.show()

# Get cross-validation of over-fitted model.

model = DecisionTreeClassifier(criterion = "entropy")

crossValScores = cross_val_score(model, x, y.reshape(-1), cv = 5)
print("Cross Valuation Scores for unpruned tree are ", crossValScores)
print("Mean accuracy score is ", crossValScores.mean())
print("Std accuracy score is ", crossValScores.std())

# Let's train on the whole data set, and see what the model looks like.

print("Training Tree")
model.fit(x, y)
print("Finished Training")

ax = plt.gca()
prune.plotTreeOutput(ax, model.tree_, [0,0], [1,1], edges = True)
plt.title("Output for Unpruned Tree")
plt.savefig('graphs/unprunedModel.svg')
plt.show()

####################################
##### Now do Pruning
####################################

# Get the model's tree

tree = model.tree_
print("Number of Nodes in Tree = ", tree.node_count)
print("Features of model.tree_ are\n", dir(tree))

# Set up the pruner.

pruner = prune.Pruner(model.tree_)
nPrunes = len(pruner.pruneSequence) # This is the length of the pruning sequence.

# Pruning doesn't need to be done in any particular order.

print("Now pruning up and down pruning sequence.")
pruner.prune(10)
pruner.prune(-1)
pruner.prune(3)

# Let's see what the result of pruner.pruning(-5) looks like.
# Number of splits for prune of -5 will be 4, because prune of -1 is just the root node (i.e. no splits).

print("Now getting pruned graph for pruner.prune(-5)")
pruner.prune(-5)
g = prune.makeSimpleGraphViz(model.tree_) 
g.render("graphs/pruneMinus5.dot")

# Graph the output of pruner.prune(-5).

ax = plt.gca()
prune.plotTreeOutput(ax, model.tree_, [0,0], [1,1], edges = True)
plt.title("Output for pruner.prune(-5)")
plt.savefig('graphs/manyPrune.svg')
plt.show()

# Now let's take a look at the cost-complexity curves for several different choices of the complexity weight.

weights = np.linspace(0.0, 3.0, 5)
for weight in weights:

    sizes, costComplexity = pruner.costComplexity(weight)
    minI = np.argmin(costComplexity)
    plt.plot(sizes, costComplexity)
    plt.scatter(sizes[minI], costComplexity[minI], color = 'black')

plt.legend(weights)
plt.title("Cost-complexity vs Size for Different Weights (Minima Marked)")
plt.xlabel("Size of Pruned Tree")
plt.ylabel("Cost-complexity")
plt.savefig('graphs/costComplexity.svg')
plt.show()  

# Now let's do cross-validation to find an optimal weight.

weights = np.linspace(0.0, 6.0, 45)
nCrossVal = 20 
model = DecisionTreeClassifier(criterion = "entropy")

sizes, scores = prune.doCrossValidation(model, x, y, nCrossVal, weights)

# Let's look at the statistics of the scores for different weights.

means = scores.mean(axis = 0)
stds = scores.std(axis = 0)

# Find the weight giving the best mean score.

maxWeighti = np.argmax(means)

print("Best mean accuracy is ", means[maxWeighti])
print("Std of mean accuracy is ", stds[maxWeighti])
print("with best weight is ", weights[maxWeighti])

# Plot the mean scores with standard deviations in the scores.

plt.plot(weights, means)
plt.scatter(weights[maxWeighti], means[maxWeighti])
plt.plot(weights, means + stds, color = 'green')
plt.plot(weights, means - stds, color = 'green')
plt.title("Mean Cross Validated Accuracy vs Complexity Weight")
plt.legend(['mean', 'mean + std', 'mean - std'], loc = (0.6, 0.05))
plt.xlabel('Complexity Weight')
plt.ylabel('Accuracy')
plt.savefig('graphs/cvAccuracy.svg')
plt.show()

# Plot how the size of the trees vary with the weights.

sizeStd = sizes.std(axis = 0)
sizes = sizes.mean(axis = 0)
plt.plot(weights, sizes)
plt.scatter(weights[maxWeighti], sizes[maxWeighti])
plt.plot(weights, sizes + sizeStd, color = 'green')
plt.plot(weights, sizes - sizeStd, color = 'green')
plt.title("Tree Size vs Complexity Weight")
plt.ylabel("Tree Size")
plt.xlabel("Complexity Weight")
plt.legend(['mean', 'mean + std', 'mean - std'])
plt.savefig('graphs/cvSize.svg')
plt.show()
        
# Let's retrain for the optimal weight, and find what the output
# looks like.
 
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(x, y)
pruner = prune.Pruner(model.tree_)
pruner.pruneForCostComplexity(weights[maxWeighti]) 

ax = plt.gca()
prune.plotTreeOutput(ax, model.tree_, [0,0], [1,1], edges = True)
plt.title('Output of Final Model')
plt.savefig('graphs/finalModel.svg')
plt.show()

# Now let's get some visualizations of the final result.
g = prune.makePrunerGraphViz(pruner)
g.render('graphs/prunedpic.dot')
export_graphviz(model, out_file='graphs/prunedTree')
