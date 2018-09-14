---
layout : post
title : Pruning an Sklearn Decision Tree in Python
date : 2018-09-13
---

In this post we will look at performing cost-complexity pruning on a sci-kit learn decision tree classifier
in `python`. A decision tree classifier is a general statistical model for predicting which target class a
data point will lie in. There are several methods for preventing a decision tree from overfitting the data
it is trained on; we will be looking at the particular method of cost-complexity pruning as discussed in 
["The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman](https://web.stanford.edu/~hastie/ElemStatLearn/).
To perform our pruning, we will create the class `Pruner` in `prune.py`.

We will be looking at simple simulated data where features are simply xy-coordinates in the unit square, and there
are simply two target classes. Here is a picture of our simulated data:

![Simulated Data]({{site . url}}/assets/2018-09-14-files/graphs/data.svg)

The points with `x + y > 1` are in class `1`, while other points are in class `0`; however, there is a `10%`
error rate where a random `10%` of the data have their classes flipped.

We train a model on this data using `sklearn.tree.DecisionTreeClassifier` and we find that this model has 
the following output:

![Overfitted Model]({{site . url}}/assets/2018-09-14-files/graphs/unprunedModel.svg)

As we can see, this model is overfitting the data. Ideally, it should learn to cut the classes along the line
`x + y = 1`, but we see that it is learning the variance created by the error rate.

We will later use cost-complexity pruning combined with cross validation to keep the model from overfitting. The
result of the pruned model is the following:

![Final Model]({{site . url}}/assets/2018-09-14-files/graphs/finalModel.svg)

We see that there is a small amount of unwanted blue in the upper right, but most of the unwanted variation is gone.
The model is still having trouble tracking the entirety of the edge `x + y = 1`, but this is mostly a problem
created by using a decision tree without doing any preprocessing on the xy-coordinates. A decision tree does
a better job of dealing with class edges that are nearly horizontal or vertical, not diagonal. However, we will
not doing any preprocessing as we are mainly interested in demonstrating how the pruning will "unlearn" the
random variation.

The pruning method "ungrows" the decision tree by selecting removing nodes. For example, the following picture
shows all of the original nodes; nodes no longer included in the final model are marked in red, while those
still active are marked in green.

<img src = "{{site . url}}/assets/2018-09-14-files/graphs/prunedpic.svg" alt = "Final Model Pruning" width = "1000"/>

In the above picture, each node is marked with its index inside the list of nodes in the classifying tree (which
is make by `sklearn.tree.DecisionTreeClassifier` when the model is trained) and the entropy-cost increase resulting
from pruning away the split related to that node.

Below is the logic of the decision tree of the final model.

<img src = "{{site . url}}/assets/2018-09-14-files/graphs/prunedTree.svg" alt = "Final Model Tree" width = "1000" />

Before we discuss our implementation details, let us show how to use the class `Pruner` to do cost-complexity pruning on 
an `sklearn` decision tree classifier. 




# Using `prune.py`

The code for this example can be found in `example.py`.

First, let us import the modules we will need and also set up the random state.
``` python
import prune 

from sklearn.tree import (DecisionTreeClassifier, export_graphviz)
from sklearn.model_selection import cross_val_score

from graphviz import Digraph

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(20180904)
```

Next, we will need a function to create our simulated data.
``` python
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
```
Now, let's get the data, and see what it looks like.
``` python
# Get the simulated data.

nPoints = 500
errorRate = 0.1
x, y = simulateData(nPoints, errorRate)

# Graph the data.

plt.scatter(x[:, 0], x[:, 1], c = y, cmap = 'winter', s = 100)
plt.title('Simulated Data, Error Rate = ' + str(errorRate * 100) + "%")
plt.savefig('graphs/data.svg')
plt.show()
```
This generates the scatter plot found in the introduction of the post.

Next, let's train an `sklearn.tree.DecisionTreeClassifier` model on this data, but we won't restrict the growth of the model 
in any way, i.e. it will be over-fitted. In fact, let's get some cross-validation scores to see the accuracy of this overfit.
``` python
# Get cross-validation of over-fitted model.

model = DecisionTreeClassifier(criterion = "entropy")

crossValScores = cross_val_score(model, x, y.reshape(-1), cv = 5)
print("Cross Valuation Scores for unpruned tree are ", crossValScores)
print("Mean accuracy score is ", crossValScores.mean())
print("Std accuracy score is ", crossValScores.std())
```
This gives the following output:
```
Cross Valuation Scores for unpruned tree are  [ 0.85  0.79  0.75  0.74  0.78]
Mean accuracy score is  0.782
Std accuracy score is  0.0386781592116
```
Now, the data has a random error rate of 0.15, and we see that the mean accuracy score is more than two of its standard
deviations below 0.85, the ideal accuracy rate. So let us take a look at what the model looks like if we train on 
the entire data set.
``` python
# Let's train on the whole data set, and see what the model looks like.

print("Training Tree")
model.fit(x, y)
print("Finished Training")

ax = plt.gca()
prune.plotTreeOutput(ax, model.tree_, [0,0], [1,1], edges = True)
plt.title("Output for Unpruned Tree")
plt.savefig('graphs/unprunedModel.svg')
plt.show()
```
We get

![Overfitted model]({{site . url}}/assets/2018-09-14-files/graphs/unprunedModel.svg)

So we can see that the model is learning some of the variance that is created by the random error rate, which is a result
of overfitting. We can fix this by pruning some of the tree away by using cost-complexity pruning. Let's see how we can 
use the `Pruner` class in `prune.py` to accomplish this.

``` python
# Get the model's tree

tree = model.tree_
print("Number of Nodes in Tree = ", tree.node_count)
print("Features of model.tree_ are\n", dir(tree))

# Set up the pruner.

pruner = prune.Pruner(model.tree_)
nPrunes = len(pruner.pruneSequence) # This is the length of the pruning sequence.
```
When we pass the tree into the pruner, it automatically finds the order that the nodes (or more properly, the splits)
should be pruned. We may then use `Pruner.prune()` to prune off a certain number of splits. Be aware that `Pruner.prune(0)`
will prune off **zero** splits, i.e. return the tree to its original order. Also, you can pass in negative numbers to `Pruner.prune()`
which acts similarly to using negative indices in python lists or arrays. 

As you can see, you don't have to use `Pruner.prune()` in any particular order.
``` python
# Pruning doesn't need to be done in any particular order.

print("Now pruning up and down pruning sequence.")
pruner.prune(10)
pruner.prune(-1)
pruner.prune(3)
```

Let's take a look at what the active tree looks like for a pruned tree.
``` python
# Let's see what the result of pruner.pruning(-5) looks like.
# Number of splits for prune of -5 will be 4, because prune of -1 is just the root node (i.e. no splits).

print("Now getting pruned graph for pruner.prune(-5)")
pruner.prune(-5)
g = prune.makeSimpleGraphViz(model.tree_) 
g.render("graphs/pruneMinus5.dot")
```
To convert the `.dot` file into an `.svg` file, we need to run the following from the command prompt:

```
dot -Tsvg graphs\pruneMinus5.dot -o graphs\pruneMinus5.svg
```
Now, we can see what the graph looks like:

![Active graph for `pruner.prune(-5)`]({{site . url}}/assets/2018-09-14-files/graphs/pruneMinus5.svg)

Each node is an active node in the pruned graph, and it contains information on the index of the node inside
the tree and the cost of pruning the node's split when the node isn't an active leaf.

Next, let's take a look at what the output of the model looks like after `pruner.prune(-5)`.
``` python
# Graph the output of pruner.prune(-5).

ax = plt.gca()
prune.plotTreeOutput(ax, model.tree_, [0,0], [1,1], edges = True)
plt.title("Output for pruner.prune(-5)")
plt.savefig('graphs/manyPrune.svg')
plt.show()
```
The graph we get is

![Graph of pruner.prune(-5) output.]({{site . url}}/assets/2018-09-14-files/graphs/manyPrune.svg)

When we do cost-complexity pruning, we find the pruned tree that minimizes the cost-complexity. The cost is the measure
of the impurity of the tree's active leaf nodes, e.g. a weighted sum of the entropy of the samples in the active leaf nodes with
weight given by the number of samples in each leaf. The complexity is some measure of how complicated the three is; in our case,
the complexity is the number of nodes in the tree. The cost-complexity is then a weighted sum `cost + (complexityWeight) * complexity`. 
We use cross-validation to pick out a good complexity weight.

Before we do that, let's take a look at cost-complexities for different values of the complexity weight. We mark the minimum
cost-complexity for each weight.
``` python
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
```
We get the following graph.

![Graph of cost-complexities for different weights.]({{site . url}}/assets/2018-09-14-files/graphs/costComplexity.svg)
