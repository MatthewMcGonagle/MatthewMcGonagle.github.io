---
layout : post
title : Pruning an Sklearn Decision Tree in Python
date : 2018-09-13
---

## [Download example.py Here]({{site . url}}/assets/2018-09-14-files/example.py)
## [Download prune.py Here]({{site . url}}/assets/2018-09-14-files/prune.py)

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
To convert the `.dot` file into an `.svg` file, you need to make sure you have `graphviz` installed (see [the graphviz homepage](https://www.graphviz.org/)).
We also need to run the following from the command prompt:

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

Next, let's do cross-validation for finding the optimal complexity weight. The class `Pruner` and `Pruner.prune()` doesn't really
fit very well in an `sklearn` pipeline, so we can't really use the built in cross-validation scorers. Instead the function
`doCrossValidation()` in `prune.py` will be useful. We will get the cross-validated scores and then graph the results.
``` python
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
```

The terminal output is
```
Best mean accuracy is  0.8296
Std of mean accuracy is  0.0260583959598
with best weight is  2.72727272727
```
We also get the following graphs.

![Mean Accuracy vs Weight]({{site . url}}/assets/2018-09-14-files/graphs/cvAccuracy.svg)

![Tree Size vs Weight]({{site . url}}/assets/2018-09-14-files/graphs/cvSize.svg)

Let's now retrain for the optimal weigth on the entire data set and see what the output of
the model looks like.
``` python
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
```
The output of the model is:

![Output of Optimal Model]({{site . url}}/assets/2018-09-14-files/graphs/finalModel.svg)

We see that the optimal model has a little trouble with learning some variation in the upper right (the thin blue box in the upper right).
However, it learned much less variation than the original overfitted model.

To get the picture of which nodes were pruned and the picture of the logic of the final active tree, we use the following:
``` python
# Now let's get some visualizations of the final result.
g = prune.makePrunerGraphViz(pruner)
g.render('graphs/prunedpic.dot')
export_graphviz(model, out_file='graphs/prunedTree')
```

For `prunedpic.dot` you need to run the following from a terminal:
```
dot -Tsvg graphs\prunedpic.dot -o graphs\prunedpic.svg
```

The results of these visualizations are given in the introduction to the post.

Next, let's take a look at the implementation details of the members of `prune.py`.

# Implementation of the Class `Pruner` 

The class `Pruner` is implemented in `prune.py`.

First, let's take a look at the members of the tree inside an `sklearn.tree.DecisionTreeClassifier`. This tree is the member
`DecisionTreeClassifier.tree_` and is of type `sklearn.tree.Tree`. To see its members, you can run something similar to
``` python
# Examine the model's tree

tree = model.tree_
print("Features of model.tree_ are\n", dir(tree))
```
We get the following output:
``` 
Features of model.tree_ are
 ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', 
'__gt__', '__hash__', '__init__', '__le__', '__lt__', '__ne__', '__new__', '__pyx_vtable__', '__reduce__', '__reduce_ex__', 
'__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', 'apply', 'capacity', 'children_left', 
'children_right', 'compute_feature_importances', 'decision_path', 'feature', 'impurity', 'max_depth', 'max_n_classes', 
'n_classes', 'n_features', 'n_node_samples', 'n_outputs', 'node_count', 'predict', 'threshold', 'value', 'weighted_n_node_samples']
```

Its important to know that the information for the nodes of the tree are stored in arrays such as `impurity`, `children_left` and
`children_right`. Also, children always come in pairs. There is a left child if and only if there is a right child. The arrays for
the children store the indices that the children correspond to in the arrays of node information. The absence of children (i.e.
a leaf node) is handled by setting the nodes children to be `-1`.

The class `Pruner` at initialization finds the order to prune the (non-leaf) nodes in. A node may be pruned only when its children
have already been pruned (or are leaves). The order is then determined by the fact that the next node to be pruned is the one that
will result in the **smallest** increase in the cost of the (active) tree. The pruning cost for each node can be calculated from
its information and that of its children.

Now, let's look at the implementation details of the class `Pruner`.
``` python
class Pruner:
    '''
    Class for doing pruning of a sci-kit learn DecisionTreeClassifier. At initialization, the order of the nodes to prune is found, but no 
    pruning is done. The order of pruning is determined by the pruning that results in the smallest increase in the cost (e.g. entropy or gini index)
    of the tree is done first. Note, a node can only be pruned if both of its children are leaf nodes (also recall that for a sci-kit learn 
    DecisionTreeClassifier the children always come in pairs).

    Members
    -------
    tree : sklearn.tree.Tree
        A reference to the tree object in a sci-kit learn DecisionTreeClassifier; in such a classifier, this member object is usually called tree_.

    leaves : List of Int 
        A list of the indices which are leaf nodes for the original decision tree.

    parents : Numpy array of Int 
        Provides the index of the parent node for nodei. Note, the fact that the root has no parent is indicated by setting
        parents[0] = -1.        
    
    pruneCosts : Numpy array of Float
        The cost increase if nodei is pruned. Note that cost is calculated by weighting a node's impurity (e.g. entropy or gini index) by
        the number of samples in the node.

    originalCost : Float
        The original cost for the fully grown tree, i.e. the total cost for all of the original leaf nodes.

    originalChildren : Pair of Numpy array of Int
        A copy of the original left and right children indices for the original tree. So it is (children_left.copy(), children_right.copy()).

    pruned : Numpy array of Bool
        Used for calculating the prune sequence. Holds whether nodei has been pruned. The leaf nodes are considered to automatically have
        been pruned.

    pruneSequence : Numpy array of Int
        The order to prune the nodes. pruneSequence[0] = -1 to indicate the sequence starts with no pruning; so pruneSequence[i] is the ith node
        to prune. 

    pruneState : Int
        Holds the current number of nodes pruned. So a state of 0 means no pruning has occurred. This is changed with the member function prune().
        Initialized to 0. 
    '''

    def __init__(self, tree):

        '''
        Finds the prune sequence and initializes the prune state to be 0.

        Parameters
        ---------
        tree : sklearn.tree.Tree
            A reference to the tree that will be pruned by this pruner. Note that for sklearn.tree.DecisionTreeClassifier, the tree
            is the member variable DecisionTreeClassifier.tree_.
        '''

        self.tree = tree

        self.leaves = self._getLeaves(tree)
        self.parents = self._getParents(tree)
        self.pruneCosts = self._getPruneCosts(tree)
        self.originalCost = self.pruneCosts[self.leaves].sum()
        self.originalChildren = list(zip(tree.children_left.copy(), tree.children_right.copy()))

        # Initially, only the leaves count as being already pruned.

        self.pruned = np.full(len(tree.impurity), False)
        self.pruned[self.leaves] = True

        self.pruneSequence, self.costSequence = self._makePruneSequence(tree)
        self.pruneState = 0

    def _getLeaves(self, tree):

        '''
        Find the leaf nodes of the tree.
        Parameters
        ----------
        tree : sklearn.tree.Tree
            The tree to find the leaf nodes of.

        Returns
        -------
        List of Int
            The list of indices that correspond to leaf nodes in the tree.
        '''

        leaves = []

        # Note that children always come in pairs.

        for nodei, lChild in enumerate(tree.children_left):

           if lChild == -1:
                leaves.append(nodei) 
       
        return leaves 

    def _getParents(self, tree):
        '''
        Find the list of indices of parents for each node. The parent of the root node is defined to be -1.
        
        Parameters
        ----------
        tree : sklearn.tree.Tree
            Tree to find the list of parents for.

        Returns
        -------
        Numpy array of Int
            The indices of the parent node for each node in the tree. We consider the parent of the root to be -1.
        '''

        parents = np.full(len(tree.children_left), -1) 

        for nodei, children in enumerate(zip(tree.children_left, tree.children_right)):

            lChild, rChild = children

            # Children always come in pairs for a decision tree.

            if lChild != -1:
               parents[lChild] = nodei
               parents[rChild] = nodei

        return parents

    def _getCost(self, tree, nodei):

        '''
        Get the cost of a node; i.e. the product of the impurity and the number of samples. Note, this is not
        the cost of pruning the node.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The tree that the node is in.
        nodei : Int
            The index of the node to calculate the cost of.

        Returns
        -------
        Float
            The cost of the node (NOT the cost of pruning the node).
        '''

        cost = tree.n_node_samples[nodei] * tree.impurity[nodei]

        return cost

    def _getPruneCosts(self, tree):
        '''
        Calculate the cost of pruning each node. This is the amount that the total cost of the current pruned
        tree will increase by if we prune the node. Is given by the difference between the cost of this node and
        the costs of its children.

        Note, there isn't really any cost associated with pruning a leaf node as they aren't prunable; so they are given a
        cost of 0.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree to calculate the costs for.

        Returns
        -------
        Numpy array of Float
            The costs of pruning each node in the tree.
        '''

        pruneCosts = np.zeros(len(tree.impurity))
        nodeCosts = tree.n_node_samples * tree.impurity

        for nodei, (lChild, rChild) in enumerate( zip( tree.children_left, tree.children_right) ):

            # Children always come in pairs.

            if lChild != -1:

                decrease = nodeCosts[nodei] - nodeCosts[lChild] - nodeCosts[rChild] 
                pruneCosts[nodei] = decrease

        return pruneCosts

    def _getInitialCandidates(self, tree):
        '''
        Find the initial list of prunable nodes (i.e. parents whose both left and right children are leaf nodes).
        Also find their prune costs.

        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree.

        Returns
        -------
        List of Int
           The indices of the initial candidates to prune. 
        List of Float
           Their corresponding list of prune costs.
        '''

        candidates = []
        candidateCosts = []

        for leafi in self.leaves:

            parenti = self.parents[leafi]
            if parenti != -1:
                lChild = tree.children_left[parenti]
                rChild = tree.children_right[parenti]

                if self.pruned[lChild] and self.pruned[rChild] and parenti not in candidates:
                    candidates.append(parenti)
                    candidateCosts.append(self.pruneCosts[parenti])

        return candidates, candidateCosts

    def _popNextPrune(self, candidates, costs):
        '''
        Remove the next prune node from the list of candidates, and also remove its cost from the list of costs. 

        The next node to prune is found by minimizing over the costs of all of the candidates.

        Parameters
        ----------
        candidates : List of Int
            The list of indices of nodes that we could potentially prune next.
        costs : List of Float
            The corresponding list of pruning costs for each candidate.

        Returns
        -------
        Int
            The index of the next prune node.
        '''

        minCosti = np.argmin(costs)
        nextPrune = candidates.pop(minCosti)
        costs.pop(minCosti)

        return nextPrune
        

    def _makePruneSequence(self, tree):
        '''
        Find the order to prune the nodes for cost-complexity pruning. The order is determined by the fact that nodes with the smallest
        pruning cost are pruned first. Also find the accumulative pruning cost for pruning in this order.

        Note that pruneSequence[0] = -1, indicating the no pruning. Also costSequence[0] = 0 as no pruning has occured. 
        Parameters
        ----------
        tree : sklearn.tree.Tree
            The original unpruned tree.

        Returns
        -------
        Numpy array of Int
            The order to prune the nodes.

        Numpy array of Float
            The total accumulative pruning cost for pruning the nodes in order.
        '''

        pruneSequence = [-1]
        costSequence = [0]
        currentCost = 0

        candidates, costs = self._getInitialCandidates(tree)

        while candidates:

            prunei = self._popNextPrune(candidates, costs)
            self.pruned[prunei] = True
            pruneSequence.append(prunei)
            currentCost += self.pruneCosts[prunei]
            costSequence.append(currentCost)

            parenti = self.parents[prunei]
            if parenti != -1:
                lChild = tree.children_left[parenti]
                rChild = tree.children_right[parenti]

                if self.pruned[lChild] and self.pruned[rChild]:
                    candidates.append(parenti)
                    costs.append(self.pruneCosts[parenti])

        return np.array(pruneSequence), np.array(costSequence)

    def prune(self, prunei):
        '''
        Do pruning/unpruning on the tree. Technically, pruning is done on splits (and not on nodes).
        We specify the number of split to prune away from the ORIGINAL tree.

        If the number of splits to prune is greater than what we have pruned so far, we prune off
        more splits. If it is less, then we unprune (i.e. restore) splits.
        Parameters
        ----------
        prunei : Int
            The number of splits to prune off the original tree. Negative values specify offset
            from the maximum number of prunes possible, similar to how negative indexing of
            arrays works. 

        '''

        nPrunes = len(self.pruneSequence)

        if prunei < 0:
            prunei += nPrunes  

        # If the new state involves more prunes than the old state, we have to prune nodes.
        # Else we need to restore children to their old state.

        if prunei > self.pruneState:

            for prune in range(self.pruneState + 1, prunei + 1):
                nodei = self.pruneSequence[prune]
                self.tree.children_left[nodei] = -1
                self.tree.children_right[nodei] = -1

        elif prunei < self.pruneState: 

            for prune in range(prunei + 1, self.pruneState + 1):
                nodei = self.pruneSequence[prune]
                lChild, rChild = self.originalChildren[nodei]
                self.tree.children_left[nodei] = lChild
                self.tree.children_right[nodei] = rChild

        # Update the prune state.

        self.pruneState = prunei

    def costComplexity(self, complexityWeight):
        '''
        Compute the cost-complexity curve for a given weight of the complexity. The complexity is simply the number
        of nodes in the pruned tree. So the cost-complexity is a combination of the cost of the tree and the weighted
        size. To the find the optimal complexity weight, one can do something such as cross-validation.
        
        Also, return a list of the sizes for each cost-complexity.

        Paramters
        ---------
        complexityWeight : Float         
            The weight to apply to the complexity measure.

        Returns
        -------
        Numpy of Int
            The size of the pruned tree for each cost-complexity.

        Numpy of Float
            The cost-complexity measure for each tree size.
        '''

        nPrunes = len(self.pruneSequence)

        # Recall that each prune removes two nodes.
        sizes = self.tree.node_count - 2 * np.arange(0, nPrunes)

        costs = np.full(len(sizes), self.originalCost)
        costs += self.costSequence 
        costComplexity = costs + complexityWeight * sizes

        return sizes, costComplexity

    def pruneForCostComplexity(self, complexityWeight):
        '''
        Prune the tree to the minimal cost-complexity for the given provided weight.

        Parameters
        ----------
        complexityWeight : Float
            The complexity weight to use for calculating cost-complexity.
        '''

        sizes, costComplexity = self.costComplexity(complexityWeight)

        minI = np.argmin(costComplexity)

        self.prune(minI)
```

Next, let's take a look at `doCrossValidation()`, a function that helps with doing cross-validation of the 
complexity weight.
``` python
def doCrossValidation(model, x, y, nCrossVal, weights):
    '''
    Do cross validation for different complexity weights. Use the results to determine
    the best weight to use. For each weight, this finds the optimal pruning that
    minimizes the cost-complexity for the given complexity weight.

    Paramters
    ---------
    model : sklearn.tree.DecisionTreeClassifier
        The tree model to use.

    x : Numpy Array of Shape (nPoints, 2)
        The dependent variables, i.e. features.

    y : Numpy Array of Shape (nPoints, 1)
        The target class for each data point.

    nCrossVal : Int
        The number of cross validations to do for each weight.

    weights : Numpy array of Float
        The different values of the complexity weights to do cross validation over.

    Returns
    -------
    Numpy Array of Int of Shape (nCrossVal, len(weights))
        The sizes of the optimal trees for each run.
    Numpy Array of Float of Shape (nCrossVal, len(weights))
        The accuracy score of the optimal tree for each run. 
    '''

    scores = []
    sizes = []

    # For each repetition of cross-validation, we iterate over all weights.

    for i in range(nCrossVal):
    
        xtrain, xtest, ytrain, ytest = train_test_split(x, y)
        model.fit(xtrain, ytrain)
        pruner = Pruner(model.tree_)

        # Find the optimal pruning for each weight.
    
        runScores = []
        runSizes = []

        for weight in weights:
            
            treeSizes, costComplexity = pruner.costComplexity(weight)
            minI = np.argmin(costComplexity)
            pruner.prune(minI)
            ypredict = model.predict(xtest)
            acc = accuracy_score(ytest, ypredict)
    
            runScores.append(acc)
            runSizes.append(treeSizes[minI])
    
        scores.append(runScores)
        sizes.append(runSizes)
    
    scores = np.array(scores) 
    sizes = np.array(sizes)
   
    return sizes, scores 
```

# Implementation of Visualization Functions

Next, let's discuss the visualization function implemented in `prune.py`.

To visualize a model's output, we make the class `Box` that will allow us to easily translate the splitting of nodes into
visual information.
``` python
class Box:
    '''
    Class to keep track of the xy-rectangle that a node in a decision tree classifier applies to.

    Can be used to accurately and precisely draw the output of a decision tree classifier.

    Members
    -------
    lowerBounds : List of Float of size 2.
        Holds the lower bounds of the x and y coordinates.

    upperBounds : List of Float of size 2.
        Holds the upper bounds of the x and y coordinates.

    value : None or Int
        If desired, one can specify the value that the tree is supposed to resolve the node to. 
    '''

    def __init__(self, lowerBounds, upperBounds, value = None): 
        '''
        Initialize the member variables.
        Parameters
        ----------
        lowerBounds : List of Float of size 2
            Holds the lower bounds of the x and y coordinates.

        upperBounds : List of Float of size 2
            Holds the upper bounds of the x and y coordinates.

        value : The value of a node that the box represents. Default is None. 
        '''

        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.value = value

    def split(self, feature, value):
        '''
        Split the box into two boxes specified by whether x or y coordinate (i.e. feature) and the threshold value to split at.
        This corresponds to how a node in a classifying tree splits on a feature and threshold value.

        Parameters
        ----------
        feature : Int
            0 for x-coordinate or 1 for y-coordinate.

        value : Float
            The threshold value to do the split. The left box is feature less than or equal to value. The right box is the feature
            greater than this value.

        Returns
        -------
        Box
            This is the left box corresponding to the left child of a split node in a decision tree classifier.

        Box
            This is the right box corresponding to the right child of a split node in a decision tree classifier. 
        '''

        newUpper = self.upperBounds.copy()
        newUpper[feature] = value

        newLower = self.lowerBounds.copy()
        newLower[feature] = value

        return Box(self.lowerBounds, newUpper), Box(newLower, self.upperBounds)

    def _color(self):
        '''
        Get the color for this box using a colormap.

        Returns
        -------
        Color 
            The colormap output. 
        '''

        cmap = cm.get_cmap("winter")

        if self.value == None:
            return cmap(0.0)

        return cmap(float(self.value))

    def convertMatPlotLib(self, edges = False):
        '''
        Convert the box a matplotlib.patches.Rectangle.

        Parameters
        ----------
        edges : Bool
            Whether to include edges in the rectangle drawing. Default is False.

        Returns
        -------
        matplotlib.patches.Rectangle
            The matplotlib patch for this box.
        '''

        width = self.upperBounds[0] - self.lowerBounds[0]
        height = self.upperBounds[1] - self.lowerBounds[1]

        kwargs = {'xy' : (self.lowerBounds[0], self.lowerBounds[1]),
                  'width' : width,
                  'height' : height,
                 } 
        if edges:
            kwargs['facecolor'] = self._color()
        else:
            kwargs['color'] = self._color()

        return Rectangle(**kwargs)
```

The output of the model is determined by the boxes represented by the leaf nodes of the tree. So we make the function
 `getLeafBoxes()` to find these.
``` python
def getLeafBoxes(tree, lowerBounds, upperBounds):
    ''' 
    Get a list of Box for the leaf nodes of a tree and the initial bounds on the output.

    Paramters
    ---------
    tree : sklearn.tree.Tree
        The tree that determines how to split the boxes to find the boxes corresponding to the
        leaf nodes.
    lowerBounds : List of Float of Size 2
        The initial lower bounds on the xy-coordinates of the box.

    upperBounds : List of Float of Size 2
        The inital upper bounds on the xy-coorinates of the box.

    Returns
    -------
    List of Box
        A list of Box for each leaf node in the tree.
    '''

    rootBox = Box(lowerBounds, upperBounds)

    boxes = [(0, rootBox)]
    leaves = []

    # Keep splitting boxes until we are at the leaf level. Use the thresholds and features contained in
    # the tree to do the splitting.

    while boxes:

        nodei, box = boxes.pop()
        lChild = tree.children_left[nodei]

        # If there are no children then we are at a leaf; recall that children always come in pairs for a decision
        # tree.

        if lChild == -1:

            box.value = np.argmax(tree.value[nodei])
            leaves.append(box)

        else:

            rChild = tree.children_right[nodei]

            lBox, rBox = box.split(tree.feature[nodei], tree.threshold[nodei]) 
            boxes.append((lChild, lBox))
            boxes.append((rChild, rBox))

    return leaves
```

Finally, we can make the function `plotTreeOutput()` to make the visualization of a tree model.
``` python
def plotTreeOutput(axis, tree, lowerBounds, upperBounds, edges = False):
    '''
    Get a precise and accurate plot of the output of a decision tree inside a given box.

    Parameters
    ----------
    axis : pyplot axis
        The axis to plot to.

    tree : sklearn.tree.Tree
        The tree to plot.

    lowerBounds : List of Float of Size 2
        The lower bounds of the xy-coordinates of the box to graph.

    upperBounds : List of Float of Size 2
        The upper bounds of the xy-coordinates of the box to graph.

    edges : Bool
        Whether to include edges of the leaf boxes in the output. Default is False.
    '''

    # The output is determined by the leaf nodes of the decision tree.

    leafBoxes = getLeafBoxes(tree, lowerBounds, upperBounds) 
    
    for box in leafBoxes:
    
        rect = box.convertMatPlotLib(edges)
        axis.add_patch(rect)
```

Lastly, we have the function `makeSimpleGraphViz()` to make a `graphviz` representation of the tree including node index and prune costs; and the
function `makePrunerGraphViz()` to make a `graphviz` representation of which nodes are active and which ones have been pruned away.

``` python
def makeSimpleGraphViz(tree):
    ''' 
    Make a simple graphviz vizualization for a decision tree. For each node, we simply
    output the index of the node in the tree, and if the node isn't a leaf, then its
    pruning cost.

    Parameters
    ----------
    tree : sklearn.tree.Tree
        The tree to visualize.

    Returns
    -------
    Digraph
        Reference to the graphviz object created.
    '''

    g = Digraph('g')

    # Make nodes

    nodes = [0]

    while nodes:

        node = nodes.pop()

        lChild = tree.children_left[node]
        rChild = tree.children_right[node]

        # Non-leaf nodes contain information on the cost of pruning away their split.

        if lChild != -1:
            costDecrease = tree.impurity[node] * tree.n_node_samples[node]
            costDecrease -= tree.impurity[lChild] * tree.n_node_samples[lChild]
            costDecrease -= tree.impurity[rChild] * tree.n_node_samples[rChild] 
            costDecrease = "\n" + ("% .1f" % costDecrease) 

            nodes.append(lChild)
            nodes.append(rChild) 
        else:
            costDecrease = ""
        g.node(str(node), str(node) + costDecrease) 

    # Make edges

    for node, children in enumerate(zip(tree.children_left, tree.children_right)):

        lchild, rchild = children 

        if lchild != -1:
            g.edge(str(node), str(lchild))

        if rchild != -1:
            g.edge(str(node), str(rchild))

    return g

def makePrunerGraphViz(pruner):
    '''
    Make a simple graphviz vizualization of a pruner's state of pruning a decision tree. For each node, 
    we simply output its index and its prune cost (if it isn't a leaf node of the original tree). Also,
    we highlight active nodes in green, and unactive nodes (i.e. pruned away) in red.

    Parameters
    ----------
    pruner : Pruner
        The pruner attached to the pruned tree that we wish to visualize.

    Returns
    -------
    Digraph
        Reference to the graphviz object for the visualization.
    '''

    tree = pruner.tree
    g = Digraph('g') #, filename = filename)

    # Make nodes

    for node, ((lChild, rChild), newChild, parent) in enumerate(zip(pruner.originalChildren, tree.children_left, pruner.parents)):

        # The root node never has a pruned parent.

        if parent != -1:
            parentPruned = tree.children_left[parent] == -1
        else:
            parentPruned = False

        nodePruned = newChild == -1

        # Non-leaf nodes (in the original tree) contain information on the cost of pruning away their split.

        if lChild != -1:
            costDecrease = tree.impurity[node] * tree.n_node_samples[node]
            costDecrease -= tree.impurity[lChild] * tree.n_node_samples[lChild]
            costDecrease -= tree.impurity[rChild] * tree.n_node_samples[rChild] 
            costDecrease = "\n" + ("% .1f" % costDecrease) 
        else:
            costDecrease = ""

        # Active nodes are green and non-active nodes are red. Non-active includes nodes that have
        # been pruned, but are still leaves in the prune tree.

        if parentPruned:
            g.node(str(node), str(node) + costDecrease, color = "red", style = 'filled')
   
        else:
            g.node(str(node), str(node) + costDecrease, color = "green", style = 'filled')

    # Make edges

    for node, children in enumerate(pruner.originalChildren):

        lchild, rchild = children 

        if lchild != -1:
            g.edge(str(node), str(lchild))

        if rchild != -1:
            g.edge(str(node), str(rchild))

    return g
```

## [Download example.py Here]({{site . url}}/assets/2018-09-14-files/example.py)
## [Download prune.py Here]({{site . url}}/assets/2018-09-14-files/prune.py)
