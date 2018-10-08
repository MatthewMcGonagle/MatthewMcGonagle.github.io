---
layout : post
title : Random Traversal of A Binary Tree
date : 2017-11-21
tags: Python
---

## [Download the Source Code for this Post]({{site . url}}/assets/2017-11-21-randomTraversal.py)

In this post, we will look at the probability distribution for the order of the nodes in a binary tree when the tree is traversed in a random order. If `nNodes` is the number of nodes in the tree, then as we are processing the nodes, we are marking them in the order they are processed. So the nodes are marked from 0 to `nNodes - 1`. We will be investigating the probability distribution for these marking for each node in the tree. We will derive a theoretical description of this formula, and we will verify it using numerical simulations in `python`. 

How is the traversal random? As we visit each node, we randomly choose between the three following methods of traversal: 
* preorder traversal - process this node, then the left sub-tree, and then the righ sub-tree.
* inorder traversal - process the left sub-tree, then this node, and then the right sub-tree. 
* postorder traversal - process the left sub-tree, then the right sub-tree, and then this node.

To make sure this is clear, let's consider an example. First consider the following tree where we have labeled the nodes using the letters `A` to `G`.

![Example tree, not processed yet]({{site . url}}/assets/2017-11-21-basicExample.svg)

Let us do a random traversal of this tree. The random traversal starts at the root, node `A`. Suppose as we randomly traverse the tree, we get the following results (a picture of the results is below):
1. `A` : We roll Postorder Traversal. Don't mark the root yet. First process the left and right sub-trees.
2. `B` : We roll Preorder Traversal. So we mark node `B` as the first visited as `0`. Then process sub-trees.
3. `D` : We roll Inorder Traversal, however it doesn't matter. At the last level of the tree, the roll doesn't affect anything. Mark node `D` as `1`.
4. `E` : We roll Preorder Traversal, but it doesn't matter. Mark node `E` as `2`.
5. `C` : We roll Inorder Traversal. First process left-subtree.
6. `F` : We roll Postorder Traversal, but it doesn't matter. Mark node `F` as `3`.
7. `C` : We now return to node `C` and mark it as `4`.
8. `G` : We roll Postorder Traversal, but it doesn't matter. Mark as node `G` as `5`.
9. `A` : Finally we return to node `A` since we are done processing its sub-trees. We mark node `A` as `6`.

![Random Traversal Results]({{site . url}}/assets/2017-11-21-basicExample2.svg)

We will consider a binary tree with a given number of levels `nLevels`. Every node on these levels will exist (i.e. the tree will not be missing any nodes). This is for simplicity, but our theoretical conversation can be adapted to the case where nodes are missing.

We will be investigating the probability distribution of these order markings of the nodes when the traversal probabilities at each node are uniform (so 1/3), and the choices at each node are independent.

## Theoretical Formula

Let's consider the theoretical formula for the probability distribution of a given node. Let's consider an example; let's work with a tree that has five levels and a particular node in the tree. We will consider the distribution of node `D` in level `3` (recall that the root is at level `0`) as marked in the diagram below:

![Diagram of Example]({{site . url}}/assets/2017-11-21-choices.svg)

The probability distribution of the order markings depend on the following:
1. The number of ancestors of node `D` that have node `D` in their right sub-tree. For our example, these are nodes `A` and `C`. The right edges below them are marked `red` to indicate so.
2. For the above nodes in item 1, the size of their left sub-trees. As marked in the diagram, we see that node `A` has a left sub-tree of size 15 nodes. The node `C` has a left sub-tree of 3 nodes.
3. The number of ancestors of node `D` that have node `D` in their left sub-tree. For our example, this is only node `B` marked in our diagram. The left edge below `B` is marked `blue` to indicate this.
4. The sizes of the left sub-tree and right sub-tree of node `D` (the node we are computing the distribution for). We have marked the left and right edges below node `D` as `purple`. We see that both of these sub-trees of `D` have only 1 node each.

To understand the distribution at node `D`, there are a couple things that need to be understood:
* The nodes in the left sub-trees of the ancestors described in item 2 will always be processed before node `D`. This is due to the fact that no matter which choice of traversals in the ancestors are made, the left-subtree nodes will be processed before the right sub-tree nodes. The right sub-tree nodes of these ancestors always include node `D`.
* Similarly, all of the nodes in the right sub-trees of the ancestors described in item 3 will always be processed after node `D`. This is due to the fact that for such ancestors, node `D` is always in their left sub-tree.
* Every ancestor described in item 1 has the same probability of being processed before node `D`. Since node `D` is in their right sub-trees, these ancestors are processed before `D` if these ancestors roll a Preorder Traversal or an Inorder Traversal. So they have a `2/3` probability of being processed before `D`.
* Every ancestor described in item 3 has the same probabilty of being processed before node `D`. Since node `D` is in their left sub-tree, these ancestors are processed before `D` only if these ancestors roll a Preorder Traversal. So they have a `1/3` probability of being processed before `D`.

Now, let's compute the probability distribution at `D`. We will write this as a sum of random variables. First let use denote a binomially distributed random variable with probability p as `Binom(n, p)`, which represents the number of heads in `n` independent tosses of a coin with probability `p` of heads. Also let `Unif()` be a uniformly distributed random variable taking the three values `0`, `1`, and `2`. Then we have that the random variable `N(D)` for the order of node `D` during a random traversal is given by
```
N(D) = 18 + Binom(2, 2/3) + Binom(1, 1/3) + 1 * Unif().
```
Let us discuss each term:
* The constant `18` comes from the total number of nodes in the left sub-trees of the ancestors in item 1.
* The term `Binom(2, 2/3)` comes from the `2` ancestors of `D` in item 1.
* The term `Binom(1, 1/3)` comes from the `1` anestor of `D` in item 3.
* The term `1 * Unif()` comes from the sub-trees of `D` of size `1`.

In general, if a node has:
* `R` ancestors in item 1,
* `Offset` total number of nodes in the left sub-trees of those ancestors.
* `L` ancestors in item 3,
* `nSubtree` number of nodes in the each sub-tree of the particular node we are computing the distribution for.

Then the random variable for the order number of our node is given by:
```
N = Offset + Binom(R, 2/3) + Binom(L, 1/3) + nSubtree * Unif()
```
Note that `Offset` is just a constant that results in shifting the random variable `N`.

The above formula for `N` is enough to compute a probability distribution for each node that we can graph and compare to numerical simulation. A more explicit formula really won't add any more clarity.

## Classes for Computing the Theoretical Model and Simulation

Now we will investigate computing the theoretical model in `python`, and we will compare it to a numerical simulation. We will build some classes in `python` to help us do so.

First, let's import the modules we will need:
``` python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as colors
```
Now, let's make a class for the nodes in our tree.
``` python
class Node:
    '''
    Class for nodes in tree.

    Members
    -------
    self.data : Any type.
        The data held by this node.
    self.left : None or class Node.
        Holds reference to the left child. Should be set to None if there is no left child.
    self.right : None or class Node.
        Holds reference to the right child. Should be set to None if there is no right child.
     '''

    def __init__(self, data = None):
        '''
        Initializer.

        Left and right children are defaulted to being None. The data held at the node may be passed
        as a parameter, but if nothing is passed then self.data = None.
        
        Parameters
        ----------
        self : 
            Reference to this node that is always passed.
        data : Any type.
            The data for this node to hold, defaults to None.
        '''

        self.data = data
        self.left = None 
        self.right = None 
```

Next, we will make a class for making a tree holding the data of our numerical simulations. Each node of the tree will hold a `numpy` array holding the counts of the number of times the node has been assigned each possible order number. Running one instance of the random traversal is done by calling the member function `SimulationTree.randTraversal()`.
``` python
class SimulationTree:
    '''
    Class for running simulation of random traversal of binary tree.

    Class sets up nodes for simulation of random traversal of binary tree. Each node holds a numpy array
    of counts that are recorded while running simulation. The tree is of specified number of levels
    (i.e. height) and is full. That is, for a specified number of levels, the tree contains all of its nodes.

    Member Variables
    ----------------
    nNodes : int
        The number of nodes in the tree.
    maxLevel : int
        The number identifying the largest level the tree. The root is at level 0.
    root : Node
        The root node of the tree.
    __rollValue : dict
        The values of random rolls to use to determine whether to use a preorder traversal, inorder traversal,
        or postorder traversal.
    '''

    def __init__(self, nLevels):
        '''
        Initializer.

        Sets up all of the nodes in the tree and all of the member variables.
    
        Parameters
        ----------
        nLevels : int
            The number of levels in the tree, including the root. So if nLevels is 1, then the tree consists
            of only the root.
        '''
        self.nNodes = 2**nLevels - 1
        self.maxLevel = nLevels - 1
        self.root = Node(np.zeros(self.nNodes))

        # Create subtrees of root.
        self.__addChildren(0, self.root)

        self.__rollValue = {'preorder' : 0, 'inorder' : 1, 'postorder' : 2}

    def __addChildren(self, level, node):
        '''
        Creates subtrees by adding children.

        Continues recursively adding children nodes until we are at one level below the maximum level.

        Parameters
        ----------
        self : self
        level : int
            The level of the current node. The root should start at 0.
        node : Node
            The current node to add children to.        
        '''
        node.left = Node(np.zeros(self.nNodes))
        node.right = Node(np.zeros(self.nNodes))

        # Check if we still need to add children.
        if level < self.maxLevel - 1:
            self.__addChildren(level + 1, node.left) 
            self.__addChildren(level + 1, node.right)

    def __randTraversal(self, n, node):
        '''
        Perform random traversal decision at node.
        
        Randomly decide whether to do a preorder, inorder, or postorder traversal decision at this node. 
        For the next node we process, we will call it n (that is add +1 to the n statistic at that node).
        For a preorder, that will be this node. For inorder and postorder, the next node processed will be in the
        left subtree (if non-empty). However, for inorder, this node will be processed after the left sub-tree;
        for postorder, this node will be processed after the right sub-tree. 

        Parameters
        ----------
        self : self
        n : int
            We will call the next processed node n.
        node : Node
            The current node to which we make a random traversal decision. 

        Returns
        -------
        Int
            After processing this node, the number to mark the next node processed.
        '''

        # For empty leaves, don't do anything.

        if node == None:
            return n

        # Randomly select which traversal patter to use at this node.

        diceroll = np.random.randint(0, 3)

        # For roll of preorder, process this node, then the left sub-tree, and then the right sub-tree. 

        if diceroll == self.__rollValue['preorder']:
            node.data[n] += 1
            newN = self.__randTraversal(n+1, node.left)
            newN = self.__randTraversal(newN, node.right) 
            return newN

        # Else, if we roll inorder traversal, then process left sub-tree, this node, and then the
        # the right sub-tree.

        elif diceroll == self.__rollValue['inorder']:
            newN = self.__randTraversal(n, node.left)
            node.data[newN] += 1 
            newN = self.__randTraversal(newN + 1, node.right)
            return newN

        # Else we have a post order roll. So we process left sub-tree, right sub-tree, and then this node.
        else: 
            newN = self.__randTraversal(n, node.left)
            newN = self.__randTraversal(newN, node.right)
            node.data[newN] += 1
            return newN + 1

    def randTraversal(self):
        '''
        Do a random traversal of the entire tree starting at the root.

        We will mark each node from 0 to self.nNodes - 1. However, we do this processing in a random order.
        At each node, we randomly decide to process this node in either a preorder, inorder, or postorder
        fashion. Since we are interested in the statistics, we increment the count of that number at the node
        contained in the node.data array. This function will only do one traversal, incrementing one single
        statistic for each node.

        Parameters
        ----------
        self : self
        '''
        checkN = self.__randTraversal(0, self.root)

    def getDataList(self):
        '''
        Gets statistics of all the nodes.

        Groups the statistics of the nodes into lists for each level. So we get a list of lists of
        numpy arrays.

        Parameters
        ----------
        self : self

        Returns
        -------
        List of list of numpy arrays.

            A list of list of numpy arrays where each list of numpy arrays 
            holds the statistics of all the nodes in one level. That is, 
            each of these lists has length equal to the number of nodes in the level, and
            every array has length equal to the number of nodes in the tree. 
        '''

        dataList = []
        nodeList = [self.root]
        while(nodeList != []):

            nextLevelNodes = []
            thisLevelData = []
            for node in nodeList:
               
                thisLevelData.append(node.data)
 
                if node.left != None:
                    nextLevelNodes.append(node.left)

                if node.right != None:
                    nextLevelNodes.append(node.right)

            dataList.append(thisLevelData)
            nodeList = nextLevelNodes

        return dataList
```

Now, we will construct come classes for helping us compute the theoretical distribution. First, we create a class
representing a random variable. This class will allow us to add together two independent random variables without much effort. We also add a function `RandomVar.shift()` to allow us to shift the values in our random variable.
``` python
class RandomVar:
    '''
    Class for representing random variables that can take a finite number of values. 
    Will allow us to easily add together two independent random variables.

    Member Variables
    ----------------
    self.values : numpy array
        Should hold array of values that the variable can obtain. May or may not include values for the
        probability is 0.
    self.probs : numpy array, dtype = float
        Array of values that contains the probabilities of corresponding values in self.values. The user
        is responsible for making sure they are a valid probability distribution.
    '''

    def __init__(self, values, probs):
        '''
        Initializer.

        Sets up values of random variable and their respective probabilities.
    
        Parameters
        ----------
        self : self
        values : numpy array
            Array of values the random variable may take. This can include values that have probability 0.
        probs : numpy array
            Array of probabilities for the respective values of the random variables.
        '''

        if len(values) > 0:
            self.values = values
        else:
            raise ValueError("values is empty. RandomVar must take on values")

        if len(probs) > 0:
            self.probs = probs
        else:
            raise ValueError("probs is empty. RandomVar must take on probabilies")

    def add(self, randVar2):
        '''
        Adds this random variable to another independent random variable.

        Computes the values and probabilities for the random variable that corresponds to
        summing this random varaible with another independent random variable.
        
        Parameters
        ----------
        self : self
        randVar2 : RandomVar
            The other random variable (assumed to be independent of this random variable).
    
        Returns
        -------
        RandomVar
            The random variable representing the sum.
        '''

        # We first store values of new random variable in a dictionary. So as we sum our values,
        # we can check to see if we have already seen this sum valued and then add to its
        # already existing probability.

        newprobs = {}
        for val, prob in zip(self.values, self.probs):
            for val2, prob2 in zip(randVar2.values, randVar2.probs):
                newval = val + val2
                newprob = prob * prob2 

                # If the value already exists for our new random variable, then we need to
                # add in the new probability to probability in the dict. Else we just add 
                # the probability to the dict under the key given by the value. 

                if newval in newprobs.keys():
                    newprobs[newval] += newprob
                else:
                    newprobs[newval] = newprob 

        newvals = np.array(list(newprobs.keys()))
        newprobs = np.array(list(newprobs.values()))
        return RandomVar(newvals, newprobs)

    def shift(self, dvals):
        '''
        Shifts the values of the random variable by a fixed amount.

        Parameters
        ----------
        self : self
        dvals : Number
            The amount to shift all of the values of the random variable.
        '''

        self.values += dvals

```

We will want to use binomial coefficients for constructing the binomially distributed random variables for our model. To do so, we construct a class for pre-computing a complete binomial coefficient table. We will be using the entire table for our tree so after we compute it, we keep it until we are done computing the theoretical model for the entire tree. The binomial coefficients will be computed using dynamic programming in the standard manner.

``` python
class BinomialTable:
    '''
    Class for storing a computed binomial coefficient table. These values are computed in a dynamic
    programming manner. The table computed is square since we will be using all of the binomial
    coefficients with parameters less than some value when computing our model tree.

    Member Variables
    ----------------
    n : int
        When thinking the binomial coefficients as storing the number of subsets of a set of size n,
        then this is the n.
    table : 2d numpy array of int, shape (n, n)
        The table of computed values of binomial coefficients. 
    '''

    def __init__(self, n):
        '''
        Initializer

        Compute the table.
        Parameters
        ----------
        self : self
        n : int
            We compute a table for binomial 0 choose 0 to binomial n choose n.
        '''

        self.n = n
        self.__computeTable(n)

    def __computeTable(self, n):
        '''
        Compute the table of binomial coefficients.

        We make the computation in a dynamic programming manner.
    
        Parameters
        ----------
        self : self
        n : int
            We compute the table for binomial 0 choose 0 to binomial n choose n.
        '''

        # Initialize the table and precompute the case of n = 0.

        self.table = np.zeros((n + 1, n + 1))
        self.table[0,0] = 1
        if n == 0:
            return 

        # Now we use the fact that C(n, k) = C(n-1, k) + C(n - 1, k - 1). This fact comes from determing
        # the two cases of whether the last element is in the k-subset.
        for nballs in range(1, n + 1):
          self.table[nballs, 0] = 1
          self.table[nballs, 1:] = self.table[nballs - 1, 1:] + self.table[nballs - 1, :-1] 

    def getRandomVar(self, n, p):
        '''
        Gets the binomial distribution for n trials with probability of success p.

        Parameters
        ----------
        self : self
        n : int
            The number of trials in the binomial distribution. The random variable will be
            computed from the nth row of the precomputed binomial coefficient table. 
        p : float
            The probability of success for each independent trial.
        
        Returns
        -------
        RandomVar
            Random variable for the desired binomial distribution.
        '''

        if n > self.n:
            raise ValueError("n is too large for binomial table when creating random binomial variable")

        q = 1.0 - p # The probability of failure.
        vals = np.arange(n+1) 
        probs = self.table[n, :n+1]

        # Since the table has int values and the probabilties are supposed to be float values, we
        # are wary of using *=.

        probs = probs * p**vals
        probs = probs * q**(n - vals) 
        return RandomVar(vals, probs)
```

Now we will make a class for constructing a tree such that at each node it holds the theoretical probability distribution of the random orders for the node.
``` python
class ModelTree:
    '''
    Class for computing the theoretical probabilities for a random traversal of our tree. We assume an
    even probability of choosing preorder, inorder, and postorder at each node. Furthermore, we assume
    that the choices at the nodes are independent.

    Member Variables
    ----------------
    maxLevel : int
        The number of the largest level. The root is located at level 0.
    nNodes : int
        The number of nodes in the tree.
    leftEdgeProb : float
        When an ancestor that holds the current node in its left sub-tree, the probability that
        the ancestor will be processed before the current node. Happens only for preorder of ancestor. 
    rightEdgeProb : float
        When an ancestor that holds the current node in its right sub-tree, the probability that
        the ancestor will be processed before the current node. Happens for preorder and inorder
        of ancestor.
    binTable : BinomialTable
        The binomial table we will use to compute binomial distributions for our model.
    root : Node
        The root of the tree.
    ''' 

    def __init__(self, nLevels):
        '''
        Initializer
        
        Parameters
        ----------
        self : self
        nLevels : int
            The number of levels in the tree. The levels will be enumerated from 0 to nLevels - 1.
        '''

        if nLevels < 0:
            raise ValueError("Tree should have non-negative number of levels.")

        self.maxLevel = nLevels - 1
        self.nNodes = 2**nLevels - 1 
        self.leftEdgeProb = 1.0 / 3.0
        self.rightEdgeProb = 2.0 / 3.0
        self.binTable = BinomialTable(nLevels) 
        self.root = Node()

        # Set up the nodes of the tree while computing the theoretical model at each node.
        self.__addRandVars(self.root, 0, 0, 0, 0)
        
    def __finalNodeVar(self, nodeLevel):
        '''
        Computes random variable for offsets coming from nodes in sub-tree below current node.

        For preorder, these is no offset as this node is processed before sub-trees.
        For inorder, the number of nodes in left sub-tree come before this node.
        For postorder, both the nodes in the left sub-tree and the right sub-tree come
        before this node.

        Parameters
        ----------
        self : self
        nodeLevel : int
            The level of the current node we are computing the model for.

        Returns
        -------
        RandomVar
            The random variable for the offset coming from the nodes below the current node
            we are computing the model for.
        '''

        nSubLevels = self.maxLevel - nodeLevel 
        nSubNodes = 2**nSubLevels - 1
        values = np.array([0, nSubNodes, 2 * nSubNodes])

        # All of these values are equally likely.
        probs = np.full(values.shape, 1.0 / len(values)) 

        return RandomVar(values, probs)

    def __leftEdgeVar(self, nLeftAbove):
        '''
        Gets random variable for offsets coming from ancestors where the current node is sitting in
        their left sub-tree. This is a binomially distributed random variable. It counts which of
        these ancestors are processed before the current node.
        
        Parameters
        ----------
        self : self
        nLeftAbove : int
            The number of ancestors of the current node such that the current node is sitting in their
            left sub-tree.

        Returns
        -------
        RandomVar
            The random variable representing the offset coming from the ancestors. This is a binomially
            distributed for which of these ancestors contributes one single offset. 
        '''

        var = self.binTable.getRandomVar(nLeftAbove, self.leftEdgeProb) 
        return var 

    def __rightEdgeVar(self, nRightAbove):
        '''
        Gets random variable that counts for the ancestors of the current node which have the current node
        sitting in their right sub-tree. The variable counts how many of these ancestors are processed 
        before the current node.

        Parameters
        ----------
        self : self
        nRightAbove : int
            The number of ancestors such that the current node is sitting in its right sub-tree.

        Returns
        -------
        RandomVar
            The random variable counting how many of these ancestors are processed before the current node.
        '''

        var = self.binTable.getRandomVar(nRightAbove, self.rightEdgeProb)
        return var 

    def __convertRandVar(self, randVar):
        '''
        Converts the random variable of probabilities of value for a node into an array of probabilities
        for all values between 0 and nNodes - 1.

        Parameters
        ----------
        self : self
        randVar : RandomVar
            Random variable holding statistics of some node.

        Returns
        -------
        Numpy array of floats of length self.nNodes
            Array holding all probabilities for all values, including those for which the node has
            probability 0.
        '''

        probs = np.zeros(self.nNodes)
        probs[randVar.values] = randVar.probs
        return probs
    
    def __addRandVars(self, node, level, shift, nLeftAbove, nRightAbove):
        '''
        Recursively compute the model probabilities for order of being processed in random tree
        traversal, add children, and then do so for children.

        Parameters
        ----------
        self : self
        node : Node
            Current node.
        level : int
            The level of the current node. The root is at level 0.
        shift : int
            The amount to shift the random variable for this node. Represents the total size of left
            sub-trees for ancestors that contain the current node in their right sub-tree.
        nLeftAbove : int
            The number of ancestors of this node such that this node is in the left sub-tree of the
            ancestor.
        nRightAbove : int
            The number of ancestors of this node such that this node is in the right sub-tree of the
            ancestor.
        '''

        # First set up random variable for shift. Shift always happens with probability 1.0.
        randVar = RandomVar(np.array([shift]), np.array([1.0]))

        # The rest of the random variable is the sum of contributions from sub-trees of this
        # this node and the ancestors of this node. For each case, check that these are 
        # non-empty before adding in.
        
        if level < self.maxLevel:
            randVar = randVar.add(self.__finalNodeVar(level))

        if nLeftAbove > 0:
            randVar = randVar.add(self.__leftEdgeVar(nLeftAbove))

        if nRightAbove > 0:
            randVar = randVar.add(self.__rightEdgeVar(nRightAbove))

        node.data = self.__convertRandVar(randVar)

        # If we aren't on the last level, then we need to compute for the children.

        if level < self.maxLevel:

            nSubLevels = self.maxLevel - level

            # For the right child, we need to add in size of left sub-tree to shift.

            newShift = 2**nSubLevels - 1 + shift 

            node.left = Node()
            node.right = Node()
            self.__addRandVars(node.left, level + 1, shift, nLeftAbove + 1, nRightAbove)
            self.__addRandVars(node.right, level + 1, newShift, nLeftAbove, nRightAbove + 1)

    def getDataList(self):
        '''
        Gets probability distributions of all the nodes.

        Groups the distributions of the nodes into lists for each level. So we get a list of lists. 

        Parameters
        ----------
        self : self

        Returns
        -------
        List of list of numpy arrays.

            A list of list of numpy arrays where each list of numpy arrays 
            holds the probabilty distributions of all the nodes in one level. That is, 
            each of these lists has length equal to the number of nodes in the level, and
            every array has length equal to the number of nodes in the tree. 
        '''

        dataList = []
        nodeList = [self.root]
        while(nodeList != []):

            nextLevelNodes = []
            thisLevelData = []
            for node in nodeList:
               
                thisLevelData.append(node.data)
 
                if node.left != None:
                    nextLevelNodes.append(node.left)

                if node.right != None:
                    nextLevelNodes.append(node.right)

            dataList.append(thisLevelData)
            nodeList = nextLevelNodes

        return dataList
```
## Testing Some of Our Classes

Here let us briefly run some tests of our classes and their member functions to make sure they are coded properly. First, let's try computing a binomial table to make sure we are correctly computing our binomial coefficients.
``` python
######### Test of BinomialTable class and RandomVar class

binomTable = BinomialTable(5)
print('Binomial Table for n = 5 is\n', binomTable.table)
print('RandomVar Binom(3, 0.25) values are \n', binomTable.getRandomVar(3, 0.25).values, 
      '\nProbabilities are\n', binomTable.getRandomVar(3, 0.25).probs)
```
We get
```
Binomial Table for n = 5 is
 [[  1.   0.   0.   0.   0.   0.]
 [  1.   1.   0.   0.   0.   0.]
 [  1.   2.   1.   0.   0.   0.]
 [  1.   3.   3.   1.   0.   0.]
 [  1.   4.   6.   4.   1.   0.]
 [  1.   5.  10.  10.   5.   1.]]
RandomVar Binom(3, 0.25) values are
 [0 1 2 3]
Probabilities are
 [ 0.421875  0.421875  0.140625  0.015625]
```

The table is definitely correct (just consult Pascal's triangle). One can also double check that the binomial random variable values and probabilities are correct, but we won't go through the details here.

Next, let's double check the addition of random variables.

``` python
randVar1 = binomTable.getRandomVar(1, 0.5)
print('RandomVar Binom(1, 0.5) values = \n', randVar1.values, 
       '\nProbabilities are = \n', randVar1.probs)
randVar2 = binomTable.getRandomVar(2, 0.5) 
print('\nRandomVar Binom(2, 0.5) values = \n', randVar2.values, 
      '\nProbabilites are = \n', randVar2.probs)
randVar3 = randVar1.add(randVar2)
print('\nBinom(1, 0.5) + Binom(2, 0.5) values = \n', randVar3.values, 
       '\nProbabilities are = \n', randVar3.probs)
```

We get that 

```
RandomVar Binom(1, 0.5) values =
 [0 1]
Probabilities are =
 [ 0.5  0.5]

RandomVar Binom(2, 0.5) values =
 [0 1 2]
Probabilites are =
 [ 0.25  0.5   0.25]

Binom(1, 0.5) + Binom(2, 0.5) values =
 [0 1 2 3]
Probabilities are =
 [ 0.125  0.375  0.375  0.125]
```

So we see that the addition of random variables is working correctly.

## Computing the Model and Running The Simulation

Let's first compute the model; we will use a tree with five levels.
``` python
nLevels = 5 # Number of levels in tree.
nTraversals = 7000 # Number of times to run simulation.
np.random.seed(20171121) 

# Compute model.

model = ModelTree(nLevels)
modelList = model.getDataList()
```

Now, set up the simulation tree and run it for `nTraversals`.
``` python
# Set up tree for doing simulation.

statTraversals = SimulationTree(nLevels)

# Do nTraversals random traversals.
 
printI = 0
for i in range(nTraversals):

    statTraversals.randTraversal()

    if printI > nTraversals / 10:
        print('Finished ' + str(i))
        printI = 0
    printI += 1

statList = statTraversals.getDataList()
```

Now, let's graph the results of the theoretical model and simulation. We will graph the distributions of the nodes in each level. The distribution for each node will be graphed with separate colors. We horizontally shift the data for the theoretical model a little bit to the right so that it can be easily seen next to the simulated data.

![Graph of level 0]({{site . url}}/assets/2017-11-21-graphs/level0.svg)

![Graph of Level 1]({{site . url}}/assets/2017-11-21-graphs/level1.svg)

![Graph of Level 2]({{site . url}}/assets/2017-11-21-graphs/level2.svg)

![Graph of Level 3]({{site . url}}/assets/2017-11-21-graphs/level3.svg)

![Graph of Level 4]({{site . url}}/assets/2017-11-21-graphs/level4.svg)

We see that the simulated data and theoretical model are in very close agreement.

## [Download the Source Code for this Post]({{site . url}}/assets/2017-11-21-randomTraversal.py)

