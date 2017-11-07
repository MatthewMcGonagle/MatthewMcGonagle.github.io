---
layout: post
title: 'Drawing a Binary Tree Using Gradient Descent and a Potential Function'
date: 2017-11-03
---

## [Download The Source Code for this Post]({{site . url}}/assets/2017-11-03-potentialTree.py)

In this post, we will look at using gradient descent combined with the use of potential functions to decide on the spacing for drawing a binary tree. For example, we use this to draw the following tree:

![Final Binary Tree Drawing]({{site . url}}/assets/2017-11-03-graphs/final.svg)

That is, we iteratively update the horizontal positions of the nodes in our tree that so that they are more appropriately spaced out. We start with the following initial positions of the nodes:
![Pic of Initial Positions of Nodes]({{site . url}}/assets/2017-11-03-graphs/iterate0.svg)
Then we update the horizontal positions of each node using gradient descent on a potential coming from these positions. The horizontal potential will be set up to have the following properties:
* The nodes of the graph will be kept from shooting off to infinity.
* The nodes on each level will push each other around so that they appear in the appropriate left and right order. Furthermore, when in the correct order they repel each other; they are unable to come too close to each other.
* Each parent is attracted to both their left child and right child; However, they are also repelled by their children so that they don't stack. However this repulsion is only in the appropriate directions for left and right children to be on the correct side of the parent. Note, these effects are only for horizontal positions.

Here are some selected iterates of the gradient descent so may get an idea of how the positions of the nodes evolve.

![Pic of Iterate 10]({{site . url}}/assets/2017-11-03-graphs/iterate10.svg)

![Pic of Iterate 20]({{site . url}}/assets/2017-11-03-graphs/iterate20.svg)

![Pic of Iterate 100]({{site . url}}/assets/2017-11-03-graphs/iterate100.svg)

![Pic of Iterate 500]({{site . url}}/assets/2017-11-03-graphs/iterate500.svg)

Before we get started, let's import `numpy` and `pyplot`; also let's set up the figure size we will be using to draw our graphs with `pyplot`.

``` python
import matplotlib.pyplot as plt
import numpy as np

figsize = (11, 5)
```

In the next section, we will set up a class for constructing our binary search tree. We will be inserting a random numbers into our tree; so we won't be bothering with balancing our tree. We will do the most basic binary search tree insertion. So the following section could be skipped if you are confident enough with the construction of a binary search tree.

## Setting Up Our Tree

First, we set up a class for the nodes in our tree.

``` python
class Node:

    def __init__(self, val, left = None, right = None):
        self.left = left 
        self.right = right 
        self.val = val
```

Now let's setup our class for creating a binary search tree. The purpose of this post isn't to explain how binary search trees work, so I will only show the code for the curiosity of the reader. Understanding the details of this class isn't really necessary.

``` python
class Tree:

    # root should be initialized as a Node.    
    def __init__(self, root = None):
        self.root = root 

    # Private member function for doing recursion to insert a value into the tree.

    def __insertsubtree(self, val, root):
        if root.val == val:
            return
        elif val < root.val:
            if root.left == None:
                root.left = Node(val)
            else:
                self.__insertsubtree(val, root.left)
        else:
            if root.right == None:
                root.right = Node(val)
            else:
                self.__insertsubtree(val, root.right)

    # Public member function for inserting a value into the tree.
    
    def insert(self, val):
        if self.root == None:
            self.root = Node(val)
        else:
            self.__insertsubtree(val, self.root)

    # Private member function for printing a particular level of the tree.
    # This is a simple ASCII print to the terminal; this has nothing to do
    # with the spacing we aim to make in this program. It is simply for
    # double checking our results.

    def __printlevel(self, nodelist):
        nextlevel = []
        levelstr = ""
        for node in nodelist:
            if node != None:
                levelstr += str(node.val) + " "
                nextlevel.append(node.left)
                nextlevel.append(node.right)
            else:
                levelstr += "None "
        print(levelstr)
        return nextlevel
            
    # Public member function for printing the entire tree. 
    # This is a simple ASCII print to the terminal; this has nothing to do
    # with the spacing we aim to make in this program. It is simply for
    # double checking our results.

    def print(self):

        if self.root == None:
            print("None")
            return
        level = [self.root]
        while level != [] :
            level = self.__printlevel(level)
```

Now let's construct our tree and do a very rough ASCII printout of its levels. We insert 100 random integers in the range 0 to 99 (inclusive) into our tree.
``` python
np.random.seed(20171102)
nums = np.random.randint(0, 100, size = 100)
print(nums)
tree = Tree()
for num in nums:
    tree.insert(num)  
tree.print()
```
We get the following print out of the tree:
``` 
80
39 83
32 70 81 98
8 37 51 79 None None 94 99
5 27 33 38 40 55 75 None 88 95 None None
0 6 13 29 None 35 None None None 43 53 61 74 None 85 92 None 97
None 3 None None 11 18 None None None None 42 49 52 None 59 65 72 None None None 89 None None None
2 4 10 12 16 26 None None 45 None None None 58 None 64 69 None None None 91
None None None None None None None None 15 17 20 None 44 48 56 None 62 None 66 None 90 None
14 None None None None None None None None None None 57 None None None 68 None None
None None None None None None
```
This printout isn't super neat and helpful. However, it will still allow us to double check our results.

## Create Class to get Arrays of Variables for Gradient Descent

Now we will create a class `TreeProcessor` to find the variables we will need to do our gradient descent. First, we will later be using a flat array `positions` of the **horizontal** positions of each node in our tree.

``` python
class TreeProcessor:

    # # self.levelList will be a list of lists. Each member is a list of the nodes
    #       in each level.
    # # self.indices will be a dictionary of lists of lists of integers. Each member of each key value
    #       is a list of the indices in the variables arrays (to be constructed) that corresponds to
    #       to each node in each level. 
    #       indices['level'] will record position array indices of each node in each level.
    #       indices['left'] will record position array indices of the left child of each node in each level.
    #       indices['right'] is similar for right children.
    # # self.NOCHILD is a constant for showing when a node doesn't have a particular
    #       type of child.
    # # self.childI is a variable that will be needed when iterating over recording
    #       the indices of children.
    # # self.edges will be a list of the edges inside a tree.

    def __init__(self):
        self.levelList = None
        self.indices = None
        self.NOCHILD = -1
        self.childI = None 
        self.edges = None
```
Now, let's write some private member functions for determining the indices of children inside the positions array.
``` python
    # Private member function for adding left child to the next level array
    # and also for recording the relative index of where the left child occurs in the
    # array of positions. Relative index is relative to the index of the last node
    # in the current level.

    def __updateLeftChild(self, node, nextLevel, newLeftInd):
        
         if node.left != None:
             nextLevel.append(node.left)
             newLeftInd.append(self.childI)
             self.childI += 1
         else:
             newLeftInd.append(self.NOCHILD)

    # Private member function for adding right child to the next level array
    # and also for recording the relative index of where the right child occurs in the
    # array of positions. Relative index is relative to the index of the last node
    # in the current level.

    def __updateRightChild(self, node, nextLevel, newRightInd):
        
         if node.right != None:
             nextLevel.append(node.right)
             newRightInd.append(self.childI)
             self.childI += 1
         else:
             newRightInd.append(self.NOCHILD)

    # Take the relative index of children and find the absolute index of the
    # children in the position array.

    def __offsetChild(self, childIndex, numPrevious):

        if childIndex != self.NOCHILD:
            return childIndex + numPrevious
        else:
            return childIndex
```
Now let's create a public member function for finding the list of the nodes in each level, and also
the lists of the indices of each node and its children in each level.
``` python
    # Find self.levelList and self.indices for a specific tree. 

    def getLevelInfoLists(self, tree):

        self.indices = {}

        if tree.root == None:
            self.indices['left'] = None
            self.indices['right'] = None
            self.indices['level'] = None
            self.levelList = None
            return

        # Initialize info we will find.

        self.indices['level'] = []
        self.indices['left'] = []
        self.indices['right'] = [] 
        self.levelList = []

        # Initialize loop variables.

        level = [tree.root]
        nodeI = 0
        
        # Keep processing while current level is non-empty.

        while level != []:

            # Initialize info for processing this level.

            nextLevel = []
            newLeftInd = []
            newRightInd = []
            newLevelInd = []
            self.childI = 0

            # For each node in this level, add the appropriate relative position index to the different
            # index lists.

            for node in level:

                newLevelInd.append(nodeI)
                nodeI += 1

                self.__updateLeftChild(node, nextLevel, newLeftInd)
                self.__updateRightChild(node, nextLevel, newRightInd)

            # Turn the relative position indices into absolute indices for position array.

            newLeftInd = [self.__offsetChild(i, nodeI) for i in newLeftInd]
            newRightInd = [self.__offsetChild(i, nodeI) for i in newRightInd]

            # Add list of indices for this level to the appropriate members of our indices dictionary.

            self.indices['level'].append(np.array(newLevelInd)) 
            self.indices['left'].append(np.array(newLeftInd))
            self.indices['right'].append(np.array(newRightInd))                    
            self.levelList.append(level)
            level = nextLevel
```
Finally let's make some functions for finding the edges of our tree and finding the text
representing the value in each node.
``` python
    # Private member function for getting the edge between a parent and its child based on
    # the index of each inside a positions array.

    def __addEdge(self, parenti, childi, positions): 

            parentPos = positions[parenti]
            if childi != self.NOCHILD:
                newPos = positions[childi]
                self.edges.append([parentPos, newPos])

    # Get the edges for the entire tree. The coordinates of the endpoints of each edge come from the
    # array of positions.

    def getEdges(self, positions):
        self.edges = []
        for leveli, leftiS, rightiS in zip(self.indices['level'], self.indices['left'], self.indices['right']):
            for parenti, lefti, righti in zip(leveli, leftiS, rightiS):
                self.__addEdge(parenti, lefti, positions)
                self.__addEdge(parenti, righti, positions)
        self.edges = np.array(self.edges)

    # Get an array of the text representing the value of each node. The order is the same as
    # the order in self.levelList

    def getNodeText(self):

        textList = []

        for level in self.levelList:
            for node in level:
                textList.append(str(node.val))

        return textList
```

## Create Functions for Gradient Descent

Now we will create our functions that will incrementally update the positions of the nodes in our graph. The positions are stored in a flat array of floats which we denote by `positions`. Every function will use a dictionary of parameters denoted by `params` that stores values that adjust the weights of the effects of each part of our gradient descent.

First, we create an update that keeps the nodes in the graph from shooting off to infinity.
``` python
# Update to keep the graph from flying off to inifinity.

def getUpdateForBounded(levelPos):

    return -levelPos
```

Next, we get updates to the horizontal position of each node that comes from the nodes on the same level. Each node pushes other nodes in the correct direction according to which order they should appear. The magnitude depends on the inverse distance squared.
``` python
# Update coming from nodes on same level. They repel each other.
  
def getUpdateFromSameLevel(levelPos):

    # Get differences in index between all possibilities of two elements in level (including repeats).
    
    diffIndex = np.arange(len(levelPos))
    diffIndex = diffIndex[:, np.newaxis] - diffIndex
    diffIndex = np.sign(diffIndex)
    mask = diffIndex != 0
    
    # Get differences in position between all possibilities of two elements in level (including repeats).
    
    diffMatrix = levelPos[:, np.newaxis] - levelPos
    diffMatrix[mask] =  1 / diffMatrix[mask]**2 
    diffMatrix = diffMatrix * diffIndex

    return diffMatrix.sum(axis = -1)
```
Now, let's get an update for each node coming from its left child. Note, the function takes as parameter the nodes on a level that have a left child. That is we have already filtered out those nodes on a level that don't have left children. For each node that has a left child, there is quadratic potential that results in an attraction between the node and its left child. However, there is also a repulsion that pushes the node the right of its left child.
``` python
def getUpdateFromLeftChildren(parentPos, lChildPos, childRepulsion):  

    lChildOnRight = lChildPos > parentPos
    lChildOnLeft = ~lChildOnRight

    # Put in an attraction between the parent and its left child.

    diffPos = lChildPos - parentPos
    update = diffPos

    # Add in a repulsive effect for the parent and its left child. This effect will push the parent to the right
    # past the child. Furthermore, it is different in the two cases that the child is on the parent's left or right.
    # However, the repulsion changes continuously.

    update[lChildOnRight] += childRepulsion 
    update[lChildOnLeft] += (1 / np.sqrt(childRepulsion) - diffPos [lChildOnLeft] )**-2

    return update
```
Now, let's do the same for the right child. This time however, the repulsion pushes the parent to the left of the right child.
``` python
def getUpdateFromRightChildren(parentPos, rChildPos, childRepulsion):

    # Get masks for which side of the parent the child is on.
    rChildOnLeft = rChildPos < parentPos
    rChildOnRight = ~rChildOnLeft

   
    # We put in the effects of an attractive potential to bring together the parent with its right child.

    diffPos = rChildPos - parentPos
    update = diffPos

    # However, we put in a repulsive effect to keep them from stacking; furthermore, this effect will push the
    # the parent to the left past the child. Note that we set up the repulsion differently from when the
    # the child is on the left vs on the right; however, it is continuous across this change.

    update[rChildOnLeft] += - childRepulsion
    update[rChildOnRight] += - (1 / np.sqrt(childRepulsion) + diffPos[rChildOnRight] ) ** -2

    return update
```
Now let's make a function to combine all of these updates into one incremental update.
``` python
def updatePos(positions, indices, params):

    baseIndex = 0
    allupdates = np.array([])

    for leveli, lefti, righti in zip(indices['level'], indices['left'], indices['right']):

        levelPos = positions[leveli]

        # Update to keep the graph from flying off to infinity.

        update = params['bounded'] * getUpdateForBounded(levelPos) 

        # Updates for spacing out nodes.

        update += params['level'] * getUpdateFromSameLevel(levelPos)

        # Update coming from left children

        hasLeft = lefti > -1
        parentPos = levelPos[hasLeft]
        lChildPos = positions[lefti[hasLeft]]

        newUpdate = getUpdateFromLeftChildren(parentPos, lChildPos, params['childRepulsion'])
        update[hasLeft] += params['children'] * newUpdate 

        # Update coming from right children 

        hasRight = righti > -1
        parentPos = levelPos[hasRight]
        rChildPos = positions[righti[hasRight]]

        newUpdate = getUpdateFromRightChildren(parentPos, rChildPos, params['childRepulsion'])
        update[hasRight] += params['children'] * newUpdate

        # Add updates for this level to updates for entire tree.

        allupdates = np.concatenate([allupdates, update])
    
    return positions + params['learning_rate'] * allupdates
```

## Create Functions to Draw our Tree

First we create a function to get the vertical positions of each node in the tree. This does not change as we update the current positions.
``` python
# Function for getting vertical position of each node, in the order of levelList.

def getYPos(levelList):
    ypos = np.array([])
    pos = 0
    for level in levelList: 
        newpos = np.full(len(level), pos)
        ypos = np.concatenate([ypos, newpos])
        pos -= 1
    return ypos
```

Now let's make our function for drawing our tree given its position information.
``` python
# Function for drawing the tree according to the horizontal position given by
# positions and the vertical position given by ypos. We use processor to find the edges.

def drawTree(positions, ypos, processor):

    coords = np.stack([positions, ypos], axis = -1)
    processor.getEdges(coords)
    nodeNames = processor.getNodeText()
    
    plt.clf()
    plt.plot(processor.edges.T[0], processor.edges.T[1], color = 'black', lw = 2, zorder = 1)
    plt.scatter(coords.T[0], coords.T[1], zorder = 2, s = 300, color = '#00FF00') 
    ax = plt.gca()
    ax.set_xlim(left = np.amin(coords[:,0]) - 0.5, right = np.amax(coords[:,0]) + 0.5)
    ax.set_ylim(bottom = np.amin(coords[:, 1]) - 0.5, top = np.amax(coords[:, 1]) + 0.5)
    for pos, name in zip(coords, nodeNames):
        ax.text(pos[0], pos[1], name, fontsize = 12, horizontalalignment = 'center', verticalalignment = 'center') 
    plt.subplots_adjust(left = 0.05, right = 1.0)
```

## Iterate the Updates to the Positions

First let's set up the tree processor to get the appropriate arrays of indices of of nodes and children inside the `positions` array.
``` python
# Set up the processor and get the arrays of indices in position array of each node.

processor = TreeProcessor()
processor.getLevelInfoLists(tree)
positions = getInitPositions(processor.levelList)
for key in processor.indices:
    print('processor.indices[', key, '] = ')
    print(processor.indices[key])
```
Now let's set up some parameters for adjusting the effects of each part of the gradient descent. We have found these values by trial and error.
``` python
# Parameters for adjusting the effects of each part of gradient descent.

params = {'bounded' : 1e-3, 'level' : 1.0, 'children': 1, 'childRepulsion':50, 'learning_rate':0.001}
```
Now let's actually run the gradient descent and record the result for several times of our iteration. We also record the norms of the changes in the positions of all of the nodes as we iterate.
``` python
# Iterate the updates on the horizontal positions of each node. Graph the result after certain
# iterations.

fig = plt.figure(figsize = figsize)
ypos = getYPos(processor.levelList)
changeNorms = []
normi = []
tracki = [0, 10, 20, 100, 500]
for i in range(5000):

    if i in tracki: 
        drawTree(positions, ypos, processor)
        plt.title('Iterate ' + str(i))
        plt.savefig('2017-11-03-graphs/iterate' + str(i) + '.svg')

    newpos = updatePos(positions, processor.indices, params) 
    newChangeNorm = np.linalg.norm(newpos - positions)
    if i % 50 == 0:
        normi.append(i)
        changeNorms.append(newChangeNorm)
    positions = newpos
```
This produces the graph of iterates we gave at the beginning of this post.

Now we graph the norms of the changes and the final positions of our graphs.
```python
# Graph the sizes of the changes after each iteration.
  
plt.clf()
plt.plot(normi, changeNorms)
plt.title('Norms of Changes in Positions of Each Iteration')
plt.savefig('2017-11-03-graphs/changeNorms.svg')

# Draw the final positions of our graph.

drawTree(positions, ypos, processor)
plt.title('Final Form, Iteration 4999') 
plt.savefig('2017-11-03-graphs/final.svg')
plt.show()
```
The graph of the norms of the changes is:

![Graph of Changes]({{site . url}}/assets/2017-11-03-graphs/changeNorms.svg)

The final graph of the binary tree is the one given at the very beginning of the post.

## [Download The Source Code for this Post]({{site . url}}/assets/2017-11-03-potentialTree.py)
