import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as colors

# Class : Node
# This class is used to hold data in a node of a binary tree. It has an initializer for its data
# which defaults to None. Furthermore, its left and right children initialize as None.

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

        Groups the statistics of the nodes into lists for each level. So we get a list of 2d numpy arrays.

        Parameters
        ----------
        self : self

        Returns
        -------
        A list of 2d numpy arrays where each 2d numpy array holds the statistics of all the nodes in one level.
        That is, each 2d numpy array has shape (number of nodes in level, number of nodes in tree).
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

class RandomVar:

    def __init__(self, values, probs):
        if len(values) > 0:
            self.values = values
        else:
            raise ValueError("values is empty. RandomVar must take on values")

        if len(probs) > 0:
            self.probs = probs
        else:
            raise ValueError("probs is empty. RandomVar must take on probabilies")

    def add(self, randVar2):
        newprobs = {}
        for val, prob in zip(self.values, self.probs):
            for val2, prob2 in zip(randVar2.values, randVar2.probs):
                newval = val + val2
                newprob = prob * prob2 
                if newval in newprobs.keys():
                    newprobs[newval] += newprob
                else:
                    newprobs[newval] = newprob 
        newvals = np.array(list(newprobs.keys()))
        newprobs = np.array(list(newprobs.values()))
        return RandomVar(newvals, newprobs)

    def shift(self, dvals):
        self.values += dvals

class BinomialTable:

    def __init__(self, n):
        self.n = n
        self.__computeTable(n)

    def __computeTable(self, n):

        self.table = np.zeros((n + 1, n + 1))
        self.table[0,0] = 1
        if n == 0:
            return 
        # Now we use the fact that C(n, k) = C(n-1, k) + C(n - 1, k - 1). This fact comes from determing
        # the two cases of whether the last element is in the k-subset.
        for nballs in range(1, n + 1):
          self.table[nballs, 0] = 1
          self.table[nballs, 1:] = self.table[nballs - 1, 1:] + self.table[nballs - 1, :-1] 

    def getTable(self):
        return self.table

    def C(n, k):
        if n < 0 or n > self.n:
            return  0.0

        elif k < 0 or k > self.n:
            return 0.0 

        else:
            return table[n, k]

    def getRandomVar(self, n, p):
        if n > self.n:
            raise ValueError("n is too large for binomial table when creating random binomial variable")
        q = 1.0 - p
        vals = np.arange(n+1) 
        probs = self.table[n, :n+1]
        probs = probs * p**vals
        probs = probs * q**(n - vals) 
        return RandomVar(vals, probs)

class modelTree:

    def __init__(self, nLevels):

        if nLevels < 0:
            raise ValueError("Tree should have non-negative number of levels.")

        self.maxLevel = nLevels - 1
        self.nNodes = 2**nLevels - 1 
        self.leftEdgeProb = 1.0 / 3.0
        self.rightEdgeProb = 2.0 / 3.0
        self.binTable = BinomialTable(nLevels) 
        self.root = Node()
        self.__addRandVars(self.root, 0, 0, 0, 0)
        
    def __finalNodeVar(self, nodeLevel):
        nSubLevels = self.maxLevel - nodeLevel 
        nSubNodes = 2**nSubLevels - 1
        values = np.array([0, nSubNodes, 2 * nSubNodes])
        probs = np.full(values.shape, 1.0 / len(values)) 
        return RandomVar(values, probs)

    def __leftEdgeVar(self, nLeftAbove):
        var = self.binTable.getRandomVar(nLeftAbove, self.leftEdgeProb) 
        return var 

    def __rightEdgeVar(self, nRightAbove):
        var = self.binTable.getRandomVar(nRightAbove, self.rightEdgeProb)
        return var 

    def __convertRandVar(self, randVar):

        probs = np.zeros(self.nNodes)
        probs[randVar.values] = randVar.probs
        return probs
    
    def __addRandVars(self, node, level, shift, nLeftAbove, nRightAbove):

        print(level, shift, nLeftAbove, nRightAbove)
        randVar = RandomVar(np.array([shift]), np.array([1.0]))
        if level < self.maxLevel:
            randVar = randVar.add(self.__finalNodeVar(level))
        if nLeftAbove > 0:
            randVar = randVar.add(self.__leftEdgeVar(nLeftAbove))
        if nRightAbove > 0:
            randVar = randVar.add(self.__rightEdgeVar(nRightAbove))
        print(randVar.values)
        print(randVar.probs)
        node.data = self.__convertRandVar(randVar)
        if level < self.maxLevel:
            nSubLevels = self.maxLevel - level
            newShift = 2**nSubLevels - 1 + shift 
            node.left = Node()
            node.right = Node()
            self.__addRandVars(node.left, level + 1, shift, nLeftAbove + 1, nRightAbove)
            self.__addRandVars(node.right, level + 1, newShift, nLeftAbove, nRightAbove + 1)

    def getDataList(self):
        dataList = []
        self.__getDataList(dataList, [self.root])
        return dataList

    # In order traversal

    def __getDataList(self, dataList, nodeList):

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

################### Start of main execution #############

nLevels = 5
nTraversals = 7000
binomTable = BinomialTable(nLevels)
np.random.seed(20171121)
model = modelTree(nLevels)
modelList = model.getDataList()
statTraversals = SimulationTree(nLevels)

print(binomTable.getTable())
print(binomTable.getRandomVar(3, 0.25).values, binomTable.getRandomVar(3, 0.25).probs)

randVar1 = binomTable.getRandomVar(1, 0.5)
print(randVar1.values, randVar1.probs)
randVar2 = binomTable.getRandomVar(2, 0.5) 
print(randVar2.values, randVar2.probs)
randVar3 = randVar1.add(randVar2)
print(randVar3.values, randVar3.probs)
printI = 0
for i in range(nTraversals):

    statTraversals.randTraversal()

    if printI > nTraversals / 10:
        print('Finished ' + str(i))
        printI = 0
    printI += 1

statList = statTraversals.getDataList()

cNorm = colors.Normalize(vmin = 0.0, vmax = 1.0)
cmap = cm.ScalarMappable(norm = cNorm, cmap = 'jet') 
levelNames = [str(i) for i in np.arange(len(statList)) ]
for levelData, modelData, levelName in zip(statList, modelList, levelNames):

    cScalar = np.linspace(0, 1.0, len(levelData))
    cRGB = cmap.to_rgba(cScalar)
    xdata = np.arange(len(levelData[0]))
    levelData = np.stack(levelData)
    levelData /= nTraversals
    modelData = np.stack(modelData)

    plt.cla()
    for curve, model, color in zip(levelData, modelData, cRGB):
        plt.plot(curve, color = color)
        plt.scatter(xdata, curve, color = color)
        plt.plot(xdata + 0.5, model, color = color, linestyle = '--')
    plt.title("Level " + levelName) 
    ax = plt.gca()
    ax.set_xlabel('Number Processed')
    ax.set_ylabel('Probability')
    plt.show()
