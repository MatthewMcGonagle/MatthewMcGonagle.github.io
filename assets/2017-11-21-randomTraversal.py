import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as colors

class Node:

    def __init__(self, data = None):
        self.data = data
        self.left = None 
        self.right = None 

class Tree:

    def __init__(self, nLevels):
        self.nNodes = 2**(nLevels + 1) - 1
        self.root = Node(np.zeros(self.nNodes))
        self.addChildren(nLevels, self.root)
        self.rollValue = {'preorder' : 0, 'inorder' : 1, 'postorder' : 2}

    def addChildren(self, level, node):
        node.left = Node(np.zeros(self.nNodes))
        node.right = Node(np.zeros(self.nNodes))
        if level > 1:
            self.addChildren(level - 1, node.left) 
            self.addChildren(level - 1, node.right)

    def __randTraversal(self, n, node):

        # For empty leaves, don't do anything.

        if node == None:
            return n

        # Randomly select which traversal patter to use at this node.

        diceroll = np.random.randint(0, 3)

        if diceroll == self.rollValue['preorder']:
            node.data[n] += 1
            newN = self.__randTraversal(n+1, node.left)
            newN = self.__randTraversal(newN, node.right) 
            return newN

        elif diceroll == self.rollValue['inorder']:
            newN = self.__randTraversal(n, node.left)
            node.data[newN] += 1 
            newN = self.__randTraversal(newN + 1, node.right)
            return newN

        else: # post order case
            newN = self.__randTraversal(n, node.left)
            newN = self.__randTraversal(newN, node.right)
            node.data[newN] += 1
            return newN + 1

    def randTraversal(self):
        checkN = self.__randTraversal(0, self.root)

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

nLevels = 4
nTraversals = 7000
binomTable = BinomialTable(nLevels)
np.random.seed(20171121)
model = modelTree(nLevels+1)
modelList = model.getDataList()
statTraversals = Tree(nLevels)

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
cmap = cm.ScalarMappable(norm = cNorm, cmap = 'winter') 
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
