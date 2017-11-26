import numpy as np
import matplotlib.pyplot as plt

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
            print( "ERROR: values is empty. RandomVar must take on values")

        if len(probs) > 0:
            self.probs = probs
        else:
            print( "ERROR: probs is empty. RandomVar must take on probabilies")

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
            print( "ERROR: n not in bounds for binomial table")
            return 

        elif k < 0 or k > self.n:
            print( "ERROR: k not in bounds for binomial table")
            return 

        else:
            return table[n, k]

    def getRandomVar(self, n, p):
        if n > self.n:
            print("ERROR: n is too large for binomial table when creating random binomial variable")
        q = 1 - p
        vals = np.arange(n+1) 
        probs = self.table[n, :n+1]
        probs *= p**vals
        probs *= q**(n - vals) 
        return RandomVar(vals, probs)

################### Start of main execution #############

nLevels = 4
nTraversals = 1000
binomTable = BinomialTable(nLevels)
np.random.seed(20171121)
statTraversals = Tree(nLevels)

print(binomTable.getTable())
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

for i in range(len(statList)):
    statList[i] = np.stack(statList[i], axis = -1)
    statList[i] /= nTraversals

print(statList[0].shape)

for levelData in statList:
    plt.plot(levelData)
    xdata = np.arange(len(levelData[:, 0])).reshape(-1,1)
    xdata = np.full(levelData.shape, xdata) 
    plt.scatter(xdata, levelData)
    plt.show()

