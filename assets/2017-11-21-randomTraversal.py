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

nLevels = 4
nTraversals = 1000
np.random.seed(20171121)
statTraversals = Tree(nLevels)

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

