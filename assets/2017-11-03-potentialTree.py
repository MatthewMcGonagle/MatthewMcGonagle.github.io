import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

figsize = (15, 5)

class Node:

    def __init__(self, val, left = None, right = None):
        self.left = left 
        self.right = right 
        self.val = val

class Tree:
    
    def __init__(self, root = None):
        self.root = root 

    def insertsubtree(self, val, root):
        if root.val == val:
            return
        elif val < root.val:
            if root.left == None:
                root.left = Node(val)
            else:
                self.insertsubtree(val, root.left)
        else:
            if root.right == None:
                root.right = Node(val)
            else:
                self.insertsubtree(val, root.right)
    
    def insert(self, val):
        if self.root == None:
            self.root = Node(val)
        else:
            self.insertsubtree(val, self.root)

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
            
    def print(self):

        if self.root == None:
            print("None")
            return
        level = [self.root]
        while level != [] :
            level = self.__printlevel(level)

class TreeProcessor:

    def __init__(self):
        self.levelList = None
        self.indices = None
        self.NOCHILD = -1
        self.childI = None 
        self.edges = None


    def __updateLeftChild(self, node, nextLevel, newLeftInd):
        
         if node.left != None:
             nextLevel.append(node.left)
             newLeftInd.append(self.childI)
             self.childI += 1
         else:
             newLeftInd.append(self.NOCHILD)

    def __updateRightChild(self, node, nextLevel, newRightInd):
        
         if node.right != None:
             nextLevel.append(node.right)
             newRightInd.append(self.childI)
             self.childI += 1
         else:
             newRightInd.append(self.NOCHILD)

    def __offsetChild(self, childIndex, numPrevious):

        if childIndex != self.NOCHILD:
            return childIndex + numPrevious
        else:
            return childIndex
 
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

            for node in level:

                newLevelInd.append(nodeI)
                nodeI += 1

                self.__updateLeftChild(node, nextLevel, newLeftInd)
                self.__updateRightChild(node, nextLevel, newRightInd)

            newLeftInd = [self.__offsetChild(i, nodeI) for i in newLeftInd]
            newRightInd = [self.__offsetChild(i, nodeI) for i in newRightInd]

            self.indices['level'].append(np.array(newLevelInd)) 
            self.indices['left'].append(np.array(newLeftInd))
            self.indices['right'].append(np.array(newRightInd))                    
            self.levelList.append(level)
            level = nextLevel

    def __addEdge(self, parenti, childi, positions): 

            parentPos = positions[parenti]
            if childi != self.NOCHILD:
                newPos = positions[childi]
                self.edges.append([parentPos, newPos])

    def getEdges(self, positions):
        self.edges = []
        for leveli, leftiS, rightiS in zip(self.indices['level'], self.indices['left'], self.indices['right']):
            for parenti, lefti, righti in zip(leveli, leftiS, rightiS):
                self.__addEdge(parenti, lefti, positions)
                self.__addEdge(parenti, righti, positions)
        self.edges = np.array(self.edges)

    def getNodeText(self):

        textList = []

        for level in self.levelList:
            for node in level:
                textList.append(str(node.val))

        return textList
        
def getInitPositions(levelList):
    positions = np.array([])
    for level in levelList:
        levelpos = np.arange(len(level), dtype = 'float32')
        positions = np.concatenate([positions, levelpos])
    return positions

def getYPos(levelList):
    ypos = np.array([])
    pos = 0
    for level in levelList: 
        newpos = np.full(len(level), pos)
        ypos = np.concatenate([ypos, newpos])
        pos -= 1
    return ypos
 
# Update to keep the graph from flying off to inifinity.

def getUpdateForBounded(levelPos):

    return -levelPos
   
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

np.random.seed(20171102)
nums = np.random.randint(0, 100, size = 100)
print(nums)
tree = Tree()
for num in nums:
    tree.insert(num)  
tree.print()

processor = TreeProcessor()
processor.getLevelInfoLists(tree)
positions = getInitPositions(processor.levelList)
for key in processor.indices:
    print('processor.indices[', key, '] = ')
    print(processor.indices[key])
tree.print()

params = {'bounded' : 1e-3, 'level' : 1.0, 'children': 1, 'childRepulsion':50, 'learning_rate':0.001}

fig = plt.figure(figsize = figsize)
ypos = getYPos(processor.levelList)
changeNorms = []
tracki = [0, 10, 20, 100, 500, 1000, 2000]
for i in range(3000):

    if i in tracki: 
        drawTree(positions, ypos, processor)
        plt.savefig('2017-11-03-graphs/iterate' + str(i) + '.svg')

    newpos = updatePos(positions, processor.indices, params) 
    newChangeNorm = np.linalg.norm(newpos - positions)
    changeNorms.append(newChangeNorm)
    positions = newpos
  
plt.clf()
plt.plot(changeNorms[50:])
plt.savefig('2017-11-03-graphs/changeNorms.svg')

drawTree(positions, ypos, processor)
plt.savefig('2017-11-03-graphs/final.svg')
plt.show()
