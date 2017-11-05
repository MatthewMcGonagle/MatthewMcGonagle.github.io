import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

class Node:

    def __init__(self, val, left = None, right = None):
        self.left = left 
        self.right = right 
        self.val = val

class Tree:
    
    def __init__(self, root = None):
        self.root = root 
        self.levelList = None
        self.indices = None 

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

    def printlevel(self, nodelist):
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
            level = self.printlevel(level)

    def nodeList(self):
        if self.root == None:
            return None
        myLevels = self.levelList()
        nodes = [node for level in myLevels for node in level] 
        return nodes 

    # Create a list where each entry is a list of the nodes occuring in each level. So self.levelList is
    # a list of lists of nodes.

    def createLevelList(self):

        if self.root == None:
            return []

        self.levelList = []
        level = [self.root]
        while level != []:
            self.levelList.append(level)
            nextlevel = []
            for node in level:
                if node.left != None:
                    nextlevel.append(node.left)
                if node.right != None:
                    nextlevel.append(node.right)
            level = nextlevel

    def setValToIndexOfFlattenedLevels(self):

        if self.levelList == None:
            raise ValueError('levelList is None when expected to be a list of lists')

        i = 0
        for level in self.levelList:
            for node in level:
                node.val = i
                i += 1
        self.indices = {}

    def __getLeftIndex(self, node):
        if node.left == None:
            return -1
        else:
            return node.left.val 
    
    def __getRightIndex(self, node):
        if node.right == None:
            return -1
        else:
            return node.right.val

    def createIndexLists(self):
        if self.indices == None:
            raise ValueError('indices is None when expecting dictionary. Make sure to call setValToIndexOfFlattenedLevels')
    
        self.indices['level'] = [[node.val for node in level] for level in self.levelList]        
        self.indices['left'] = [np.array([self.__getLeftIndex(node) for node in level]) for level in self.levelList]
        self.indices['right'] = [np.array([self.__getRightIndex(node) for node in level]) for level in self.levelList]
       
        
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

np.random.seed(20171102)
nums = np.random.randint(0, 100, size = 100)
print(nums)
tree = Tree()
for num in nums:
    tree.insert(num)  
tree.print()

tree.createLevelList()
positions = getInitPositions(tree.levelList)
tree.setValToIndexOfFlattenedLevels()
tree.createIndexLists()
tree.print()

params = {'bounded' : 1e-3, 'level' : 1, 'children': 1, 'childRepulsion':50, 'learning_rate':0.001}

changes = []
for i in range(5000):
    newpos = updatePos(positions, tree.indices, params) 
    newchange = np.linalg.norm(newpos - positions)
    changes.append(newchange)
    positions = newpos
  
plt.plot(changes[50:])
plt.show()

ypos = getYPos(tree.levelList)

nodelist = [node for level in tree.levelList for node in level]
for node, pos, ypos in zip(nodelist, positions, ypos):
    node.val = (pos, ypos) 

def getTreeLines(node, lines):
    if node.left != None:
        lines.append([node.val, node.left.val])
        getTreeLines(node.left, lines)
    if node.right != None:
        lines.append([node.val, node.right.val])
        getTreeLines(node.right, lines)

lines = []
getTreeLines(tree.root, lines)
lines = np.array(lines).T
print('lines.shape = ', lines.shape)
plt.plot(lines[0], lines[1], color = 'red')
plt.scatter(lines[0], lines[1])

plt.show()
