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

    def levelList(self):
        if self.root == None:
            return []

        mylist = []
        level = [self.root]
        while level != []:
            mylist.append(level)
            nextlevel = []
            for node in level:
                if node.left != None:
                    nextlevel.append(node.left)
                if node.right != None:
                    nextlevel.append(node.right)
            level = nextlevel
        
        return mylist

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
    update = np.zeros(parentPos.shape)
    
    # When left child is to the right, we need to move the node to the right. Do so at an exponential rate.
    update[lChildOnRight] = -1 + np.exp(lChildPos[lChildOnRight] - parentPos[lChildOnRight]) 
    
    # When left child is on the left, we move the node left towards the child but also to the right away from
    # the child at different rates.
    
    update[lChildOnLeft] = lChildPos[lChildOnLeft] - parentPos[lChildOnLeft]
    update[lChildOnLeft] += (1 / childRepulsion + parentPos[lChildOnLeft] - lChildPos[lChildOnLeft] )**-1

    return update

def updatePos(positions, indices, params):
    baseIndex = 0
    allupdates = np.array([])
    epsilon = 1e-2
    for leveli, lefti, righti in zip(indices['level'], indices['left'], indices['right']):

        levelPos = positions[leveli]

        # Update to keep the graph from flying off to infinity.

        update = -params['bounded'] * getUpdateForBounded(levelPos) 

        # Updates for spacing out nodes.

        update += params['level'] * getUpdateFromSameLevel(levelPos)

        # Update coming from left children

        hasLeft = lefti > -1
        parentPos = levelPos[hasLeft]
        lChildPos = positions[lefti[hasLeft]]

        newUpdate = getUpdateFromLeftChildren(parentPos, lChildPos, params['childRepulsion'])
        update[hasLeft] += params['children'] * newUpdate 

        hasRight = righti > -1
        wRChild = levelPos[hasRight]
        rightPos = positions[righti[hasRight]]
        rChildOnLeft = rightPos < wRChild
        newUpdate = np.zeros(levelPos.shape)
        rChildUpdate = newUpdate[hasRight]
        rChildUpdate[rChildOnLeft] =  1 - np.exp(wRChild[rChildOnLeft] - rightPos[rChildOnLeft]) 
        rChildUpdate[~rChildOnLeft] = rightPos[~rChildOnLeft] - wRChild[~rChildOnLeft] 
        rChildUpdate[~rChildOnLeft] += -(epsilon + rightPos[~rChildOnLeft] - wRChild[~rChildOnLeft] )**-1 
        newUpdate[hasRight] = rChildUpdate

        update += params['children'] * newUpdate
        
        # Add updates for this level to updates for entire tree.

        allupdates = np.concatenate([allupdates, update])
    
    return positions + params['learning_rate'] * allupdates

np.random.seed(20171102)
nums = np.random.randint(0, 100, size = 70)
print(nums)
tree = Tree()
for num in nums:
    tree.insert(num)  
tree.print()

levelList = tree.levelList()
positions = getInitPositions(levelList)
print(levelList)

i = 0
for level in levelList:
    for node in level:
        node.val = i
        i += 1

tree.print()

indices = [[node.val for node in level] for level in levelList]        
indices = {'level' : indices}
print(indices['level'])

def getLeftIndex(node):
    if node.left == None:
        return -1
    else:
        return node.left.val 

def getRightIndex(node):
    if node.right == None:
        return -1
    else:
        return node.right.val

indices['left'] = [np.array([getLeftIndex(node) for node in level]) for level in levelList]
indices['right'] = [np.array([getRightIndex(node) for node in level]) for level in levelList]

params = {'bounded' : 1e-1, 'level' : 1, 'children': 1, 'childRepulsion':1e2, 'learning_rate':0.001}

changes = []
for i in range(5000):
    newpos = updatePos(positions, indices, params) 
    newchange = np.linalg.norm(newpos - positions)
    changes.append(newchange)
    positions = newpos
  
print(positions)
plt.plot(changes[50:])
plt.show()

ypos = getYPos(levelList)

nodelist = [node for level in levelList for node in level]
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
print(lines)
lines = np.array(lines).T
print('lines.shape = ', lines.shape)
plt.plot(lines[0], lines[1], color = 'red')
plt.scatter(lines[0], lines[1])

plt.show()

