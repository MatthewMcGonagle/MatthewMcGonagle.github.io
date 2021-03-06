'''
2018-06-09-TSPArtModifiedAnnealing.py
Author: Matthew McGonagle

Uses dithering, a greedy guess, simulated annealing based on length scale, and 
simulated annealing based on k-Neighbors. 

To be more specific,
1) We use dithering to represent our image as black and white pixels (no gray). The black
    pixels become our vertices.

2) Our greedy guess consists of joining together collections of curves in a way that greedily
    attempts to minimize length. As we join the curves, their number decreases and their length increases.

3) For simulated annealing based on a length scale, at the start of each annealing job (batch of annealing steps)
    we find a pool of vertices such that the vertex is attached to segment of a curve at least as large as
    some given length scale. During the anneling process of selecting a random pair of vertices, we only
    do some from this pool. The idea is we start by trying to deal with the segments that are large.

    During each step, in addition to cooling the temperature, we geometrically 
    decay (or cool) the size scale.

    Note, that the pool is only updated between each job, not between each annealing step.  

4) For simulated annealing using k-neighbors, when we randomly select a pair, we first randomly select
    a vertex and then uniformly select from its k-neighbors. During each annealing step, in addition to
    cooling the temperature, we decay the number of k-neighbors. 
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
import io # Used for itermediary when saving pyplot figures as PNG files so that we can use PIL
          # to reduce the color palette; this reduces file size.

# Seed numpy for predictability of results.

np.random.seed(20180425)

class AnnealerTSPSizeScale:
    '''
    An iterator for performing annealing based on a size scale. The annealing is done on a pool of vertices
    that are on an edge of the cycle that is at least as long as the current size scale; so one should think
    of the annealing as starting with those edges of the cycle that are large.

    Note the vertex pool is NOT updated each step of the iteration; this would be too costly. Instead the 
    vertex poolis updated upon a warm restart. 
    
    Members
    -------
    nSteps : Int
        The total number of steps to run.
    
    stepsProcessed : Int
        The number of steps run so far.
    
    vertices : Numpy Array of Floats of Shape (nVertices, 2)
        Holds the coordinates of the vertices.
    
    nVertices : Int
        The number of vertices in travelling salesman problem.
    
    temperature : Float
        The temperature to use in the simulated annealing algorithm. It is part of
        determining the probability of moving to a state with higher length.
    
    tempCool : Float
        The multiplicative factor to use to cool the temperature at each step. For cooling,
        it should be between 0 and 1.
    
    sizeScale : Float
        The current size scale. Used when setting up the pool of vertices between jobs.
        At the time of construction, the pool consists of vertices touching a segment
        of size at least as large as sizeScale.
    
    sizeCool : Float
        The multiplicative factor to use to lower the sizeScale at each step. 
    
    vOrder : Numpy array of Int of shape (nVertices + 1)
        Holds the order of the vertices in the path. So vOrder[2] holds the index 
        (of self.vertices) for the third vertex in the path.
    
    poolV : Numpy array of Int of shape (nPool)
        The indices of the elements of vOrder that appear in the pool.
    
    nPool : Int
        The number of vertices in the pool.
    
    '''

    def __init__(self, nSteps, vertices, temperature, tempCool, sizeScale, sizeCool):
        '''
        Set up the total number of steps that the iterator will take as well as the cooling. 

        Parameters
        ----------
        nSteps : Int
            The total number of steps the iterator will take.
        
        vertices : Numpy array of shape (nVertices, 2)
            The xy-coordinates of the vertices.

        temperature : Float
            The initial temperature of the anealing process.

        tempCool : Float
            The cooling parameter to apply to the temperature at each step. The cooling is applied via multiplication.

        sizeScale : Float
            The size scale that the annealer will start at. Starts by looking at vertices that are connected to an edge 
            of size atleast as large as the size scale.

        sizeCool : Float
            The rate (or decay) of the size scale. The size cooling is applied via multiplication by sizeCool.

        '''

        # Set up step counting for iteration.
        self.nSteps = nSteps
        self.stepsProcessed = 0

        # Set up vertices data.
        self.vertices = vertices
        self.nVertices = len(vertices)

        # Set up temperature and size scale data.
        self.temperature = temperature
        self.tempCool = tempCool
        self.sizeScale = sizeScale
        self.sizeCool = sizeCool

        # Set up array that holds the order that the vertices currently appear in the cycle. 
        self.vOrder = np.arange(self.nVertices)

        # Initialize the pool of vertices for the size scale to be None. Then set up the scale pool 
        # based on the initial scale size.
        self.poolV = None
        self.nPool = None
        self.setScalePool()

    def __iter__(self):
        '''
        Iterator function for class.

        Returns
        -------
        self
        '''

        return self

    def __next__(self):
        '''
        Do the next iteration step of the annealing. Note that we do NOT update the vertex pool that
        the annealing is performed upon.
        
        Returns
        -------
        Float
            The energy difference between the new cycle and the current cycle.
        '''

        if self.stepsProcessed >= self.nSteps:
    
            raise StopIteration

        # First do cooling and update the step count.

        self.temperature *= self.tempCool
        self.stepsProcessed += 1
        self.sizeScale *= self.sizeCool

        # Find the energy difference resulting from switching the order of the cycle between two random
        # vertices taken from the size scale pool of vertices. 

        begin, end = self.getRandomPair()

        energyDiff = self.getEnergyDifference(begin, end)

        # Deal with whether to accept proposal switch based on the energy difference. 

        if energyDiff < 0:
            proposalAccepted = True

        else:
            proposalAccepted = self.runProposalTrial(energyDiff)

        if proposalAccepted:

            self.reverseOrder(begin, end)

        return energyDiff

    def setScalePool(self):
        '''
        Reset the pool of vertices for annealing based on the current cycle edge sizes and
        the current size scale.

        If the scale pool has only one vertex then a ValueError exception is raised.
        '''

        # First find the interior differences, i.e. the vector differences between vertices in the
        # cycle excluding the difference between the final and last vertex.
        interiorDiff = self.vertices[self.vOrder[1:]] - self.vertices[self.vOrder[:-1]] 

        # Next find the vector difference between the first vertex and the last vertex. 
        joinDiff = self.vertices[self.vOrder[-1]] - self.vertices[self.vOrder[0]]

        # Find the forward differences and backward differences for each vertex.
        forwardDiff = np.concatenate([interiorDiff, [joinDiff]], axis = 0) 
        backwardDiff = np.concatenate([[joinDiff], interiorDiff], axis = 0)

        # Find the forward lengths and backward lengths.
        forwardDist = np.linalg.norm(forwardDiff, axis = -1)
        backwardDist = np.linalg.norm(backwardDiff, axis = -1)

        # Find which vertices are in the pool based on whether the forward
        # length or backward length is large enough.

        verticesInPool = (forwardDist > self.sizeScale) | (backwardDist > self.sizeScale)

        self.poolV = np.arange(self.nVertices)[verticesInPool]
        self.nPool = len(self.poolV)

        if self.nPool < 2:

            raise ValueError('Size scale pool has less than two vertices.')

    def getRandomPair(self):
        ''' 
        Get a random pair of vertices from the pool of vertices in the current size pool. This
        should only be called if there are atleast two vertices in the size scale pool.

        Returns
        -------
        (Int, Int)
            A pair of indices for which vertices to use from the pool. The indices are for
            the array self.poolV, not the original vertices array. Also the first index
            is always less than the second.
        '''

        sameNum = True

        while sameNum:

           begin = np.random.randint(self.nPool)
           end = np.random.randint(self.nPool) 

           begin = self.poolV[begin]
           end = self.poolV[end]

           sameNum = (begin == end) 

        if begin < end: 

            return begin, end

        else:

            return end, begin


    def reverseOrder(self, i, j):
        '''
        Reverse the order of vertices between the ith vertex in the ith vertex in the cycle and the jth
        vertex in the cycle. Not that this reverses elements contained in the array self.vOrder and
        not the original array of vertices.

        Also note that we don't have to do anything to the pool of vertices. Some elements of the pool
        will point to different vertices, but this is really just a rearrangement of the vertices 
        contained in the pool. This will have no effect on random selection which is the only
        purpose of the pool. 

        Parameters
        ----------
        i : Int
            The index of the element in self.vOrder for the beginning of the reversal segment; i.e. the ith vertex
            in the cycle. The index i should be less than the index j.

        j : Int
            The index of the elemet in self.vOrder for the end of the reversal segment; i.e. the jth vertex in the
            cycle. The index j should be greater than the index i. 
        '''

        self.vOrder[i : j + 1] = np.flip(self.vOrder[i : j + 1], axis = 0)

    def getEnergyDifference(self, i, j):
        '''
        Find the energy (i.e. length) difference resulting from reversing the path between
        the ith vertex in the cycle with the jth vertex in the cycle. Both of the indices i and j 
        should be greater than 0 (i.e. excluding the first vertex in the cycle) and less than
        self.nVertices - 1 (i.e. excluding the last vertex in the cycle).

        Parameters
        ----------
        i : Int
            Index for the ith vertex in the cycle. Should be greater than 0 and less than self.nVertices - 1.

        j : Int
            Index for the jth vertex in the cycle. Should be greater than 0 and less than self.nVertices - 1.

        Returns
        -------
        Float
            The energy different (i.e. length difference) resulting from a proposed reversal.
        '''

        begin = self.vertices[self.vOrder[i]]
        if i > 0:
            beginParent = self.vertices[self.vOrder[i - 1]]
        else:
            beginParent = self.vertices[self.vOrder[self.nVertices - 1]]

        end = self.vertices[self.vOrder[j]]
        if j < self.nVertices - 1:
            endChild = self.vertices[self.vOrder[j + 1]]
        else:
            endChild = self.vertices[self.vOrder[0]]
       
        oldEnergy = np.linalg.norm(begin - beginParent) + np.linalg.norm(end - endChild)
        newEnergy = np.linalg.norm(begin - endChild) + np.linalg.norm(end - beginParent)         

        energyDiff = newEnergy - oldEnergy

        return energyDiff

    def runProposalTrial(self, energyDiff):
        '''
        Run the bernoulli trial for determining whether to accept a proposed reversal in the case
        that the reversal will result in an increase in energy.

        Parameters
        ----------
        energyDiff : Float
            The energy difference (i.e. lengthDifference) for the proposal. This should be greater than 0.

        Returns
        -------
        Bool
            Whether to accept the proposal based on the random bernoulli trial.
        '''

        prob = np.exp(-energyDiff / self.temperature)

        trial = np.random.uniform()

        return (trial < prob)

    def getCycle(self):
        '''
        Get the vertices in the order they appear in the cycle. 

        Returns
        -------
        Numpy array of shape (nVertices, 2)
            The coordinates of the vertices for the order they appear in the cycle. 
        '''
        
        cycle = self.vertices[self.vOrder]
        begin = self.vertices[self.vOrder[0]]
        cycle = np.concatenate([cycle, [begin]], axis = 0)
        return cycle 

    def getOrder(self):
        '''
        Get the order that the vertices appear in the cycle.

        Returns 
        -------
        Numpy array of Int of shape (nVertices)
            Indices of the vertices as they appear in the cycle; e.g. array[0] is the index of the first vertex
            in the original vertex array.
        '''

        return self.vOrder

    def getEnergy(self):
        '''
        Get the length of the current cycle.

        Returns
        -------
        Float
            The length of the current cycle.
        '''

        # First find the length for the edge connecting the first vertex to the last.

        energy = self.getLength(0, self.nVertices - 1) 

        # Now add in the lengths of the other edges.

        for i in np.arange(0, self.nVertices - 1, dtype = 'int'):  

            energy += self.getLength(i, i+1) 

        return energy

    def getLength(self, pathi, pathj):
        ''' 
        Get the length of the line segment between the vertex at position pathi in the cycle and 
        the position pathj in the cycle.

        Parameters
        ----------
        pathi : Int
            The first vertex is at position pathi in the path, i.e. the index of an element in vOrder.

        pathj : Int
            The second vertex is at position in the path, i.e. the index of an element in vOrder.

        Returns
        -------
        Float
            The length of the line segment between the two vertices.
        '''
        point1 = self.vertices[self.vOrder[pathi]]
        point2 = self.vertices[self.vOrder[pathj]]
        length = np.linalg.norm(point2 - point1)
        return length

    def doWarmRestart(self):
        '''
        Do a warm restart of the iterator. This also updates the size scale vertex pool.
        '''

        self.stepsProcessed = 0
        self.setScalePool()

    def getInfoString(self):
        '''
        Get a string for information of the current state of the iterator.

        Returns
        -------
        String
            Contains information for the energy, number of vertices in the size scale pool, and the 
            temperature.
        '''

        energy = self.getEnergy()
        info = 'Energy = ' + str(energy)
        info += '\tnPool = ' + str(self.nPool)
        info += '\tTemperature = ' + str(self.temperature)

        return info

##########################################
#### NeighborsAnnealer
##########################################

class NeighborsAnnealer:
    '''
    Modified simulated annealer that randomly selects a vertex and then randomly selects another vertex from the
    k-nearest neighbors of the first vertex. The point of this annealer is to do annealing when the current
    cycle is at a stage where the changes needed to be made are by switching vertices that are close ot each other.    

    Members
    -------
    nSteps : Int
        The total number of steps to make.
    
    nProcessed : Int
        The total number of steps made so far.

    vertices : Numpy array of Floats of Shape (nVertices, 2)
        The vertices in their original order.

    nVertices : Int
        The number of vertices.

    temperature : Float
        The temperature used for annealing.

    cooling : Float
        The cooling factor for the temperature. At each step, it is applied to the temperature via multiplication.    
        That is, we use geometric cooling.
   
    kNbrs : Float 
        The number of neighbors to randomly select from. This is converted to an int when doing selection. It is
        a float because we do geometric cooling on the number of neighbors as we run each step. 

    nbrsCooling : Float
        The factor to use to cool (decay) the number of neigbors. At each step, it is applied to kNbrs via 
        multiplication.        

    nearestNbrs : class NearestNeighbors
        We use sci-kit-learn NearestNeighbors to find the nearest neighbors of vertices. This is trained on the
        original order of the vertices, so we need to deal with converting between the original order
        of the vertices to the current order of the vertices in the array.

    origToCurrent : Numpy Array of Int of Shape (nVertices)
        Array for converting from original indices to current indices in cycle. That is
        origToCurrent[i] is the current index of what was originally the ith vertex. This
        is needed for dealing with results of nearest neighbors search. 

    currentToOrig: Numpy Array of Int of Shape (nVertices)
        Array for converting from the current index of a vertex to the original index of the vertex.
        That is currentToOrig[i] is the original index of what is now index i in the cycle. This
        is needed to update origToCurrent when doing a reversal. 
    '''

    def __init__(self, nSteps, vertices, temperature, cooling, kNbrs, nbrsCooling):
        '''
        Initializer. Make sure to train the nearest neighbors on the original order of the vertices.

        Parameters
        ----------

        nSteps : Int
            The total number of iterations to make.

        vertices : Numpy array of Floats of shape (nVertices, 2)
            The vertices in their initial order.

        temperature : Float
            The initial temperature to use for the annealing.

        cooling : Float
            The cooling factor to apply to the temperature at each step; it is applied
            via multiplication. That is, we have geometric cooling. 

        kNbrs : Float
            The initial value for kNbrs.

        nbrsCooling : Float
            The cooling factor (decay factor) for the number of neighbors; at each step it
            is applied to kNbrs via multiplication. Note that kNbrs is a float as well.
        '''

        self.nSteps = nSteps
        self.nProcessed = 0

        self.vertices = vertices
        self.nVertices = len(vertices)

        self.temperature = temperature
        self.cooling = cooling
        self.kNbrs = kNbrs
        self.nbrsCooling = nbrsCooling

        # Make sure to train the Nearest Neighbors class.
        self.nearestNbrs = NearestNeighbors()
        self.nearestNbrs.fit(self.vertices.copy())
        self.origToCurrent = np.arange(self.nVertices)
        self.currentToOrig = np.arange(self.nVertices)

    def __iter__(self):
        '''
        Get an iterator reference

        Returns
        -------
        self
        '''
        return self

    def __next__(self):
        '''
        Do the next step of annealing. For the annealing we randomly pick a vertex and then randomly pick from the kNeighbors
        of that vertex. Note that we convert kNbrs to an Int to determing the number of neighbors.
        '''

        if self.nProcessed >= self.nSteps:

            raise StopIteration

        # Do annealing parameter updates.

        self.nProcessed += 1
        self.temperature *= self.cooling
        self.kNbrs *= self.nbrsCooling 

        # Get a random pair of vertices and find the energy difference if we were
        # to reverse the part of the cycle between them.
        begin, end = self.__getRandomPair()  
        energyDiff = self.__getEnergyDifference(begin, end)

        # Determine whether to accept the proposed reversal based on the energy difference.

        proposalAccepted = False

        if energyDiff < 0:
            proposalAccepted = True

        else:

            proposalAccepted = self.__runProposalTrial(energyDiff) 

        if proposalAccepted:

            self.__reverse(begin, end)
        

    def __runProposalTrial(self, energyDiff):
        '''
        Run the bernoulli trial to determine whether we accept a proposed reversal in the case
        that this reversal results in an increase in the length of the cycle.

        Parameters
        ----------
        energy Diff : Float
            The difference in cycle length if we accept the proposed reversal. This should be greater than 0.

        Returns
        -------
        Bool
            Whether the proposed reversal was accepted.
        '''

        trial = np.random.uniform()
        probAccept = np.exp(-energyDiff / self.temperature)
        if trial < probAccept:
            return True
        else:
            return False
        
    def __getEnergyDifference(self, begin, end):
        '''
        Find the energy difference (length difference) if we reverse the order of the vertices between begin
        and end (inclusive).

        Parameters
        ----------
        begin : Int
            The index of the starting vertex. Should be less than end.

        end : Int
            The index of the ending vertex. Should be greater than begin.

        Returns
        -------
        Float
            The energy difference (length difference).
        '''
 
        # If the begin is vertex 0, then we need to handle wrapping around to the end. 

        if begin > 0: 
            origDiffbegin = self.vertices[begin] - self.vertices[begin - 1] 
            newDiffbegin = self.vertices[end] - self.vertices[begin - 1]
        else:
            origDiffbegin = self.vertices[begin] - self.vertices[-1]
            newDiffbegin = self.vertices[end] - self.vertices[-1]

        # If end is the last vertex, then we need to handle wrapping around to the beginning.

        if end < self.nVertices - 1:
            origDiffend = self.vertices[end] - self.vertices[end + 1]
            newDiffend = self.vertices[begin] - self.vertices[end + 1]
        else:
            origDiffend = self.vertices[end] - self.vertices[0]
            newDiffend = self.vertices[begin] - self.vertices[0]

        origEnergy = np.linalg.norm(origDiffbegin) + np.linalg.norm(origDiffend)
        newEnergy = np.linalg.norm(newDiffbegin) + np.linalg.norm(newDiffend)
        energyDiff = newEnergy - origEnergy
        
        return energyDiff

    def __getRandomPair(self):
        '''
        Get a random pair of indices for vertices. The first index is chosen uniformly. The second
        index is chosen from the k-Nearest Neighbors of the first vertex. Note that we convert
        kNbrs to an Int to get the number of neighbors.

        Returns
        -------
        (Int, Int)
            The indices of the two random vertices. The first index will be less than the second.
        '''
        sameNum = True
        trivial = True
        kNbrs = int(self.kNbrs)

        # We loop until we have a choice that is two different indices and
        # doesn't include a trivial choice of the first and last indices.

        while sameNum or trivial:
        
            begin = np.random.randint(self.nVertices)
            beginV = self.vertices[begin].reshape(1,-1)

            # Find the neighbors of begin.
            _, nbrsI = self.nearestNbrs.kneighbors(beginV, n_neighbors = kNbrs)
            nbrsI = nbrsI.reshape(-1)

            # Randomly choose from the neighbors.

            end = np.random.randint(len(nbrsI))
            end = nbrsI[end]
            end = self.origToCurrent[end]

            # Check that our pair is acceptable.

            sameNum = (begin == end)
            trivial = (begin == 0) & (end == self.nVertices - 1)
            
        if begin < end:

            return begin, end

        else:

            return end, begin  

    def __reverse(self, begin, end):
        '''
        Perform a reversal of the segment of the cycle between begin and end (inclusive). Also handles
        the effects on the conversions between the original and new indices (needed for nearest neighbor search).

        Parameters
        ----------
        begin : Int
            The index of the beginning of the segment.

        end : Int
            The index of the end of the segment.

        '''

        self.vertices[begin : end + 1] = np.flip(self.vertices[begin : end + 1], axis = 0)
        self.currentToOrig[begin : end + 1] = np.flip(self.currentToOrig[begin : end + 1], axis = 0)

        # Updating the conversion from original to current indices requires more than a flip.

        origIndices = self.currentToOrig[begin : end + 1]
        self.origToCurrent[origIndices] = np.arange(begin, end+1)

    def doWarmRestart(self):
        '''
        Do a warm restart. We only need to update the number of steps processed.
        '''

        self.nProcessed = 0 

    def getCycle(self):
        '''
        Return the cycle. Note that the cycle is closed so that the first vertex will match the last vertex.

        Returns
        -------
        Number Array of Floats of Shape (nVertices + 1, 2)
            The xy-coordinates of the vertices as they appear in the order of the cycle. Note that first
            vertex appears again at the end of the array to make the cycle closed.
        '''

        cycle = self.vertices.copy()
        cycle = np.concatenate([cycle, [self.vertices[0]] ], axis = 0)

        return cycle 

    def getEnergy(self):
        '''
        Get the energy (i.e. length) of the current cycle)
        
        Returns
        -------
        Float
            The length of the current cycle.
        '''

        differences = self.vertices[1:] - self.vertices[:-1]
        energy = np.linalg.norm(differences, axis = -1).sum()
        energy += np.linalg.norm(self.vertices[0] - self.vertices[-1])

        return energy

    def getInfoString(self):
        '''
        Get information on the current parameters of the annealing process as a string.

        Returns
        -------
        String
            Contains information on the energy, kNbrs, and the temperature. 
        '''

        info = 'Energy = ' + str(self.getEnergy())
        info += '\tkNbrs = ' + str(self.kNbrs)
        info += '\tTemperature = ' + str(self.temperature)

        return info

############################
### Greedy Guesser
############################

class GreedyGuesser:
    '''
    Make an initial greedy guess for a solution to the Traveling Salesman Problem. If there is an odd number of vertices, then
    this method will drop the last vertex to make an even number of vertices; the idea being that we wish to work with
    such a large number of vertices that one single vertex won't make much difference.

    The guess is made by initially pairing up vertices to their closest neighbors. Then iteratively start choosing path segment
    pairs to connect (from either end) based on minimizing the distance added. We try to keep the legnths of the path segments
    uniform, but we don't necessarily do this. We connect segments in order, i.e. we try connecting a given segment to the
    segments that occur after its position in our list of curves. Odd number of curve segments will force segments to
    not be the same size, but this nonuniformity is minimized. 

    Heuristically, this uniformity in size seems like a good idea, because it should allow a good portion of each segment to
    be locally correct because its construction is relatively independent of the other curves.

    We are finished when there is only one segment left. Note that we don't have any guarantee that the beginning of the segment
    is close to the end of the segment.

    Members
    -------
    curves : List of Numpy Arrays, each of shape (segmentVertexCount, 2)
        The list of current path segments. Each one is a numpy array of shape (segmentVertexCount, 2); so the xy-coordinates of the
        vertices in the segment in the order they appear in the segment. Note that the different segments don't have
        the same number of vertices. 
    
    endPts : Dictionary with Values in Numpy Arrays of Shape (SegmentCount, 2)  
        This keeps track of the beginning vertex of each segment in self.curves and the end vertex in self.curves. This is for
        quick numpy calculations for finding the shortest links when connecting existing segments. 
        endPts['begin'] = The xy-coordinates of the beginning vertex of each path segment in self.curves.
        endPts['end'] = The xy-coordinates of the ending vertex of each path segment in self.curves.

    vertices :
        The xy-coordinates of the vertices in their original order. Note that we force this to be an even number of vertices,
        dropping the last vertex if we are in the case of an odd number of vertices.

    nVertices : Int
        The number of vertices we are working with. We force this to be even.
    '''

    def __init__(self):
        '''
        Initialize all of the members to None.
        '''

        self.curves = None
        self.endPts = None
        self.vertices = None
        self.nVertices = None

    def makeGuess(self, vertices):
        '''
        We make a greedy guess for a list of vertices.

        Parameters
        ----------
        vertices : Numpy array of shape (nVertices, 2)
            The list of xy-coordinates of the vertices we wish to make a greedy guess of the solution to TSP for; however 
            we require that vertices only have an even number of members. If it has an odd number of vertices, then the 
            last vertex will be dropped.

        Returns
        -------
        Numpy array of shape (newNVertices, 2)
            The xy-coordinates in order of our greedy guess. Note that if we passed an odd number of vertices then
            the number of vertices by one; in the case that we passed an even number of vertices, the number of
            vertices stays the same.
        '''

        self.vertices = vertices
        self.nVertices = len(vertices)
        self.curves = []
   
        # Drop the last vertex if there is an odd number of vertices.
 
        if self.nVertices % 2 == 1:

            self.vertices = self.vertices[:-1]
            self.nVertices -= 1

        # Do the initial greedy pairing to create our initial curve segments. 

        self.__initializeCurves()       

        # While we have more than one curve segment, greedily connect segments.

        while len(self.curves) > 1:

            self.__connectCurves()

        # When we are left with only one curve segment, it is our greedy guess.

        return self.curves[0]

    def __initializeCurves(self):
        '''
        Do the initial greedy pairing. 
        '''

        nProcessed = 0

        beginPts = []
        endPts = []

        # For each vertex not added to a pair so far, find its nearest neighbor out of
        # the vertices that haven't been put in a pair so far.
        # For ease of determining which ones haven't been paired so far, we switch the paired
        # vertices to occur at the beginning of the array, so that all of the vertices to
        # the right of our current position are exactly the vertices that haven't been
        # processed so far.

        while nProcessed < self.nVertices:

            # Get the unpaired vertices to the right of our current position.

            candidateStart = nProcessed + 1
            candidateVerts = self.vertices[candidateStart :]

            # Find the candidate that gives the shortest connection.

            distances = np.linalg.norm(self.vertices[nProcessed] - candidateVerts, axis = -1)
            partnerI = np.argmin(distances, axis = 0) + candidateStart 

            # Swap the chosen candidate to be directly after the current position.

            self.__swapVertices(nProcessed+1, partnerI)

            # The pair gives a new curve; update the list of curves, list of beginning points,
            # and the list of end points.

            newCurve = np.array([self.vertices[nProcessed], self.vertices[nProcessed + 1]])
            self.curves.append(newCurve)
            beginPts.append(self.vertices[nProcessed])
            endPts.append(self.vertices[nProcessed + 1])

            # We processed a pair so we need to advance by 2 (recall that the next position is
            # now the chosen candidate).

            nProcessed += 2

        self.endPts = {'begin' : np.array(beginPts),
                       'end' : np.array(endPts) }

    def __connectCurves(self):
        '''
        Go through the curves in order and greedily connect them to another curve after their position in such 
        a way that gives the shortest connection. We minimize the nonuniformity in the segment size, but when there
        is an odd number of segments there will be segments of different sizes.
        '''

        nCurves = len(self.curves)
        curveI = 0

        newCurves = []


        # For each curve we find the shortest connection for all curves in a position
        # of our list of curves that occurs after the current curve. We do this to
        # try to keep the curves all of the same size (although they won't necessarily be).
        # We iterate until curveI == nCurves - 2, because we want to make sure that
        # there are atleast two curves left to join. 

        while curveI < nCurves - 1:

            sourceEndPt, destinationI, destEndPt = self.__findShortestConnection(curveI)
            self.__connectSource(sourceEndPt, curveI, destinationI, destEndPt)
            self.__removeCurve(destinationI)

            curveI += 1 
            nCurves -= 1

    def __connectSource(self, sourceEndPt, sourceI, destinationI, destEndPt):
        '''
        The connection is made so that the new connected curve always starts at one of the endpoints of the source
        and ends at one of the endpoints of the destination.

        The new curve replaces the curve at self.curves[sourceI]. This function will not delete/remove the other curve;
        it will need to be removed using self.__removeCurve(). 

        Parameters
        ----------
        sourceEndPt : String
            Should be either 'begin' or 'end' to indicate which side of the source curve the connection should
            be made.

        sourceI : Int
            The index of the source curve for the connection.

        destinationI : Int
            The index of the destination curve for the connection.

        destEndPt : String
            Should be either 'begin' or 'end' to indicate which side of the destination curve the connection
            should be made.
        '''

        # When the we connect to the beginning of the source curve, then we need to flip its order.

        if sourceEndPt == 'end':

            sourceCurve = self.curves[sourceI]

        else:

            sourceCurve = np.flip(self.curves[sourceI], axis  = 0)
            self.endPts['begin'][sourceI] = self.endPts['end'][sourceI]

        # When we connect to the end of the destination curve, then we need to flip its order.

        if destEndPt == 'begin':

            destCurve = self.curves[destinationI]
            self.endPts['end'][sourceI] = self.endPts['end'][destinationI]

        else:

            destCurve = np.flip(self.curves[destinationI], axis = 0)
            self.endPts['end'][sourceI] = self.endPts['begin'][destinationI]

        newCurve = np.concatenate([sourceCurve, destCurve], axis = 0)
        self.curves[sourceI] = newCurve

    def __removeCurve(self, curveI):
        '''
        Remove a curve, updating the list of curves and list of endpoints.

        Parameters
        ----------
        curveI : Int
            The index of the curve to remove.

        '''

        del self.curves[curveI]

        for endPt in ['begin', 'end']:

            self.endPts[endPt] = np.delete(self.endPts[endPt], curveI, axis = 0)

 
    def __findShortestConnection(self, sourceI): 
        '''
        Find the curve after the given curve at index sourceI that will give the shortest
        connection to the curve at self.curves[sourceI]. Note, the connection only
        considers connecting endpoints.

        Parameters
        ----------
        sourceI : Int
            The index of the curve to consider connecting to other curves.

        Returns
        -------
        bestSourceEndPt : String
            bestSourceEndPt is either 'begin' or 'end' to indicate which source endpoint
            should be used for the connection.

        bestDestinationI : Int
            bestDestinationI is the index of the curve that we should connect the source curve to.

        bestDestEndPt : String
            Either 'begin' or 'end' to indicate which endpoint of the destination curve to connect to. 
        '''


        # A negative bestDistance indicates that we haven't found any distances yet.

        bestDist = -1 

        bestSourceEndPt = '' 
        bestDestinationI = -1
        bestDestEndPt = ''

        # Loop over beginning and ending for the endpoints of source and destination.

        for source in self.endPts.keys():

            sourceVertex = self.endPts[source][sourceI]

            for destination in self.endPts.keys():

                # To try to keep curve size uniform, we only consider connecting
                # to curves after the source curve (which shouldn't have been connected yet
                # in this pass). 

                candidates = self.endPts[destination][sourceI + 1 :, : ]
                distances = np.linalg.norm(sourceVertex - candidates, axis = -1)

                minDistI = np.argmin(distances) 
                dist = distances[minDistI]

                # If we have a new best distance then record needed info.

                if bestDist < 0 or dist < bestDist:
                    bestDist = dist
                    bestSourceEndPt = source
                    
                    # Make sure to account for the fact that indices of candidates is offset
                    # from indices in self.curves.

                    bestDestinationI = minDistI + sourceI + 1
                    bestDestEndPt = destination

        return bestSourceEndPt, bestDestinationI, bestDestEndPt

    def __swapVertices(self, i, j):
        '''
        Swap vertices at indices i and j inside self.vertices.

        Parameters
        ----------
        i : Int
        
        j : Int
        '''

        temp = self.vertices[i, :].copy()
        self.vertices[i, :] = self.vertices[j, :].copy()
        self.vertices[j, :] = temp.copy()
     
##########################
#### Dithering Maker
##########################

class DitheringMaker:
    '''
    Performs dithering on an image using the classical Floyd-Steinberg dithering algorithm. The dithering isn't
    applied to the last row or the edge columns. Instead the last row and edge columns are made all white as this 
    will later not add any vertices when we extract vertices from the dithering.

    Members
    -------
    dithering : Numpy array of Int of Shape(nRows, nCol)
        The array holding the dithering of the image. Note that the last row and edge columns will always
        be converted to white (i.e. 255). 

    rowDisp : Numpy array of Int of shape (2, 3)
        The row offsets to apply the diffusions to.

    colDisp : Numpy array of Int of shape (2, 3)
        The column offsets to apply the diffusions to.

    diffusionProp : Numpy array of Float of shape (2, 3)
        The Floyd-Steinberg coefficients for diffusing the error in the
        dithering. 
    '''

    def __init__(self):
        '''
        Initialize the dithering to None as we haven't performed any yet.

        The displacement of indices and the diffusion coefficients are for the classic
        Floyd-Steinberg dithering algorithm.
        '''

        self.dithering = None

        self.rowDisp = np.full((2,3), np.arange(0, 2)[:, np.newaxis], dtype = 'int')
        self.colDisp = np.full((2,3), np.arange(-1,2), dtype = 'int')
        self.diffusionProp = np.array([[ 0, 0, 7],
                                        [ 3, 5, 1]]) / 16

    def makeDithering(self, pixels, cutoff = 255 / 2):
        '''
        Apply the classic Floyd-Steinberg dithering algorithm, but for simplicity
        we just make the edge columns and the last row all white (which will give
        us no vertices when we later extract vertices).

        Parameters
        ----------
        pixels : Numpy array of Int of shape (nRows, nCols)
            The gray scale pixels to apply the dithering to.

        cutoff : Float
            The cut off for making a dithering pixel either 0 or 255.

        Returns
        -------
        Numpy array of Int of Shape (nRows, nCols)
            The final dithering; each pixel is either 0 or 255 (black or white).
        '''

        # We use Floyd-Steinberg dithering. Error diffusion is
        # _     x     7/16
        # 3/16  5/16  1/16

        self.dithering = pixels.copy().astype('float') 

        nRows, nCols = pixels.shape

        # Initialize the first column to be white.

        self.dithering[:][0] = 255 

        # Iterate over each row, applying the dithering and diffusing the error. 

        for row in range(nRows - 1):
            for col in range(1, nCols - 1):
                
                dither, error = self.getDither(row, col, cutoff)
                self.dithering[row, col] = dither
                self.diffuseError(error, row, col)

        # Make the last column and the last row all white.

        self.dithering[:, -1] = 255 
        self.dithering[-1, :] = 255 


        # Convert dithering to Numpy array of Int.

        self.dithering = self.dithering.astype('int')

        return self.dithering

    def getDither(self, row, col, cutoff):
        '''
        Turn (dithered) pixel into either 0 or 255 using cutoff.

        Parameters
        ----------
        row : Int
            Index of pixel row.

        col : Int
            Index of pixel column.
    
        cutoff : Float
            The cutoff value to use for converting dithering value to either 0 or 255 (black or white).

        Returns
        -------
        dither : Float
            Floating point value that is either 0.0 or 255.0 (black or white).

        error : Float
            The error in applying the conversion, this needs to be diffused to other pixels
            according to the dithering algorithm. 
        '''

        pixel = self.dithering[row][col]

        if pixel < cutoff: 
            dither = 0.0
        else:
            dither = 255.0

        error = pixel - dither 

        return dither, error 

    def diffuseError(self, error, row, col):
        '''
        Diffuse the error from a (dithered) pixel conversion to black or white. The diffusion
        is applied to the neighbors of the pixel at position [row, col] according to the
        Floyd-Steinberg algorithm.

        Parameters
        ----------
        error : Float
            The size of error to diffuse to other pixels.

        row : Int
            The row index of where the conversion took place.

        col : Int
            The column index of where the conversion took place.
        '''
        
        self.dithering[row + self.rowDisp, col + self.colDisp] += error * self.diffusionProp

def getVertices(dithering):
    '''
    Get the vertices from a black and white image, not grayscale (in particular a dithered image). 
    Every black pixel (value 0.0) gives a vertex.

    Parameters
    ----------
    dithering : Numpy array of shape (nRows, nCols)
        The array of pixels for the dithered image.

    Returns
    -------
    Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.
    '''

    nRows, nCols = dithering.shape

    # Each black pixel gives a vertex.
    keepPixelMask = (dithering == 0)

    # Get the row and column indices of the vertices.

    rows = np.full(dithering.shape, np.arange(nRows)[:, np.newaxis]).reshape(-1)
    cols = np.full(dithering.shape, np.arange(nCols)).reshape(-1)

    rows = rows[keepPixelMask.reshape(-1)]
    cols = cols[keepPixelMask.reshape(-1)]

    # Get the xy-coordinate of the vertices. Make sure to transform row index so
    # that the last row has y value 0.

    vertices = np.stack([cols, nRows - rows], axis = -1)

    return vertices


def plotCycle(cycle, title, doScatter = True, figsize = (5, 5)):
    ''' 
    Plot a cycle through all of the vertices. Can optionally do a scatter plot of all the vertices.

    Parameters
    ----------
    cycle : numpy array of floats of shape (nVertices, 2)
        Holds the vertices of the cycle. The endpoints should be the same to make a closed cycle,
        i.e. cycle[0] == cycle[nVertices - 1].

    title : String
        The title of the graph.

    doScatter : Boolean
        Whether to do a scatter plot of the different vertices. Default is True, i.e. do the
        scatter plot.
    '''

    plt.figure(figsize = figsize) 
    plt.plot(cycle[:, 0], cycle[:, 1])
    if doScatter:
        plt.scatter(cycle[:, 0], cycle[:, 1], color = 'red')
    ax = plt.gca()
    ax.set_title(title)

def plotEnergies(energies, title):
    '''
    Plot energies collected while running the annealing algorithm.

    Parameters
    ----------
    energies : numpy array of shape (Num Energies)
        The energies to graph.

    title : String
        The title of the graph.
    '''

    plt.figure(figsize = (5, 5))
    plt.plot(np.log(energies) / np.log(10))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel('Nth Run of Annealing')
    ax.set_ylabel('Energy')

def normalizeVertices(vertices):
    '''
    Normalize the xy-coordinates by scaling by the reciprocal of the largest y-coordinate.

    Parameters
    ----------
    vertices : Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (nVertices, 2)
        The normalized xy-coordinates of the vertices. 
    '''

    scale = np.amax(vertices[:, 1])
    vertices = vertices.astype('float')
    vertices[:, 0] = vertices[:, 0] / scale
    vertices[:, 1] = vertices[:, 1] / scale

    return vertices

def doAnnealingJobs(annealer, nJobs):
    '''
    Do the annealing jobs (or annealing runs) for a particular annealer. This will perform a
    warm restart of the annealer between each job.

    After each job, it collects information on energy and prints information on the job. 

    Parameters
    ----------
    annealer : An annealing iterator class
        The annealer to do the job.

    nJobs : Int
        The number of jobs to run.

    Returns
    -------
        The energies (or length) of the path at the end of each job.
    '''

    energies = [annealer.getEnergy()]
    
    for i in np.arange(nJobs):
        print('Annealing Job ', i)
       
        annealer.doWarmRestart() 
        for step in annealer:
            continue
   
        energy = annealer.getEnergy() 
        energies.append(energy)

        print(annealer.getInfoString())
    
    energies = np.array(energies)

    return energies

def getPixels(image, ds = 1):
    '''
    Get the pixels as a numpy array from a PIL image.
    We can take the mean of each ds x ds subsquare as an array element inorder to down-size
    the size of the image if we want to.

    Parameters
    ----------
    image : PIL Image
        The PIL image to convert.

    ds : Int
        We take the mean of each ds x ds sub-square for a single element of our array. 

    Returns
    -------
    2d Numpy array of floats
        The converted values of the pixels in the image. We use mean because we
        possibly took a mean over sub-squares.
    '''

    imwidth, imheight = image.size
    pixels = list(image.getdata())
    pixels = np.array(pixels).reshape((imheight, imwidth))
    
    pixels = [[pixels[i:i+ds, j:j+ds].mean() for j in np.arange(0, imwidth, ds)] 
                for i in np.arange(0, imheight, ds)]
    pixels = np.array(pixels)
    return pixels
   
def preprocessVertices(vertices):
    '''
    Normalize vertices and make our intial greedy guess as to the solution of the Traveling Salesman Problem.
    If there is an odd number of vertices, then our greedy guess will drop the last vertex.

    Parameters
    ----------
    vertices : Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (newNVertices, 2)
        The xy-coordinates of our processed vertices.
    '''

    vertices = normalizeVertices(vertices)
    guesser = GreedyGuesser()
    vertices = guesser.makeGuess(vertices)
    vertices = np.roll(vertices, 100, axis = 0)

    return vertices

def savePNG(filename):
    '''
    Save the current pyplot figure as a png file, but first reduce the color palette to reduce the file size.

    Parameters
    ----------
    filename : String
        The name of the file to save to.
    '''

    imageIO = io.BytesIO()
    plt.savefig(imageIO, format = 'png')
    imageIO.seek(0)
    image = Image.open(imageIO)
    image = image.convert('P', palette = Image.WEB)
    image.save(filename , format = 'PNG')


###########################
#### Main executable
##########################

def main():

    # Set the figure size for graphs of cycles.

    cycleFigSize = (8, 8)
    finalCycleSize = (10, 10)

    # Open the image.

    image = Image.open('2018-06-09-graphs/tigerHeadResize.png').convert('L')
    pixels = getPixels(image, ds = 1)
    plt.imshow(pixels, cmap = 'gray')
    plt.title('Grayscale Version of Tiger Head')
    plt.tight_layout()
    savePNG('2018-06-09-graphs/grayscale.png')
    plt.show()

    # Get the dithered image.

    ditheringMaker = DitheringMaker()
    dithering = ditheringMaker.makeDithering(pixels)
    plt.imshow(dithering, cmap = 'gray')
    plt.title('Dithering of Grayscale')
    plt.tight_layout()
    savePNG('2018-06-09-graphs/dithering.png')
    plt.show()

    # Get the vertices from the dithered image and then
    # do the preprocessing.
    
    vertices = getVertices(dithering)
    print('Num vertices = ', len(vertices))
    print('Preprocessing Vertices')
    vertices = preprocessVertices(vertices)
    print('Preprocessing Complete')
    plt.scatter(vertices[:, 0], vertices[:, 1], s = 0.5)
    plt.title('Vertices From Dithering')
    plt.show()

    # Plot the result of the greedy guess and other preprocessing.

    cycle = np.concatenate([vertices, [vertices[0]]], axis = 0)
    plotCycle(cycle, 'Greedy Guess Path', doScatter = False, figsize = cycleFigSize) 
    plt.tight_layout()
    savePNG('2018-06-09-graphs/greedyGuess.png')
    plt.show()

    ######################################
    ############# Annealing based on size
    ######################################

    # Set up parameters for annealing based on size scale.
    nVert = len(vertices) 
    initTemp = 0.1 * 1 / np.sqrt(nVert) / np.sqrt(np.pi) 
    nSteps = 5 * 10**5 
    decimalCool = 1.5
    cooling = np.exp(-np.log(10) * decimalCool / nSteps) 
    nJobs = 200

    # Size scale parameters are based on statistics of sizes of current edges.

    distances = np.linalg.norm(vertices[1:] - vertices[:-1], axis = -1)
    initScale = np.percentile(distances, 99.9)
    finalScale = np.percentile(distances, 90.8)
    sizeCooling = np.exp(np.log(finalScale / initScale) /nSteps)
    
    # Set up our annealing steps iterator.
    
    annealingSteps = AnnealerTSPSizeScale(nSteps / nJobs, vertices, initTemp, cooling, initScale, sizeCooling)
    print('Initial Configuration:\n', annealingSteps.getInfoString())
    
    energies = doAnnealingJobs(annealingSteps, nJobs)
 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    plotEnergies(energies, 'Energies for Size Scale Annealing')
    savePNG('2018-06-09-graphs/sizeScaleEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.getCycle()
    plotCycle(cycle, 'Final Path for Size Scale Annealing', doScatter = False, figsize = cycleFigSize)
    plt.tight_layout()
    savePNG('2018-06-09-graphs/afterSizeScale.png')
    plt.show()   

    vertices = cycle[:-1]
    print('Double check: num vertices = ', len(vertices))

    #################################
    ### Now do TSP based on neighbors
    #################################
       
    # Set up parameters for annealing based on neighbors.
 
    nVert = len(vertices) 
    initTemp = 0.1 * 1 / np.sqrt(nVert) / np.sqrt(np.pi) 
    nSteps = 5 * 10**5 
    decimalCool = 1.5
    cooling = np.exp(-np.log(10) * decimalCool / nSteps) 
    nJobs = 200

    # Neighbor parameters are set up by trial and error.

    initNbrs = int(nVert * 0.01)
    initNbrs = 50
    finalNbrs = 3 
    nbrsCooling = np.exp(np.log(finalNbrs / initNbrs) / nSteps) 
    
    # Set up our annealing steps iterator.
    
    annealingSteps = NeighborsAnnealer(nSteps / nJobs, vertices, initTemp, cooling, initNbrs, nbrsCooling)
    print('Initial Configuration:\n', annealingSteps.getInfoString())
    
    # Now run the annealing steps for the vonNeumann.png example.
   
    energies = doAnnealingJobs(annealingSteps, nJobs) 
    print('Finished running annealing jobs') 
    
    # Plot the energies of the annealing process over time.
    
    plotEnergies(energies, 'Energies for Neighbors Annealing')
    savePNG('2018-06-09-graphs/nbrsEnergies.png')
    plt.show()
    
    # Plot the final cycle of the annealing process.
    
    cycle = annealingSteps.getCycle()
    plotCycle(cycle, 'Final Path for Neighbors Annealing', doScatter = False, figsize = finalCycleSize)
    plt.tight_layout()
    savePNG('2018-06-09-graphs/afterNbrs.png')
    plt.show()   

##########################
#### The actual execution 
            
main()     
