import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set up the random seed for numpy for consistent results.

np.random.seed(20180306)

##################################################
#### Class Definitions 
##################################################

class MetropolisWalk:
    '''
        Iterator class for generating a walk of the metropolis algorithm, i.e. it doesn't introduce waiting
        between samples. The metropolis algorithm is used to generate samples for a probability distribution
        when the probability density is explicitly known. The samples are generated by a walk depending on a 
        distribution that we know how to sample from, e.g. uniform probability distribution.

        Class Members
        -------------
        self.i : Int
            The current sample number.
        self.n : Int
            The total number of samples to collect.
        self.position : Float
            The initial position to start the Metropolis algorithm at.

        self.density : Float
            The probability density at the current position. We record the value at the current point in
            case probability density evaluation is expensive. 

        self.getDensity : Function Reference
            The density function for the probability distribution that we wish to draw samples from. 

            Parameters
            ----------
            pos : Float
            The position at which to evaluate the probability density.

            Returns
            -------
            Float
                The probability density at this point. Note that this does not have to be between 0 and 1, but it
                should be non-negative.

        self.makeStep : Function reference
            The function for making the next step in the walk that we know how to draw samples from. Explicitly,
            it should give the displacement to move the current position. The displacement should be drawn from 
            a probability distribution that we know how to sample. For example, makeStep could return the result
            of sampling from some fixed uniform distribution. 
            
            Parameters
            ----------
            None

            Returns
            -------
            Float
                The displacement to move from the current position.

        self.nAccepted : Int
            Records the number of times our walk of the metropolis algorithm has accepted a move. Can be used to
            get an idea of the rate with which the Metropolis algorithm is rejecting moves proposed by the 
            underlying random walk. If the rate is too high, then the walk may need to take smaller steps.
            The trade off is that the waiting time for independent samples increases. 

            The class MetropolisWalk doesn't handle waiting, but it is used to generate samples in the class  
            IndependentMetropolis that does.
    '''

    def __init__(self, start, n, getDensity, makeStep): 
        '''

        Initializer. Initialize self.i to 0, set self.density to getDensity(start), and set self.nAccepted to 0.

        Parameters
        ----------
        start : Float
            Initial position to start the Metropolis walk at.

        n : Int
            The total number of samples to draw.

        getDensity : Function Reference 
            The function to use for finding the probability density at a given point. For details on how the
            function should behave, see the docstring for self.getDensity for the class MetropolisWalk.

        makeStep : Function Reference
            The function to use for sampling the displacement from the current position in the underlying random
            walk. For details on how the function should behave, see the docstring for self.makeStep for the
            class MetropolisWalk.
        '''

        self.i = 0
        self.n = n
        self.position = start

        # Get the density at the initial position.

        self.density = getDensity(start)

        self.getDensity = getDensity 
        self.makeStep = makeStep 

        # Initialized the number of accepted moves as zero.

        self.nAccepted = 0

    def __iter__(self):
        '''
        Called to get reference to an iterator, which is just itself.

        Returns
        -------
        Reference to self.
        '''

        return self

    def __next__(self):
        '''
        Called to generate next sample in the iterator.

        Returns
        -------
        Float
            Returns the next sample position.
        '''

        # First handle check to see if there are more samples that need to be drawn.

        if self.i < self.n:

            # Generate a proposed move.

            proposal = self.position + self.makeStep()

            # As per the Metropolis algorithm, if the probability density at the proposal is higher
            # than the density at the current position, we automatically move to the proposal. 
            
            # On the other hand, if the proposal density is lower, then we use a random Bernoulli trial
            # to decide whether to accept the proposal or stay where we are. The probability of moving
            # is the ratio of the proposal density to the current density. The random Bernoulli trial 
            # is accomplished using a uniform distribution. 

            propDensity = self.getDensity(proposal) 
            if propDensity > self.density:
                accepted = True

            else:
                sample = np.random.uniform()
                accepted = sample < (propDensity / self.density)

            if accepted:
                self.nAccepted += 1
                self.position = proposal
                self.density = propDensity

            # Increase the sample counter.
 
            self.i += 1

            # As per Metropolis, we return the current position as a sample, even if we rejected 
            # the proposal.            
            return self.position

        # If we have drawn enough samples, then we raise an excpetion to stop the iteration.

        else:
            raise StopIteration()

class IndependentMetropolis(MetropolisWalk):
    '''
    Iterator class for getting independent samples of a given probability distribution using the Metropolis
    algorithm. This class uses the class MetropolisWalk to draw a user specified number of dependent samples
    (the number of such samples is also called the waiting time). Hopefully, if the user has specified a 
    long enough waiting time, then the walk has had enough time to cover the domain of the distribution. 
    So hopefully, the last sample from the class MetropolisWalk will be independent of the last
    sample collected by the class IndependentMetropolis. So the class IndependentMetropolis only keeps
    the last sample collected by the class MetropolisWalk. 

    The class IndependentMetropolis inherits a large part of its initialization and members from
    the class IndependentMetropolis.

    Members
    -------
    self.wait : Int
        Non-negative number of samples to draw from class MetropolisWalk before keeping a new sample. This is
        used to help make sure the samples are closer to independent.

    See members for class MetropolisWalk. 

    '''

    def __init__(self, start, n, getDensity, makeStep, wait): 
        '''
        Initializer. Calls the baseclass initializer MetropolisWalk.__init__. Also sets up the wait time
        for drawing indpendent samples.

        Parameters
        ----------
        See parameters for MetropolisWalk.__init__.

        self.wait : Int
            Non-negative number of samples from MetropolisWalk to ignore before keeping the sample. This is
            used to help the samples be independent by giving the underlying random walk a chance to cover
            the enough of the domain of the probability distribution.
        '''

        MetropolisWalk.__init__(self, start, n, getDensity, makeStep)
        self.wait = wait

    def __next__(self):
        '''
        Get the next independent sample from the probability distribution using the Metropolis algorithm with
        waiting.

        Returns
        -------
        Float
            The next independent sample from the distribution. 
        '''      

        # If we don't have enough samples, then we need to do a Metropolis walk for a number
        # of samples equalling the waiting time. 

        if self.i < self.n:
 
            walk = MetropolisWalk(self.position, self.wait, self.getDensity, self.makeStep)
            nextSample = None
    
            # We don't record the samples of the walk before the end of the waiting time, but
            # we still need to step through the iterator.

            for nextSample in walk:
               pass 

            self.i += 1
            self.position = nextSample

            return nextSample

        # If we already have enough samples, then raise an exception to stop iteration.
        else:

            raise StopIteration()

class RunningMeans:
    '''
    Iterator class for finding the running means of a time series of data. 

    Members
    -------
    self.valIter : Iterator Class
        Iterator for the time series data. 

    self.valSum : Float
        Current running sum of the data.

    self.i : Int
        The number of samples from the time series processed so far. 
    '''

    def __init__(self, valList):
        '''
        Initializer. 
        
        Parameters
        ----------
        valList : Iterable Object
            Time series data. Should be able to create an iterator based on valList.

        '''
        self.valIter = iter(valList)
        self.valSum = 0.0
        self.i = 0

    def __iter__(self):
        '''
        Returns self as reference to iterator.
        '''
        return self

    def __next__(self):
        '''
        Get the next running mean.

        Returns
        -------
        Float
            The next running mean.
        '''

        # First try to get the next value from the time series. If there is no next value,
        # then we are done, and we raise an exception to stop the iteration.

        try:
            val = next(self.valIter)

        except StopIteration:
            raise StopIteration()

        # Update the necessary members and return the mean.

        self.valSum += val
        self.i += 1
       
        return self.valSum / self.i

########################################################
##### Main Execution
########################################################

# Set up the step distribution.
 
walkR = 2.0
makeUniformStep = lambda : np.random.uniform(low = -walkR, high = walkR)

# Set up the non-normalized probability density for the distribution we wish to sample.
# The distribution consists of two gaussian peaks at plus and minus peakCenter, both with
# variance of 1/2.

peakCenter = 1.5
getDesiredDensity = lambda x : np.exp(-(x-peakCenter)**2) + np.exp(-(x+peakCenter)**2)

# The normalization constant for the probability distribution.

normalization = 1 / np.sqrt(np.pi) * 0.5

# Plot what the probability distribution looks like.

xvals = np.arange(-6, 6, 0.1)
plt.plot(xvals, getDesiredDensity(xvals) * normalization)
plt.gca().set_title('True Probability Density')
plt.gca().set_ylabel('Density')
plt.savefig('2018-03-19-graphs/desired.svg')
plt.show()

# Set up parameters for doing Metropolis sampling.

start = 0   # Where to start sampling. 
nSamples = 10**3   # Number of samples to take. 

normalStd = np.sqrt(0.5)    # The standard deviation of each gaussian peak.
walkStd = np.sqrt(1/3) * walkR  # The standard deviation of the step distribution.
width = 2 * peakCenter + 4 * normalStd  # Approximate width of the probability distribution.

# Distance traveled = walkStd * sqrt(N); needs to be atleast width of distribution.

# The number of steps to wait before sample is now considered independent.

wait = int(2 * (width/ walkStd)**2)

# Print out some of the parameters.

print('width = ', width)
print('walkStd = ', walkStd)
print('wait = ', wait)

# Get samples that are dependent.

print('\nRunning Metropolis Walk')
walk = MetropolisWalk(start, nSamples, getDensity = getDesiredDensity, makeStep = makeUniformStep)
depSteps = list(walk)
depSteps = pd.DataFrame(depSteps, columns = ['position'])

# Get independent samples by waiting enough steps to take a sample. 

print('\nRunning Independent Metropolis Sampler')
indMet = IndependentMetropolis(start, nSamples, getDensity = getDesiredDensity, makeStep = makeUniformStep, wait = wait)
indSteps = list(indMet)
indSteps = pd.DataFrame(indSteps, columns = ['position'])

# Some parameters used for graphing the results.

dx = 0.1
dBin = 0.25
bins = np.arange(-6, 6, 0.25)
trueX = np.arange(-6, 6, dx)

# Compute the true curve for the size of this population.

true = getDesiredDensity(trueX) * normalization * dBin * len(depSteps.position) 

# Do a combination of histogram of walk values and true curve.

depSteps.position.hist(bins = bins)
plt.plot(trueX, true, color = 'red', linewidth = 3)
plt.legend(['True Values'])
plt.gca().set_title('Dependent Samples')
plt.savefig('2018-03-19-graphs/depHist.svg')
plt.show()

# Do a combination of histogram of independent samples and the true curve.

indSteps.position.hist(bins = bins)

# Get the true curve.

true = getDesiredDensity(trueX) * normalization * dBin * len(indSteps.position)

plt.plot(trueX, true, color = 'red', linewidth = 3)
plt.legend(['True Values'])
plt.gca().set_title('Independent Metropolis')
plt.savefig('2018-03-19-graphs/indHist.svg')
plt.show()

print('Let\'s look at the running mean for the different cases')

# Compute the running mean and the running variances.

running = {}
for posList, name in zip([depSteps.position, indSteps.position], ['Dep', 'Ind']):
    newKey = 'means' + name
    running[newKey] = list(RunningMeans(posList))
    running[newKey] = np.array(running[newKey])

    newKey2 = 'vars' + name
    running[newKey2] = list(RunningMeans(posList**2))
    running[newKey2] = np.array(running[newKey2]) - running[newKey]**2

true = {'means' : 0.0,
        'vars' : 0.5 + peakCenter**2 }

# Graph the running means and variations.

for stat, name in zip(['means', 'vars'], ['Means', 'Variances']):

    print('True value of ' + stat + ' = ', true[stat])
    print('Last Running Mean of Dependent ' + stat + ' = ', running[stat + 'Dep'][-1])
    print('Last Running Mean of Independent ' + stat + ' = ', running[stat + 'Ind'][-1])

    # Plot the running stats.

    plt.plot(running[stat + 'Dep'])
    plt.plot(running[stat + 'Ind'])
    plt.gca().axhline(true[stat], color = 'red')
    plt.legend(['Dependent Samples', 'Independent Samples', 'True Value'], loc = (0.5, 0.05))
    plt.gca().set_title('Running ' + name + ' vs Sample Number')
    plt.gca().set_xlabel('Sample Number')
    plt.gca().set_ylabel('Running ' + name)
    plt.savefig('2018-03-19-graphs/running' + name + '.svg')
    plt.show()

    # Get the plot of log errors.

    for dependency in ['Dep', 'Ind']:
        absError = np.abs(running[stat + dependency] - true[stat])
        logError = np.log(absError) / np.log(10)
        plt.plot(logError)
    plt.gca().set_title('Log Base 10 of Error of Running ' + name)
    plt.gca().set_xlabel('Sample Number')
    plt.gca().set_ylabel('Log Error')
    plt.gca().legend(['DependentSamples', 'Independent Samples'], loc = (0.5, 0.05))
    plt.savefig('2018-03-19-graphs/logErrors' + name + '.svg')
    plt.show()
