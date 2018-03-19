---
layout: post
date: 2018-03-18
title: Metropolis Sampling Using Python Iterators
---

## [Download the Source Code for this Post]({{site . url}}/assets/2018-03-19-MetropolisSamples.py)

In this post we will look at implementing the Metropolis algorithm using iterators in python.
The Metropolis algorithm allows one to create samples for a probability distribution where you know the probability density at any given point. 
We will give a brief overview of how the Metropolis algorithm works in a later section. 

Note that we will be considering the more restrictive case of symmetric steps; so we are just using the Metropolis algorithm and not the more general Metropolis-Hastings algorithm. 
Therefore, especially since we are interested in keeping our class names short, we will use the name "Metropolis" instead of "Metropolis-Hastings". So, hopefully the knowledgeable
reader will not take any offense at our ommission of "Hastings". 

If you are interested in a nice (and legally free!) reference on the Metropolis algorithm, please see [*Information Theory, Inference, and Learning Algorithms* by David MacKay](http://www.inference.org.uk/itprnn/book.pdf). 
However, we will give a very brief overview of the Metropolis algorithm later.

Now let us briefly describe iterators in python. Iterators are classes that are defined to be iterable so that you can make use of the function `next()`. Iterators make sense for
the Metropolis algorithm as the function `next()` should just give the next random sample from our probability distribution. 

We will make use of two iterator classes: `MetropolisWalk` and `IndependentMetropolis`. The class `IndependentMetroplis` gives truly independent samples from our distribution 
(at least to better approximation they are independent); that is, `IndependentMetropolis` gives better quality samples than `MetropolisWalk`. However, we need `MetropolisWalk` as
`IndependentMetropolis` uses `MetropolisWalk` to generate its samples. 

Consecutive samples from `MetropolisWalk` are not truly independent. As the name implies, the samples are drawn from a "walk." That is, one shouldn't expect two consecutive samples
to be very far from each other.

Before we begin, let's import the necessary modules and set a random seed for consistency:

``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set up the random seed for numpy for consistent results.

np.random.seed(20180306)
```

In the next section we will take a look at how to use these iterators, and we will investigate how the samples from `IndependentMetropolis` are better than `MetropolisWalk`.

## Using the Iterators

First, let us decide on a desired probability distribution that we wish to draw independent samples from. We will use two gaussians of equal variance, each centered 
at plus/minus `peakCenter`. 

``` python
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
```

So our desired distrbution looks as follows:

![Desired Probability Distribution]({{site.url}}/assets/2018-03-19-graphs/desired.svg)

Now, it is possible to sample from this distribution using methods more simple than the Metropolis algorithm, but it is a nice simple distribution
with known properties that we can use to test the effectiveness of our iterator classes.
 
Next, let us discuss the ingredients that go into making the Metropolis algorithm work.

1. We need to know the desired probability density, UPTO a normalization factor. It is a nice feature of the Metropolis algorithm that it doesn't explicitly need the 
normalization factor as it is often impossible to compute. We will denote the density (without the normalization factor) by the function `getDesiredDensity()`. 

2. Another random probability distribution that we know how to sample from and will be used to decide how the algorithm randomly walks around the domain of our
desired probability distribution. We will use a uniform probability density, and denote this process by the function `makeUniformStep()`.

3. A starting position to feed into the initialization of the algorithm.

4. A waiting time that tells the algorithm how long to generate new samples before keeping a sample as "independent". The underlying mechanism of the Metropolis
algorithm is a random walk, so consecutive samples generated from the Metropolis algorithm aren't truly independent. They are usually close to each other. To get around
this problem, one simply waits enough iterations that the random walk has had enough opportunity to try to "tour" a huge portion of the domain of our 
desired probability distribution. 

Now, consider a random walk starting at `0` with step size `stepSize`; if we wait `nSteps`, then the standard deviation of the position is `sqrt(nSteps) * stepSize`.
So if our domain has width `domainWidth`, then we should approximately wait `nSteps = (domainWidth / stepSize)**2`, note the square power. Now, our desired distribution actually
has infinite width, so instead of considering the explicit width we can just consider the width of where "most" of the distribution lies. This doesn't need to be too precise, so
we will just use something that seems reasonable.

We have already set up `getDesiredDistribution()`, i.e. item (1) above. Now let us set up the rest of the ingredients:

``` python
# Set up the step distribution.
 
walkR = 2.0
makeUniformStep = lambda : np.random.uniform(low = -walkR, high = walkR)

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
```

Some of the parameters we needed to calculate, so let's take a look at their values:
```
width =  5.82842712475
walkStd =  1.15470053838
wait =  50
```

Now, let's use the iterators to draw samples from our desired distribution. We store the samples in a `pandas.DataFrame` for the convenience of using `pandas` to make histograms.

``` python
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
```

Now let's graph the historgram of the results. We will also compute and graph the true curve one should obtain for as many samples as we took:

``` python
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
```

Let's take a look at the histrograms of our samples:

![Histogram of Dependent Samples]({{site . url}}/assets/2018-03-19-graphs/depHist.svg)

![Histogram of Independent Samples]({{site . url}}/assets/2018-03-19-graphs/indHist.svg)

We can see that the independent samples made by `IndependentMetropolis` look better than the dependent samples drawn by `MetropolisWalk`. In the next section, we
will more rigorously investigate this by computing the running means and variances of the samples when we consider them as a time series.

## Look at Running Means and Variances

To look at the running means and variances for both sets of samples, we will create them using an iterator class for computing running means; recall that the variance of a random variable `X` can be
computed as the difference of two means: `E(X^2) - E(X)^2`. We will call our iterator class `RunningMeans`. 

Before we see the definition of `RunningMeans`, let's see how it is used to get the running statitistics of our samples:

``` python
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
```

Now, let's checkout the results of finding the running means:

``` python
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
```

Let's take a look at the results. Here is a graph of the running means:

![Graph of Running Means]({{site . url}}/assets/2018-03-19-graphs/runningMeans.svg)

It is clear from the graph that the running means of the independent samples drawn by the class `IndependentMetropolis` are converging
much better and faster to the true mean of `0`. We can get an idea of how many decimal places of accuracy these running means achieve by
looking at the base 10 logarithms of the absolute errors:

![Graph of Log Errors Running Means]({{site . url}}/assets/2018-03-19-graphs/logErrorsMeans.svg)

So it looks like we have about two decimal places of accuracy for the independent samples. We can check this with the last running mean of the samples:

```
True value of means =  0.0
Last Running Mean of Dependent means =  0.154066380049
Last Running Mean of Independent means =  0.00536688753315
```

Its almost accurate to two decimal places, it actually rounds up to `0.01`, but it is pretty close to roughly two decimal places.

Now let's check the variances:

![Graph of Running Variances]({{site . url}}/assets/2018-03-19-graphs/runningVariances.svg)

We see that the independent samples again have much better convergence to the true value. Let's check the log-errors:

![Graph of Log Errors of Running Variances]({{site . url}}/assets/2018-03-19-graphs/logErrorsVariances.svg)

So the variance of the independent samples seem to only be accurate to one decimal place. Let's check this:

```
True value of vars =  2.75
Last Running Mean of Dependent vars =  2.49395119812
Last Running Mean of Independent vars =  2.70635846612
```

We see that yes, it is only accurate to one decimal place (i.e. 2 significant digits).

Next, let's see how to find the running means using an iterator class.

## Code for Running Means

At initialization, the class `RunningMeans` takes an iterable list and makes itself into an iterator for each running mean of the values in the list:

``` python
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
``` 

Next, let's take a brief look at how the Metropolis algorithm works.

## How Metropolis Works

The idea is that you randomly walk around the domain of your desired probability distribution using some other simple probability distribution that we know how to draw samples from.
In our case, this is a uniform distribution.

Each step of this random walk results in a "proposal". The next sample is determined by a simple process:

1. If the probability density at the proposal is higher than the last step, then the proposal is accepted.
2. When the probability density at the proposal is NOT higher, then we randomly accept the proposal with probability given the ratio of the proposal density to the last location density. 
3. If the proposal was accepted, then we move to the proposal location and the proposal is a new sample. If the proposal was rejected, then we stay in the same location and the new sample
is our previous sample AGAIN.

Note that in the case of rejection in step 3, we repeat our previous sample. As pointed out in the original paper by Metropolis, etal., this is absolutely necessary for the flow of probabilities
in this markov process to balance out to the correct equilibrium state. Without this repetition, there will not be convergence.

To obtain truly independent samples, we need to wait long enough for our random walk over the domain to move far enough (or at least have the opportunity to move far enough). So as described in 
a previous section, we need to wait enough iterations of the Metropolis algorithm to collect a truly independent sample.

Next, let's look at implementing the dependent sample walk part of the Metropolis algorithm.

## Code for `MetropolisWalk`

``` python
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
```

Next, let's look at the code for drawing independent samples using the Metropolis algorithm.

## Code for class `IndependentMetropolis`

Finally, let's look at the code for the class `IndependentMetropolis`.
``` python
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
```

## [Download the Source Code for this Post]({{site . url}}/assets/2018-03-19-MetropolisSamples.py)

