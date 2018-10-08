---
layout: post
date: 2018-02-21
title: Regression for Randomly Choosing To Run A Poisson Process
tags: [Python, Statistics] 
---

## [Download the Source Code for this Post]({{site . url}}/assets/2018-02-21-RandomlyChoosingPoisson.py)
## [Download the Notes for Calculations for the Algorithm in This Post]({{site . url}}/assets/2018-02-21-calculations.pdf)
## [Download the LaTeX Source for the Calculation Notes]({{site . url}}/assets/2018-02-21-calculations.tex)

We will be looking at doing regression for a random process that depends on a length of time `T`; the process is done in two steps:

1. First run a Bernoulli trial to determine whether or not to run a Poisson process. We will denote the probability that we run
a Poisson process by `pRun`. If we do not run a poisson process, then we just return a count of 0; else, we continue on to step 2.

2. We now run a Poisson process with fixed poisson density denoted by `pDensity`; so the Poisson rate is the product `T * pDensity`. So our model simply
uses a Poisson rate that scales linearly with the length of the interval of time `T`. 

We will explicitly generate samples from such a random process, fixing the parameters to be of `pRun = 0.75` and `pDensity = 0.015`. The lengths of time `T` for each data point will be drawn
from a uniform distribution from `T = 0` to `T = 100`. We will make `5000` samples.

Then using simply the data of times and counts, we will use regression to try to reproduce the true values of `pRun` and `pDensity`. The regression will be done by 
maximizing the log-likelihoods of the two parameters given the data samples. This optimization is done using the Newton method to find where gradient vanishes.

After the regression we will find fitted values of our parameters to be 

```
Fitted pRun = 0.75413365449
Fitted pDensity = 0.0145516689881
```

We will need to import the following modules for this post:

```python
import numpy as np    # Necessary for calculations.
import matplotlib.pyplot as plt    # For plotting results.
import pandas as pd    # For histogram creation.
```
## Generating Samples of the Random Process

Now let's generate the samples for our random process. First, let's fix a random seed for consistency:

```python
np.random.seed(20180223)
```

Now let's fix the parameters used to generate the samples of our random process:

```python
# Set up parameters for sampling the time lengths and the parameters for the model.

maxLength = 100
pDensity = 1.5 / maxLength  
probDoPoisson = 0.75
nSamples = 5000

print('True poisson density = ', pDensity)
print('True probability of running poisson = ', probDoPoisson)
```

We have that
```
True poisson density =  0.015
True probability of running poisson =  0.75
```

Now let's generate the samples of lengths of time `T`:

```python
# Set up sample of length by using a uniform distribution from 0 to maxLength.

lengths = np.random.uniform(0, maxLength, size = nSamples)
```

When we set up the counts, we create samples of uniform distribution and samples of poisson distribution for all samples. The uniform distribution
is our Bernoulli trial to determine whether to run a Poisson process. So for each uniform sample in the correct range of values, we keep the poisson sample
as the count. For those uniform samples that are not in the right range, we just set the count to be zero.

```python
# Now set up the samples of the counts. First set up samples of bernoulli trial.
counts = np.random.uniform(size = lengths.shape)

# Now set up whether to include a poisson sample based on probDoPoisson. If not, then just make the count 0.

doPoissonMask = counts < probDoPoisson
counts[~doPoissonMask] = 0
poissons = [np.random.poisson(pDensity * x) for x in lengths]
counts[doPoissonMask] = np.array(poissons)[doPoissonMask]
```

Let's put our sample into a pandas dataframe to make histograms of what they look like.

``` python
# Put the samples in a pandas dataframe to make some histograms.

samples = pd.DataFrame(counts, columns = ['counts'])
samples['lengths'] = lengths
print(samples.head())

samples.lengths.hist()
plt.gca().set_title('Histogram of Lengths of Time')
plt.savefig('2018-02-21-graphs/lengths.svg')
plt.show()

samples.counts.hist()
plt.gca().set_title('Histogram of Counts')
plt.savefig('2018-02-21-graphs/counts.svg')
plt.show()
```

We find that we get:

![Histogram of Lengths]({{site . url}}/assets/2018-02-21-graphs/lengths.svg)

![Histogram of Counts]({{site . url}}/assets/2018-02-21-graphs/counts.svg)

Next, let's look at getting some necessary initial guesses for the parameters `pRun` and `pDensity` from just the data.

## Initial Guesses

For a just a poisson process, we can just look at the total counts divided by the total lengths in the data set to get an estimator for `pDensity`. However, now the data with zero-counts
aren't necessarily coming from a poisson process. They could simply be the result of the Bernoulli trial deciding to do nothing and return a count of 0. So, as a simple initial guess,
we don't sum over all data points; we just sum over those with non-zero count.

Once we have a guess for `pDensity`, we can use it to make a guess for `pRun`. We take the proportion of data samples with a count of 0, and adjust this proportion to account for Poisson process
producing counts of 0. As a simple initial guess, we simply look at using a poisson rate that is the product of our guess for `pDensity` with the mean length `T`.

``` python
# Now let's find some initial guesses for the poisson density and the probability of running a poisson process.

zeroMask = counts < 1 
densityGuess = counts[~zeroMask].sum() / lengths[~zeroMask].sum()
pRunGuess = (counts > 0).sum() / (1 - np.exp(-densityGuess * lengths.mean()))
pRunGuess = pRunGuess / counts.sum()
print('Initial Poisson Density Guess = ', densityGuess)
print('Initial Probability Running Poisson Guess = ', pRunGuess)
```

We have:

```
Initial Poisson Density Guess =  0.0244753634256
Initial Probability Running Poisson Guess =  0.92165214327
```

Compared to the true values of `pDensity = 0.015` and `pRun` = 0.75`, we can see that these initial estimates aren't relatively close to the true values. Now let's look at improving them with our regression.

## Doing the Regression

We will construct a class `RandomlyRunPoisson` for doing the regression on our data. Later we will go into the exact construction of the class, but for now we want to just discuss using it to do our regression and take a look at the results.

When we initialize an instance of `RandomlyRunPoisson`, we feed it our initial guesses. However, for some reason, giving an initial guess of `pRun` that is significantly higher than the true value causes
the regression algorithm to diverge to garbage values. So it is safer to give an estimate that is LOWER than the true values of `pRun`. So we will intentionally feed it a value that we know suspect is much smaller.

``` python
# For some reason, model tends to diverge if the guess for the probability of running the poisson process
# is higher than the actual value. So it seems safer to just put in a guess that is lower.

model = RandomlyRunPoisson(pDensityGuess = densityGuess, pRunGuess = pRunGuess / 2)
```

Now, when we run the algorithm to fit the regression, it will iterate over values it finds to approximate the true values. The output of our fit will be the history of values it has found.
So we will be able to view this history and get an idea of the convergence.

``` python
# Now fit the model to the data. Using 50 steps seems to work well enough.

results = model.fit(lengths, counts, nSteps = 50)
print('Final Poisson Density = ', results[-1, 0])
print('Final Probability of Running Poisson = ', results[-1, 1])
```

We get that:
```
Final Poisson Density =  0.0145516689881
Final Probability of Running Poisson =  0.75413365449
```

Now, let's take a look at the graphs of the parameters over the history of the iterations.

``` python
# Now let's plot how the parameters changed over time during the fitting of the model.

plt.plot(results.T[0])
plt.title('Iterations of Fitted Poisson Density')
plt.gca().set_xlabel('Step')
plt.gca().set_ylabel('Fitted Poisson Density')
plt.savefig('2018-02-21-graphs/density.svg')
plt.show()

plt.plot(results.T[1])
plt.title('Iterations of Fitted Probability of Running Poisson')
plt.gca().set_xlabel('Step')
plt.gca().set_ylabel('Fitted Probability of Running Poisson')
plt.savefig('2018-02-21-graphs/probPoisson.svg')
plt.show()
```

The graphs are:

![Graph of Fitted pRun over History of Iterations]({{site . url}}/assets/2018-02-21-graphs/probPoisson.svg)

![Graph of Fitted pDensity over History of Iterations]({{site . url}}/assets/2018-02-21-graphs/density.svg)

Next, let's take a look of the details of the class `RandomlyRunPoisson`.

## Detail of Class `RandomlyRunPoisson`

An important part of understanding the implementation details is understanding the calculations that go into finding the maximum of the log-likelihoods.4
I don't have MathJax enabled on this post, so the only nice way of discussing these calculations is to put them in a LaTeX file and create a pdf file. The 
[pdf file of the calculations is here]({{site . url}}/assets/2018-02-21-calculations.pdf) and the [LaTex source is here]({{site .url}}/assets/2018-02-21-calculations.tex);
the calculations require understanding vector calculus and some basics on log-likelihood. 

Now, for the class `RandomlyRunPoisson`, we make some user facing functions for finding the log-likelihood, its gradient, and its hessian given the current approximate values and the data set.
These public facing methods are not meant to be efficient. They are mostly for debugging purposes or for presenting results. 

The internal process of fitting the regression is done in such a way as to reduce the need for redundant calculations. This involves recording information for each data point in arrays as the fitting
process is done. However the references to these arrays are freed when the fitting process is done. So after the fitting is done, the instance of the class should not be holding onto large sections of
memory.

Now, for the actual details:
``` python
class RandomlyRunPoisson:
    '''
    Defines an instance of a model that given a non-negative length of time T, it randomly finds a
    non-negative count in two steps:
    
    1. Use a Bernoulli trial to decide whether or not to run a Poisson process. If a Poisson process is NOT
        run, then give a count of 0. If it is run then proceed to step 2. We denote the probability of
        running a Poisson process by pRun.
    2. Now run a Poisson process to get a count where the Poisson rate is coming from a fixed Poisson density
       times the length of time T. We denote the density by pDenisty. Note that it is possible for this
       Poisson process to also give a count of 0. 

    So our model has two parameters: pRuna and pDensity. This class allows you to fit this model to data of
    time lengths and counts by maximizing the log-likelihood of the parameters given the data. This is 
    accomplished using the Newton algorithm to find where the gradient of the log-likelihood vanishes. So
    you should provide initial guesses for the algorithm to search from.

    For some reason, the algorithm seems to diverge if you provide a guess for pRun that is too large.
    So to be safe, try a guess of pRun that you are confident is SMALLER than the true value of pRun.

    Member Variables
    ----------------
    pDensity : Float
        The density of the poisson rate. For a sample of length of time T, the poisson rate should be
        T * self.pDensity.
    pRun : Float
        The probability of deciding to run a Poisson process when we first run our Bernoulli trial.
    '''
    
    def __init__(self, pDensityGuess = 1.0, pRunGuess = 0.5):

        '''
        Initializer. Gives the model the initial guess of the poisson denisty pDensity and the probability
        of running a Poisson process pRun.

        Parameters
        ----------
        pDensityGuess : float
            The initial guess for the Poisson denisty of the model.     
        pRunGuess : float
            The initial guess for the probability of running a Poisson process. For some reason,
            the fitting process seems to diverge if the guess is too much higher than the true
            probability. So to be safe, always guess a value that is LOWER than the true value.
        '''

        # The parameters of the model.

        self.pDensity = pDensityGuess 
        self.pRun = pRunGuess 

        # These are used between functions when performing calculations for fitting. Using these
        # as member variables is to reduce redundant calculations and to reduce the number
        # of parameters in function calls.

        self.expTerms = None
        self.denoms = None
        self.denoms2 = None
        self.numNonZero = None

    def _getLL(self, lengths, counts):
        '''
        Internal function for getting the log-likelihood for the current parameter values 
        and some given data. This function expects that the lengths and counts data has
        already been grouped into those for non-zero counts and those for zero-counts.
        
        Parameters
        ----------
        lengths : Dictionary of Arrays
            The key 'zero' should hold the lengths associated to data points where counts are 0.
            The key 'non-Zero' should hold the lengths associated to data points where counts are non-zero.
        counts : Dictionary of Arrays
            The key 'non-Zero' should hold the counts data for data points which have non-zero count.

        Returns
        -------
        Float
            The log-likelihood associated to the data and the current values of the parameters pDensity and
            pRun.
        '''

        expTerms = np.exp(-self.pDensity * lengths['zero'])
        zeroLL = np.log(1 - self.pRun + self.pRun * expTerms).sum()
        nonZeroTerms = ( np.log(self.pRun) + counts['nonZero'] * np.log(self.pDensity) 
                       + counts['nonZero'] * np.log(lengths['nonZero']) 
                       - self.pDensity * lengths['nonZero'])
        ll = zeroLL + nonZeroTerms.sum()

        return ll
       
    def getLL(self, lengths, counts):
        '''
        User facing function for getting log-likelihood. This method isn't efficient for repeated calls to the same data set, but it can
        be used to get values if necessary.

        Parameters
        ----------
        lengths : Array of Floats
            The lengths of time that is the independent variable of the data set.
        counts : Array of Int
            The counts that is the dependent random variable of the data set.

        Returns
        -------
        Float
            The log-likelihood of the current values of the parameters for a given data set.
        '''

        zeroMask = counts == 0
        groupedLengths = {'zero' : lengths[zeroMask],
                          'nonZero' : lengths[~zeroMask]
                         }
        groupedCounts = { 'nonZero' : counts[~zeroMask] }

        ll = self._getLL(groupedLengths, groupedCounts)

        return ll

    def _getLLGrad(self, lengths, counts): 
        '''
        A private function for getting the gradient of the log-likelihood for the current values of the parameters and the given data set. The gradient is a 2d vector and of the form
        [dLL / dpDensity, dLL / dpRun].
    
        This function requires that self.denoms and self.expTerms have already been computed.

        Parameters
        ----------
        lengths : Dictionary of Arrays of Floats
            For the key 'zero', lengths['zero'] should be the lengths in the data set for points where the counts are zero.
            For the key 'nonZero', lengths['nonZero'] should be the lengths in the data set for points where the counts are non-zero. 

        counts : Dictionary of Int
            For the key 'nonZero', counts['nonZero'] should be the counts for points in the data set where the counts are non-zero.

        ''' 

        # First calculate dLL / dpDensity.

        terms = lengths['zero'] * self.expTerms * self.denoms 
        dpDensity = -self.pRun * terms.sum() + 1 / self.pDensity * counts['nonZero'].sum() - lengths['nonZero'].sum()
   
        # Now calculation dLL / dpRun. 

        terms = - (1 - self.expTerms) * self.denoms
        dPRun = terms.sum() + 1 / self.pRun * self.numNonZero

        # Put them together to make the gradient.    
        grad = np.array([dpDensity, dPRun])

        return grad

    def getLLGrad(self, lengths, counts):
        '''
        Public function for finding the gradient of the log-likelihood. This methods isn't optimized to be used in repetitive function calls as it will 
        make redundant calculations that only depend on the current data set.

        Parameters
        ----------
        lengths : Array of Floats
            Array of lengths of time that represent the independent variable in the model.
        
        counts : Array of Int
            Array of counts that represent the dependent random variable in the model.

        Returns
        -------
        [Float, Float]
            The gradient of the log-likelihood by the parameter for the given data. The gradient is of the form [dLL / dpDenisty, dLL / dpRun].
        '''
        
        zeroMask = counts == 0
        gLengths = {'zero' : lengths[zeroMask],
                    'nonZero' : lengths[~zeroMask]
                   }
        gCounts = { 'nonZero' : counts[~zeroMask] }

        # These terms are necessary to find before calling the private function to get the gradient.

        self.expTerms = np.exp(-self.pDensity * gLengths['zero'])
        self.denoms = 1.0 / (1 - self.pRun + self.pRun *  self.expTerms)
        self.numNonZero = len(gCounts['nonZero']) 

        # Now use the private function to get the gradient.
        
        grad = self._getLLGrad(gLengths, gCounts) 

        # Some garbage collection.

        self.expTerms = None
        self.denoms = None

        return grad

    def _getLLHessian(self, lengths, counts): 
        '''
        Private function for computing the hessian of the log-likelihood by the parameters for the given data. This function reduces redundant
        operations by requiring that self.expTerms, self.denoms, and self.denoms2 be precomputed before being called.


        Parameters
        ----------
        lengths : Dictionary of Arrays of Floats 
            lengths['zero'] should be the lengths of time for the points in the dataset where the count (i.e. dependent random variable) are 0.
            lengths['nonZero'] should similarly be for when the count is non-zero.
       
        counts : Dictionary of Arrays of Int
            counts['nonZero'] should be the counts of the points in the data set for which the count is non-zero. 

        Returns
        -------
        Numpy Array of shape (2,2)
            The is the hessian of the log-likelihood for the current values of self.pDenstiy and self.pRun. The hessian is of the form
            [[ d^2 LL / dpDensity^2, d^2 LL / dpRun dpDensity],
             [ d^2 LL / dpDensity dpRun, d^2 LL / dpRun ^ 2]]. 
        '''

        # Get d^2 LL / dDensity dPRun.

        terms = lengths['zero'] * self.expTerms * self.denoms2
        dDensity_dPRun = - terms.sum()
    
        # Get d^2 LL / dDensity^2.

        terms = terms * lengths['zero'] 
        dDensity_dDensity = ( self.pRun * (1 - self.pRun) * terms.sum() 
                              - 1 / self.pDensity**2 * counts['nonZero'].sum()
                            ) 
       
        # Get d^2 LL / dpRun^2.

        terms = - (1 - self.expTerms)**2 * self.denoms2
        dPRun_dPRun = terms.sum() - 1 / self.pRun**2 * self.numNonZero 

        # Put them together to get the density. Recall that the Hessian is symmetric so we really only need the
        # three calculations above.
    
        hessian = [[dDensity_dDensity, dDensity_dPRun],
                   [dDensity_dPRun, dPRun_dPRun]]
        return np.array(hessian)

    def getLLHessian(self, lengths, counts):
        '''
        Public function for computing the hessian of the log-likelihood for the current values of the model parameters and a given data set.
        This function is not optimized to make repeated calls (as in fitting); it will make redundant calculations that depend only of the data set and not
        the values of the parameters. This function is only meant to check on the value of the hessian and not for fitting.

        Parameters
        ----------
        lengths : Array of Float
            The lengths of time in the data set that make up the independent variable in the model.
        counts : Array of Float
            The counts in the data set that make up the dependent random variable in the model.

        Returns
        -------
        Numpy Array of shape (2,2)
            The is the hessian of the log-likelihood for the current values of self.pDenstiy and self.pRun. The hessian is of the form
            [[ d^2 LL / dpDensity^2, d^2 LL / dpRun dpDensity],
             [ d^2 LL / dpDensity dpRun, d^2 LL / dpRun ^ 2]]. 
        '''
        
        zeroMask = counts == 0
        gLengths = { 'zero' : lengths[zeroMask],
                     'nonZero' : lengths[~zeroMask]
                   }

        gCounts = { 'nonZero' : counts[~zeroMask] } 

        # These terms are necessary before calling the private function to find the hessian.

        self.expTerms = np.exp(-self.pDensity * gLengths['zero'])
        self.denoms = 1.0 / (1.0 - self.pRun + self.pRun * self.expTerms) 
        self.denoms2 = self.denoms**2
        self.numNonZero = len(gLengths['nonZero'])

        # Call the private function.

        hessian = self._getLLHessian(gLengths, gCounts) 

        # Do some garbage collection. No need to keep these around anymore.

        self.expTerms = None
        self.denoms = None
        self.denoms2 = None

        return hessian

    def newtonStep(self, lengths, counts): 
        '''
        Public function for performing the step of the newton algorithm for finding where the gradient of the log-likelihood vanishes.
        This uses the hessian of the log-likelihood to use a linear approximation of the gradient to update an approximation to where
        it vanishes.

        The function will update the values of self.pDensity and self.pRun. 

        Parameters
        ----------
        lengths : Dictionary of Arrays of Floats 
            lengths['zero'] should be the lengths of time for the points in the dataset where the count (i.e. dependent random variable) are 0.
            lengths['nonZero'] should similarly be for when the count is non-zero.
       
        counts : Dictionary of Arrays of Int
            counts['nonZero'] should be the counts of the points in the data set for which the count is non-zero. 
        '''

        # These are necessary to calculate before getting the gradient and hessian. Note that these
        # quantities depend on the current values of the parameters of the model, so we need to 
        # calculate them here.

        self.expTerms = np.exp(-self.pDensity * lengths['zero'])
        self.denoms = 1.0 / (1.0 - self.pRun + self.pRun * self.expTerms) 
        self.denoms2 = self.denoms**2

        llGrad = self._getLLGrad(lengths, counts) 
        hess = self._getLLHessian(lengths, counts) 

        # Find the inverse of the Hessian matrix (easy to do for 2 x 2 matrix).

        hessinv = np.array([[ hess[1, 1], -hess[0, 1]],
                            [ -hess[1, 0], hess[0, 0]]])
        hessinv /= hess[0,0] * hess[1,1] - hess[0,1] * hess[1,0]

        # Find the update and apply it to the parameters.

        update = - np.dot(hessinv, llGrad)
        self.pDensity += update[0]
        self.pRun += update[1]

    def fit(self, lengths, counts, nSteps = 1000):
        '''
        Public function to perform of fit of the model on a data set of lengths of time and counts. Uses the current values of self.pDensity and 
        self.pRun as initial guesses. It seems the fit tends to diverge if the guess of self.pRun is too much larger than its true values. So it
        is safer to use an initial guess of pRun that is SMALLER than the true value. 

        The fit is done by maximizing the log-likelihood of the parameters given the dataset by using the Newton algorithm.

        Parameters
        ----------
        lengths : Array of Float
            The lengths of time of the dataset that are the independent variable of the model.
        counts : Array of Int
            The counts of the dataset that are the dependent random variable of the model.
        nSteps : Int
            How many steps of the Newton algorithm to run inorder to find the maximum of the log-likelihood.

        Returns
        -------
        Numpy Array of Shape (nSteps, 2)
            Array recording the values of the parameters after each step of the Newton algorithm.
        '''

        zeroMask = counts == 0 

        # First group the data into those with zero counts and those with non-zero counts.

        gLengths = { 'zero' : lengths[zeroMask],
                     'nonZero' : lengths[~zeroMask]
                   }
        gCounts = { 'nonZero' : counts[~zeroMask] }
        self.numNonZero = len(gCounts['nonZero'])

        results = [[self.pDensity , self.pRun]]
        for i in range(nSteps):
            self.newtonStep(gLengths, gCounts) 
            results.append([self.pDensity , self.pRun]) 

        # Do some garbage collection.
          
        self.expTerms = None
        self.denoms = None
        self.denoms2  = None
         
        return np.array(results)
```

## [Download the Source Code for this Post]({{site . url}}/assets/2018-02-21-RandomlyChoosingPoisson.py)
## [Download the Notes for Calculations for the Algorithm in This Post]({{site . url}}/assets/2018-02-21-calculations.pdf)
## [Download the LaTeX Source for the Calculation Notes]({{site . url}}/assets/2018-02-21-calculations.tex)

