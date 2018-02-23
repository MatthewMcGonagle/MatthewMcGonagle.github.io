'''
Do regression for samples from a random process where we first randomly decide whether or not to run a Poisson Process.

For each sample, the decision is made by doing a Bernoulli trial with specified probability runProb.
Each sample has an independent variable giving the length of time T that the Poisson process is run on. 
The poisson rate for a length T is given by a constant Poisson rate density (denoted by pDensity) times T. 
The response (or dependent variable) is a count of observations seen. When there is no Poisson process run,
the count is 0. 

The regression is done by numerically finding the values of runProb and pDensity that maximize the
log-likelihoods associated with the sample data. This is done using formulas for the log-likelihoods
and their first and second order derivatives to run the Newton method to find a critical point of the log-likelihoods.

Theoretical calculations related to how the functions in the program work can be found in the file '2018-02-21-calculations.pdf'. The
LaTeX source for the calculations are in '2018-02-21-calculations.tex'.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(20180223)

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


########################################
#### Main executable ###################
########################################

# Set up parameters for sampling the time lengths and the parameters for the model.

maxLength = 100
pDensity = 1.5 / maxLength  
probDoPoisson = 0.75
nSamples = 5000

print('True poisson density = ', pDensity)
print('True probability of running poisson = ', probDoPoisson)

# Set up sample of length by using a uniform distribution from 0 to maxLength.

lengths = np.random.uniform(0, maxLength, size = nSamples)

# Now set up the samples of the counts. First set up samples of bernoulli trial.
counts = np.random.uniform(size = lengths.shape)

# Now set up whether to include a poisson sample based on probDoPoisson. If not, then just make the count 0.

doPoissonMask = counts < probDoPoisson
counts[~doPoissonMask] = 0
poissons = [np.random.poisson(pDensity * x) for x in lengths]
counts[doPoissonMask] = np.array(poissons)[doPoissonMask]

# Put the samples in a pandas dataframe to make some histograms.

samples = pd.DataFrame(counts, columns = ['counts'])
samples['lengths'] = lengths
print(samples.head())

samples.lengths.hist()
plt.gca().set_title('Histogram of Lengths of Time')
plt.show()

samples.counts.hist()
plt.gca().set_title('Histogram of Counts')
plt.show()

# Now let's find some initial guesses for the poisson density and the probability of running a poisson process.

zeroMask = counts < 1 
densityGuess = counts[~zeroMask].sum() / lengths[~zeroMask].sum()
pRunGuess = (counts > 0).sum() / (1 - np.exp(-densityGuess * lengths.mean()))
pRunGuess = pRunGuess / counts.sum()
print('Initial Poisson Density Guess = ', densityGuess)
print('Initial Probability Running Poisson Guess = ', pRunGuess)

# For some reason, model tends to diverge if the guess for the probability of running the poisson process
# is higher than the actual value. So it seems safer to just put in a guess that is lower.

model = RandomlyRunPoisson(pDensityGuess = densityGuess, pRunGuess = pRunGuess / 2)

# Take a look at what the log-likelihood and its derivaties are for our guess.

ll = model.getLL(lengths, counts)
grad = model.getLLGrad(lengths, counts)
hess = model.getLLHessian(lengths, counts)

print('ll = ', ll)
print('Grad = \n', grad, '\nRelative Grad = \n', grad / ll)
print('Hess = \n', hess)

# Now fit the model to the data. Using 50 steps seems to work well enough.

results = model.fit(lengths, counts, nSteps = 50)
print('Final Poisson Density = ', results[-1, 0])
print('Final Probability of Running Poisson = ', results[-1, 1])

# Now let's plot how the parameters changed over time during the fitting of the model.

plt.plot(results.T[0])
plt.title('Iterations of Fitted Poisson Density')
plt.gca().set_xlabel('Step')
plt.gca().set_ylabel('Fitted Poisson Density')
plt.show()

plt.plot(results.T[1])
plt.title('Iterations of Fitted Probability of Running Poisson')
plt.gca().set_xlabel('Step')
plt.gca().set_ylabel('Fitted Probability of Running Poisson')
plt.show()
