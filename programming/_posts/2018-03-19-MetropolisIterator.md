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
steps = list(walk)
steps = pd.DataFrame(steps, columns = ['position'])

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

true = getDesiredDensity(trueX) * normalization * dBin * len(steps.position) 

# Do a combination of histogram of walk values and true curve.

steps.position.hist(bins = bins)
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

To look at the 
## Code
