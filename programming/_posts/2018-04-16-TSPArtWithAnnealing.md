---
layout: post
title: Travelling Salesman Art With Simulated Annealing
date : 2018-04-15
---

## [Download the Source Code For This Post]({{site . url}}/assets/2018-04-16-TSPArtWithAnnealing.py)

In this post we will explore using solutions to the travelling salesman problem to create images that are drawn with a single closed curve; for a 
true solution to the travelling salesman problem, this curve will not cross itself. However, we will be using the method of simulated annealing to
find approximate solutions to the travelling salesman problem. So our curves will have some self-crossings, but the number of crossings
shouldn't be too large. 

This post is inspired by the travelling salesman art created by others, such as the art at [Professor Bosch's webpage](http://www2.oberlin.edu/math/faculty/bosch/tspart-page.html).
For example, the following applying our method to a simple tracing of a portrait of John Von Neumann:

![Final Curve for Von Neumann]({{site . url}}/assets/2018-04-16-pics/vonNeumannFinal.png)

The overview of our method is two steps:
1. Use random sampling to sample pixels where the probability density is within a normalization factor given by the intensity of each pixel. This sampling can be accomplished
with simple rejection sampling. The samples we obtain will be the vertices of our travelling salesman problem.
2. Try to find approximate solutions to the travelling salesman problem for our vertices. That is, we try to find a closed tour of our vertices of minimal length. 
We implement both of these steps in the classes `RejectionSampler` and `AnnealingTSP`.

First, we will give a small discussion of the travelling salesman problem and then simulated annealing for the travelling salesman problem. A nice discussion of applying
simulated annealing to the travelling salesman problem is given in the article [The Evolution of Markov Chain Monte Carlo Methods](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.295.4478&rep=rep1&type=pdf) by Matthew Richey. 

Then we will show examples of using the classes `RejectionSampler` and `AnnealingTSP`. Finally we will discuss the implementation details of these classes.

Before we get started, let's look at the modules we will need to import for our code:
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Necessary for opening images inorder to sample their pixels.
import io # Used for itermediary when saving pyplot figures as PNG files so that we can use PIL
          # to reduce the color palette; this reduces file size.
```

Also let's set up a seed to `numpy.random` in order to get consistent results:
``` python

# Set up a random seed for consistent results.

np.random.seed(20180327)
```

## The Travelling Salesman Problem

## Simulated Annealing for the Travelling Salesman Problem

## Example of Points on a Circle

For this example, we won't be using rejection sampling as we aren't working with a picture. This example functions more as a double check that everything is working properly.

We create vertices on the unit circle such that their order bounces back and forth across the origin. The easiest way to do this is to use some properties of modulo arithmetic
for prime numbers. I think it will be too much of a distraction to discuss the details of this since we are really only interested running our simulated annealing once
we have the points set up.
``` python
# Look at travelling salesman problem for points on a circle.

# Here are the parameter settings that work nice for our circle example.
 
nVert = 103 # This example works best if you use a prime number of vertices. This
            # is to make sure the intial path bounces around a lot. It has nothing
            # to do with the simulated annealing itself.

# Now get a set of angles on the circle that in order tend to bounce around the circle a lot.

angles = np.arange(0, nVert, dtype = 'int')
angles = np.mod(angles * int(nVert/2), nVert) * 2 * np.pi / nVert
vertices = np.stack([np.cos(angles), np.sin(angles)], axis = -1) 
```

We'll take a look at what this looks like after we set up our simulated annealing iterator:
``` python
# Now set up parameters for simulated annealing.

initTemp = 10**3 / np.sqrt(nVert) # The initial temperature.
nSteps = 10**5  # Total number of steps to run annealing for circle example.
decimalCool = 4 # The number of decimal places to cool the temperature
cooling = np.exp(-np.log(10) * decimalCool / nSteps)  # The calculated cooling factor 
                # based on the number of decimal places to cool.
nPrint = 75  # The number of runs of the annealing process to break the entire process 
                # into. Used for printing out progress.

# Set up our annealing steps iterator.

annealingSteps = AnnealingTSP(nSteps / nPrint, vertices, initTemp, cooling)
```

Let's now get some information on the initial energy and see what the initial cycle looks like:
``` python
# Get some intial information.

initEnergy = annealingSteps.energy
print('initEnergy = ', initEnergy)
print('initTemp = ', initTemp) 


# Plot the intial cycle.

cycle = annealingSteps.getPath()
plotCycle(cycle, 'Initial Path for Circle')
savePNG('2018-04-16-pics\\circleInitial.png')
plt.show()
```

Here is the graph of the initial cycle:

![Initial cycle for circle example.]({{site.url}}/assets/2018-04-16-pics/circleInitial.png)

Now let's run the simulated annealing iterator. We will record the energy for `nPrint` number of times equally spaced over our iterations.
``` python
# Now run the annealing steps for the circle example.

energies = []

for i in range(nPrint):
    print('Training run ', i)
    print('Pre energy = ', annealingSteps.getEnergy())
   
    annealingSteps.doWarmRestart() 
    for e in annealingSteps:
        continue

    energies.append(annealingSteps.getEnergy())

energies = np.array(energies)

print('Finished running annealing for circle example')

plotEnergies(energies, 'Energies for Circle Example')
savePNG('2018-04-16-pics\\circleEnergies.png')
plt.show()
```
The graph of the energies is:

![Energies for Annealing of Circle]({{site.url}}/assets/2018-04-16-pics/circleEnergies.png)

Now, let's get the final cycle:
``` python
# Plot the final cycle of the annealing process.

cycle = annealingSteps.getPath()
plotCycle(cycle, 'Final Path for Circle')
savePNG('2018-04-16-pics\\circleFinal.png')
plt.show()
```
The graph of the final cycle is:

![Final cycle for circle example]({{site . url}}/assets/2018-04-16-pics/circleFinal.png)


## Example For Letters TSP

Now we look at an example where we use our method for the simple image of the letters "TSP", standing for "Travelling Salesman Problem". Here is the image:

![Image of tsp]({{site.url}}/assets/2018-04-16-pics/tsp.png)

Note, we have chosen a black background as white is represented with the value 255 which gives white a larger intensity. Therefore, the color white results in
more sampling while the color black should result in next to no samples. We intentionally do this to keep the number of vertices small.

First, let's set up some parameters that we will need:
``` python
# Parameters

nVert = 600
initTemp = 2 / np.sqrt(nVert)
nSteps = 1 * 10**6 
decimalCool = 2.0
cooling = np.exp(-np.log(10) * decimalCool / nSteps) 
```

Now, let's open the image and sample the pixels. We make use of a function `sampleImagePixels` that we have defined. We will
go into the implementation details of this function later. For now, it is important to know that the random samples are reordered
to be intially traversed in an up-down and left-right manner (see the graph of the intial cycle below); this is done as an attempt to 
give our iterator a "good" inital guess. 

``` python
# Open file tsp.png and convert to numpy array.

image = Image.open('2018-04-16-pics\\tsp.png').convert('L')
vertices = sampleImagePixels(image, nVert)
```

Next, let's get a scatter plot of the samples to see what they look like:
``` python
# Do a scatter plot of the points sampled.

plotSamples(vertices, 'Sampled Points for tsp.png')
savePNG('2018-04-16-pics\\tspSamples.png')
plt.show()
```
Here is the graph:

![Scatter plot of samples for TSP example.]({{site.url}}/assets/2018-04-16-pics/tspSamples.png)

Let's set up our annealing iterator and plot the intial cycle.
``` python
# Set up our annealing steps iterator.

annealingSteps = AnnealingTSP(nSteps / nPrint, vertices, initTemp, cooling)
initEnergy = annealingSteps.energy
print('initEnergy = ', initEnergy)
print('initTemp = ', initTemp) 

# Plot the intial cycle.

cycle = annealingSteps.getPath()
plotCycle(cycle, 'Initial Path for tsp.png', doScatter = False)
savePNG('2018-04-16-pics\\tspInitial.png')
plt.show()
```

The plot of the initial cycle is:

![Plot of Initial Cycle for TSP]({{site.url}}/assets/2018-04-16-pics/tspInitial.png)

Now run the annealing iterator and graph the recorded energies.
``` python
# Now run the annealing steps for tsp.png example.

energies = []

for i in range(nPrint):
    print('Training run ', i)
    print('Pre energy = ', annealingSteps.getEnergy())
   
    annealingSteps.doWarmRestart() 
    for e in annealingSteps:
        continue

    energies.append(annealingSteps.getEnergy())

energies = np.array(energies)

print('Finished running annealing for tsp.png example')

# Plot the energies of the annealing process over time.

plotEnergies(energies, 'Energies for tsp.png Samples')
savePNG('2018-04-16-pics\\tspEnergies.png')
plt.show()
```
The graph of the recorded energies is:

![Energies for TSP example.]({{site.url}}/assets/2018-04-16-pics/tspEnergies.png)

Finally, let's plot the final cycle:
``` python
# Plot the final cycle of the annealing process.

cycle = annealingSteps.getPath()
plotCycle(cycle, 'Final Path for tsp.png', doScatter = False)
savePNG('2018-04-16-pics\\tspFinal.png')
plt.show()
```
The graph of the final cycle is:

![Final cycle for TSP example.]({{site.url}}/assets/2018-04-16-pics/tspFinal.png)

## [Download the Source Code For This Post]({{site . url}}/assets/2018-04-16-TSPArtWithAnnealing.py)

