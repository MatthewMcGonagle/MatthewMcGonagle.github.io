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

## [Download the Source Code For This Post]({{site . url}}/assets/2018-04-16-TSPArtWithAnnealing.py)

