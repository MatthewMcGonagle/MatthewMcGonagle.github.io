---
layout: post
date: 2018-06-09
title: Improved TSP Art With Modified Annealing
---

## [Download the Source Code for This Post]({{site . url}}/assets/2018-06-09-TSPArtModifiedAnnealing.py)
## [GitHub Repo for Upto Date Code](https://github.com/MatthewMcGonagle/TSP_PictureMaker)

In this post we improve the quality of pictures we can reprduce using approximate solutions to the traveling salesman problem by using a combination of:
1. Using dithering to get vertices from a grayscale image. 
2. A greedy guess.
3. A modification of simulated annealing based on a size scale of edge length.
4. A modification of simulated annealing based on k-nearest neighbors.

With these modifications, we see significant improvements in the quality of pictures that we could make in the post
[Travelling Salesman Art With Simulated Annealing]({{site . url}}/blog/2018/04/15/TSPArtWithAnnealing). As an example,
we take the following picture of tiger head:

![Tiger Head Original]({{site.url}}/assets/2018-06-09-graphs/tigerHeadResize.png)

and get the following TSP picture:

![Final TSP cycle for tiger head.]({{site.url}}/assets/2018-06-09-graphs/afterNbrs.png)

The code for this post is much longer than the code for other posts. It is available 
[as one python file]({{site . url}}/assets/2018-06-09-TSPArtModifiedAnnealing.py), but there is
also [a GitHub Repository](https://github.com/MatthewMcGonagle/TSP_PictureMaker) where the code is
more properly broken into modules. The code in the repository is more clear, but it could potentially
be updated in the future. So we include the single file python code to insure that a static version
of the code for this post exists.

Now, due to the length of the code, we won't be discussing all of the code in detail. We will examine 
what our classes are doing at a higher level and showing how to use the classes to make our TSP image.

First, let's do some preliminary imports and set up our numpy random seed.
``` python
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
import io # Used for itermediary when saving pyplot figures as PNG files so that we can use PIL
          # to reduce the color palette; this reduces file size.

# Seed numpy for predictability of results.

np.random.seed(20180425)
```

## Dithering

Dithering takes a grayscale image and attempts to reproduce it as a pure black and white image, i.e. each pixel
is either white or black. To get a better quality representation, as one converts individual pixels from
gray to either white or black, one takes the error of such a conversion and spreads the error to neighboring
unprocessed pixels. We use the classic 
[Floyd-Steinberg dithering algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering) 
for determining how to spread the errors. This is discussed very well in many sources, so we won't
spend any time on the details of how this works; let us move on to how to the class `DitheringMaker` to
achieve this in our example.

Now, one minor detail we need to mention is that, for simplicity of execution, the class `DitheringMaker` will always set the
edge columns and the last row to be all white (which will amount to no vertices at these locations).

Before we use our class `DitheringMaker` we need to import the image and convert it to grayscale.
We define the function `getPixels` to help us convert the picture as a `PIL` `Image` to a `numpy` array. 
We don't discuss the implementation details here as they are fairly straightforward; instead we will
show how to use it to get the grayscale image:
``` python
# Open the image.

image = Image.open('2018-06-09-graphs/tigerHeadResize.png').convert('L')
pixels = getPixels(image, ds = 1)
plt.imshow(pixels, cmap = 'gray')
plt.title('Grayscale Version of Tiger Head')
plt.tight_layout()
savePNG('2018-06-09-graphs/grayscale.png')
plt.show()
```

Here is the grayscale image:

![Grayscale tiger head]({{site . url}}/assets/2018-06-09-graphs/grayscale.png)

To get the dithering of the grayscale, we create the class `DitheringMaker`. Again, we won't go into
its implementations here. Now let us get the dithering of the grayscale image.
``` python
# Get the dithered image.

ditheringMaker = DitheringMaker()
dithering = ditheringMaker.makeDithering(pixels)
plt.imshow(dithering, cmap = 'gray')
plt.title('Dithering of Grayscale')
plt.tight_layout()
savePNG('2018-06-09-graphs/dithering.png')
plt.show()
```
The dithered image is:

![Dithering of grayscale]({{site . url}}/assets/2018-06-09-graphs/dithering.png)

## Getting the Vertices

To get the vertices of our graph from the dithered image, we simply make a vertex for every black pixel. To
do so, we define a function `getVertices`, but we won't go into its implementation details. Instead let us
just use it to get the vertices from our dithered image: 
``` python
vertices = getVertices(dithering)
print('Num vertices = ', len(vertices))
```
We get the following output:
```
Num vertices =  22697
```
So we see that we get alot of vertices!

## Making a Greedy Guess

One caveat of our greedy guess is that our algorithm will drop a vertex if we give it an odd number of vertices;
our aim is to process a large number of vertices, so this single vertex shouldn't affect the final quality much. The
reason for this drop is that our algorithm first pairs up the vertices, and it is easier to just drop the left
out vertex.

Our greedy guess is made in the following manner:
1. Going through the vertices in order, pair them up with the nearest neighbor out of all vertices that
haven't paired with a partner yet. Each pair of partners gives us a curve consisting of just two vertices
(so a line segment).
2. We iterate going through the curves in order, connecting them to another curve that hasn't been connected
in this step yet and such that it will make the smallest connecting length. This connection can be
made from either the beginning endpoint for ending endpoint of the curve.
3. We repeat step 2 until we are left with a single curve passing through every vertex. This curve is 
considered a cycle by simply thinking of there being a connection between the beginning endpoint and the
ending endpoint.

The following pictures illustrate the steps of our greedy guess.

![Original Vertices]({{site . url}}/assets/2018-06-09-graphs/greedyExample0.svg)
![Initial Pairing]({{site . url}}/assets/2018-06-09-graphs/greedyExample1.svg)
![First Pass Step 2]({{site . url}}/assets/2018-06-09-graphs/greedyExample2.svg)
![Second Pass Step 2]({{site . url}}/assets/2018-06-09-graphs/greedyExample3.svg)

The reason that we don't connect a single curve to more than one other curve in step 2 is that we try 
to grow the size of the individual curves uniformly. Now, if at the beginning of step 2 there is an odd number
of curves, then we will necessarily have a difference in curve sizes when we finish. However, this
difference will be minimized (at least heuristically).

The reason we grow the curves uniformly is that it heuristically should make the interiors of the curves close
to correct. The problem will be made by connecting large curves by their endpoints. For example, if instead we
had chosen to grow one single curve, then as we progress, how vertices are are added depend more and more
on the order that previous vertices were added. 

To make our greedy guess, we make a class `GreedyGuesser`. We won't discuss the implementation details of the class,
but we will show how to use it. First, we have other preprocessing that we want to do. To keep the length of
the curve from being too large, we scale the image so that the y-coordinates are between 0 and 1. To do so,
we create the function `normalizeVertices`; its implementation is relatively straight forward so we won't 
discuss it.

Also, after we make our greedy guess, we roll the array to put the endpoints in the interior. Most likely
there is a large gap between these two points, and our algorithms will be better able to handle it
if it is explicitly in the interior of the array.

We put the greedy guess along with the normalization inside the function `preprocessVertices`; it is defined as
``` python
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
```
Now we are ready to run the preprocessing, including the greedy guess:
``` python
print('Preprocessing Vertices')
vertices = preprocessVertices(vertices)
print('Preprocessing Complete')
```

Let's take a look at the cycle created by the greedy guess (we have created a function `plotCycle` to
help with plotting our cycles). 
``` python
# Plot the result of the greedy guess and other preprocessing.

cycle = np.concatenate([vertices, [vertices[0]]], axis = 0)
plotCycle(cycle, 'Greedy Guess Path', doScatter = False, figsize = cycleFigSize) 
plt.tight_layout()
savePNG('2018-06-09-graphs/greedyGuess.png')
plt.show()
```
The greedy guess cycle is:

![Greedy guess cycle]({{site . url}}/assets/2018-06-09-graphs/greedyGuess.png)

As you can see, the greedy guess is far from perfect.

## Annealing Based on Size Scale

![Initial Cycle]({{site . url}}/assets/2018-06-09-graphs/sizeScale0.svg)
![First Vertex Pool]({{site . url}}/assets/2018-06-09-graphs/sizeScale1.svg)
![After Annealing on Pool]({{site . url}}/assets/2018-06-09-graphs/sizeScale2.svg)
![New Vertex Pool]({{site . url}}/assets/2018-06-09-graphs/sizeScale3.svg)

![Cycle After Annealing Based on Size Scale]({{site . url}}/assets/2018-06-09-graphs/afterSizeScale.png)

## Annealing Based on k-Nearest Neighbors

![Cycle After Annealing Based on Neares Neighbors]({{site . url}}/assets/2018-06-09-graphs/afterNbrs.png)

## [Download the Source Code for This Post]({{site . url}}/assets/2018-06-09-TSPArtModifiedAnnealing.py)
## [GitHub Repo for Upto Date Code](https://github.com/MatthewMcGonagle/TSP_PictureMaker)

