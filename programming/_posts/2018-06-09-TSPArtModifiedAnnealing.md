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

## [Download the Source Code for This Post]({{site . url}}/assets/2018-06-09-TSPArtModifiedAnnealing.py)
## [GitHub Repo for Upto Date Code](https://github.com/MatthewMcGonagle/TSP_PictureMaker)

