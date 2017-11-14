---
layout: post
title: 'What is Conformal Geometry?'
date: 2017-11-09
---

In this post we will explore the concept of conformal geometry. First we consider a simplified problem. Suppose we are running from location A to location B. We are start by running on grass where we can run fast. However, the grass suddenly changes into slippery mud where we run slow. What is the quickest way to run from A to B? This set up is show in the following figure:

![Pic of Simple Problem]({{site . url}}/assets/2017-11-09-SimpleProblem.svg) 

The answer uses the fact that we run faster on the grass; therefore, we should run a longer distance on the grass than on the distance we run on the mud. In fact, there is a point where the grass turns into mud that we should run to, and then turn to run across the mud straight towards B. 

The main point here is that the shortest route (by time) is **NOT** the straight line path between A and B. Our path will turn at an angle when the grass changes to mud.


## Poincare Disk Model

Now we will consider filling a disk with grass and mud in a special way to get what is called the Poincare Disk Model of Hyperbolic Space. The center of the disk will be pure grass. We add more and more mud as we get closer to the edge of the disk. So we get something similar to the following picture:

![Pic of Poincare Disk]({{site . url}}/assets/2017-11-09-PoincareDisk.svg)

To get the Poincare Disk Model, we have to add mud in a very specific way. We won't go into the exact details, but we describe some of the consequences of doing this:

* The edges of the disk are so muddy that we slow down so much that it takes an infinite amount of time to reach the edge. That is, we are slowed down so much that we can never reach the edge.
* The shortest path between two points A and B falls into two categories:
    * The path is a segment of a line passing through the center of the disk.
    * The path is an arc of a circle that intersects the edge of the disk at 90 degree agles.

Let's take a look at the possibilities for the shortest paths. We draw the shortest path between A and B in blue. The line or circle it sits on is drawn as dotted red. The following is a picture for those that are line segments:

![Pic of Line Segment Geodesics]({{site . url}}/assets/2017-11-09-geodesicline.svg)

The following is a picture for those that are arcs of circles:

![Pic of Circular Arc Geodesics]({{site . url}}/assets/2017-11-09-geodesics.svg)

Note that such a circular path moves inward towards the center where you can run faster on the grass. You can't spend too much time on the grass before your wasting time by moving too far in the wrong direction (i.e. not in the direction of B). So then it must then move away from the center and back towards B. 

The following is a picture of many examples of lines and circular arcs that the shortest paths can sit on. Note that it is possible for these to intersect.

![Pic of Many Geodesics]({{site . url}}/assets/2017-11-09-ManyGeodesics.svg)

The study of the shortest time paths in the Poincare Disk Model is called **hyperbolic geometry**. It is a very specific example of conformal geometry.

## The Conformal Distance

Geometers like to think in terms of more geometric terms instead of things like time. We have been looking at paths of shortest time between points A and B. Geometers like to think of this time as a measure of distance, which we will call the **conformal distance**. So for any path between A and B, the conformal distance is simply the time it takes to go along the curve from A to B (taking into account the effects of mud). 

So we see that the paths of shortest time are also the paths of shortest conformal distance. Such paths are called **geodesics**.

## Conformal Angles

Now we wish to talk about the angle between two geodesic curves using the conformal distance. First, let's discuss how to measure the angle between two lines in the ordinary plane using only distance. Let the lines intersect at a point P. We pick a point A on one line and then a point B on the other; Then we draw a line between A and B. This is pictured below. Our original lines are red and the added line is blue.

![Pic of Euclidean Angle]({{site . url}}/assets/2017-11-09-euclidangle.svg)

Now find the distance between P and A, the distance between P and B, and the distance between A and B. We may then use a tool from trigonometry called the Cosine Formula (or Cosine Rule) to find the angle between our two original red lines.

Now, let's consider two intersecting geodesics in the Poincare Disk. We can pull a similar trick for two intersecting geodesics. Consider two geodesics intersecting in a point P, drawn as red in the diagram below. We can draw a geodesic connecting a point on one line connecting it to a point on the other line. This is marked in blue. We can then measure the distances similar to the Euclidean Case. However, the catch is that the Cosine Formula isn't quite correct.

What we need to do, is repeat making this connecting lines for points on our geodesics closer to the point P. As we get closer to P, the value we find using the Cosine Formula gets closer to the correct value. 

![Pic of Conformal Angle]({{site . url}}/assets/2017-11-09-conformalangle.svg)

## The Conformal Angle is the Regular Angle

It turns out that the conformal angle is the exact same thing as the angle Euclidean angle between the two geodesic curves considered as either lines or circular arcs sitting in the ordinary plane. This is not true for more general geometries on the plane. In fact, this is exactly why the term **conformal** is used to describe the geometries of running through mud that we have been discussing. The word conformal is related to preserving angles.

It turns out that conformal geometry in general (at least from the Riemannian point of view) is related to covering more general shapes in mud, and the geometry is the study of the resulting paths of shortest travel time between points.

## Special Conformal Geometries

For two-dimensions, there are three special geometries:
* The plane itself.
* The Poincare Disk Model.
* The Stereographic Projection of the Sphere. We didn't discuss this, but we plan to in a future post.

So what makes these special? These are the only 2D geometries having no holes and having constant curvature. Both of these are concepts that we plan to discuss in future posts. Let us just comment that the sphere has constant positive curvature, the plane has zero curvature, and the Poincare Disk has constant negative curvature.

