---
layout: post
title: 'What is Conformal Geometry?'
date: 2017-11-09
---

In this post we will explore the concept of conformal geoemtry. First we consider a simplified problem. Suppose we are running from location A to location B. We are start by running on grass where we can run fast. However, the grass suddenly changes into slippery mud where we run slow. What is the quickest way to run from A to B? This set up is show in the following figure:

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
