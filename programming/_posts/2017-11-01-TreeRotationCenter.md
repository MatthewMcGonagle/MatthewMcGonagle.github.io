---
layout: post
title: Where is the Center of a Tree Rotation?
date: 2017-11-01
tags: Theory
---

In this post we'll look at the location of the "center of rotation" for a binary tree rotation. I find that many of the pictures one finds when looking up information on tree rotations have a very suggestive arrow, but they don't do a very good job of indicating **where** the rotation is happening. For example, the pictures often look like this:

![Usual Pic of Tree Rotation]({{site . url}}/assets/2017-11-01-usual.svg)

Here the triangular nodes can represent either single nodes or entire sub-trees.

I find this picture to be helpful, but I also feel it falls short of pointing out where the rotation is happening. In particular, I like to think of `C` as being the center of rotation, and I find the following picture more clear:

![Pic of C Being Center]({{site . url}}/assets/2017-11-01-improved.svg)

First, we remove the connection between `B` and `C`. Second, we perform the rotation. Last, we add the connection between `C` and `D`. Note this connection is the only one possible to preserve the ordering; it is the only place between `B` and `D`. 

Alternatively, one can also imagine `C` to be directly below its grandparent `D`. Now consider the segment connecting `C` to its parent `B`. We then rotate this segment to it connects `C` to its grandparent `D`. Then we rotate the rest of the tree to be in proper order. This way of thinking is pictured below:

![Rotation of Segment First]({{site . url}}/assets/2017-11-01-alternative.svg)

The pictures we have drawn are sort of simplified. When looking at more complicated diagrams, such as those that occur when studying red-black trees, it can become more difficult to see the rotation. In this case, I find the alternative method does a good job of clarifying how the rotation is going to affect the tree.
