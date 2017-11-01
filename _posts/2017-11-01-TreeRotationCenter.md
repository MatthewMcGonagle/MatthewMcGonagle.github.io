---
layout: post
title: Where is the Center of a Tree Rotation?
date: 2017-11-01
---

In this post we'll look at the location of the "center of rotation" for a binary tree rotation. I find that many of the pictures one finds when looking up information on tree rotations have a very suggestive arrow, but they don't do a very good job of indicating **where** the rotation is happening. For example, the pictures often look like this:

![Usual Pic of Tree Rotation]({{site . url}}/assets/2017-11-01-usual.svg)

Here the triangular nodes can represent either single nodes or entire sub-trees.

I find this picture to be helpful, but I also feel it falls short of pointing out where the rotation is happening. In particular, I like to think of `C` as being the center of rotation, and I find the following picture more clear:

![Pic of C Being Center]({{site . url}}/assets/2017-11-01-improved.svg)

First, we remove the connection between `B` and `C`. Second, we perform the rotation. Last, we add the connection between `C` and `D`. Note this connection is the only one possible to preserve the ordering; note that we can't necessarily connect `A` to `C` as they are both possibly sub-trees. That is, there may be no empty right children inside `A` that can be connected to `C`.
