---
layout: post
title: Understanding Red Black Tree Insertion
date: 2018-01-19
---

In this post we will take a look at understanding insertion in red-black trees on an intuitive level. We will first take an intuitive look at how red-black trees are defined and then look at how to derive the steps to do insertion from scratch. 

## Intuition of the Structure of Red-Black Trees

The purpose of a red-black tree is to create a scheme for balancing a binary tree without doing too much bookkeeping. The main properties of red-black trees that accomplish this are:

* For any path from the root node to any leaf node, the number of black nodes is always the same. That is, the black nodes by themselves are perfectly balanced. Imbalance can only exist in the red nodes.

* No red node may have a red child. This has multiple purposes:
    * This allows the number of red nodes in the tree to be restricted by the number of black nodes in the tree. So the perfectly balanced black nodes can control the number of imbalanced red nodes. 
    * The red nodes aren't perfectly balanced, but they can't be arbitrarily chained either. 

The rest of the properties of red-black trees are mostly bookkeeping statements to help with the analysis of their operation:

* Root node is black.
* Empty leaf nodes are black.

To simplify our diagrams, we won't be concentrating on the empty leaf nodes, and will mostly omit them entirely. They are always there, and so don't add much to the conversation. Now let us move on to discussing node insertion. 

## Initial Insertion of a Node

We wish to insert a node into our binary tree; what color should the node initially upon insertion? Intuitively, the black nodes by themselves are perfectly balanced inside the tree. Let us not mess up that balance. So we initially insert a RED node. Let's call the new node `N`.

Now we are left with two possibilities:

* The parent `P` of node `N` is black; so the red-black tree structure is still good and we are done.

* The parent `P` of node `N` is also red; the red-black structure is damaged and we must fix it. 

So now let us continue with the assumption that we are in the scenario that `P` is also red. Let us label the (possible empty) sibling sub-tree of `N` as `A`. So we are in the following scenario:

![Parent Also Red]({{site . url}}/assets/2018-01-20-pics/redParent.svg)

Note that we have colored the lower part of the sub-tree as red and black as we don't know the color of any of the nodes that occur there (if there are any). However, the root of `A` must be black (if it exists) so it is colored black. We will use a similar convention throughout our discussion.

Okay, so the problem with our red-black tree is that we now have a red node `P` with a red child `N`. Well, this is an easy fix; just color `P` black. We do so, but this creates an imbalance in the black nodes. So we trade one problem with the red-black structure for another. So we are now in the following situation:

![Parent Colored Black]({{site . url}}/assets/2018-01-20-pics/redParent2.svg)

Again, I repeat: the black nodes are now UNBALANCED. The fix for this now depends on the color of the sibling of `P`. We will take a look at this in the following sections.

## `P`'s Sibling is Also Red

For now, forget the color of the inserted node `N`. For this case, the fix does NOT actually depend on its color.

Since `P` was originally red, its parent `G` (i.e. the grandparent of `N`) must be black. We label `P`'s sibling (i.e. `N`'s uncle) as `U`. Since `U` is red, its child sub-trees must have black nodes (if they exist). So we are in the following situation:

![Unbalanced P Sibling Red]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack2B.svg)

Note that the black box and "+1" indicates that the left paths below `G` contain one extra black node. We have also chosen to relabel the sub-trees using `A`, `B`, `C`, and `D`.

The fix here is easy: switch the color of `G` and `U`. This removes a black node from the left paths while keeping the number of black nodes in the right path the same. So we get:

![Switched G and U colors]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack2B2.svg)

Note that we have used "+0" to indicate balance in the black nodes.

However, now we could have yet another problem in the red-black structure, the red color of `G` versus its parent:

* In the case that `G` is the root node, we just change color of `G` to black. This increases the black-height in the
tree by one. 

* In the case that `G` has a black parent then there is no problem. We are done.

* In the case that `G` has a red parent, then recurse up the tree: Change `G`'s parent to black, rebalance the
black nodes, and repeat.

Next, let's consider the case that `U` is black.

## First Step `P`'s Sibling is Black

Now consider the case that `U` is black. Note that we now don't know the color of `U`'s children. For now, forget the color of `N`; let's see if we can fix the imbalance in black nodes without using this information. So we have the following situation:

![P Sibling Also Black]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack3B.svg) 

We can undo the balance in the left paths by changing `G` to red:

![G Now Red]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack3B2.svg)

However, now the right side paths are missing a black node, denoted by "-1". We can try to fix this with a rotation. Before we continue, let's get an idea of how rotations affect black balance.

## Effect of Roations on Balance

Consider the following tree:

![Before Rotation]({{site . url}}/assets/2018-01-20-pics/initialTree.svg)

The two nodes that can affect the balance upon a rotation are marked in green and blue. So let's apply a rotation and see how things change:

![After Rotation]({{site . url}}/assets/2018-01-20-pics/afterRotation.svg)

So we see that

* Paths through sub-tree `A` lose one of the green node.
* Paths through sub-tree `B` (the sub-tree we "rotate" around) don't change their color count.
* Paths through sub-trees `C` and `D` gain one the blue node.
