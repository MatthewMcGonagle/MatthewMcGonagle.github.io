---
layout: post
title: Understanding Red Black Tree Insertion
date: 2018-01-19
tags: Theory
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

So now let us continue with the assumption that we are in the scenario that `P` is also red. Let us label the (possible empty) sibling sub-tree of `N` as `Z` (sub-trees on pictures of the level of `P` and `N` will always be labeled with `X`, `Y`, and `Z`). So we are in the following scenario:

![Parent Also Red]({{site . url}}/assets/2018-01-20-pics/redParent.svg)

Note that we have colored the lower part of the sub-tree as red and black as we don't know the color of any of the nodes that occur there (if there are any). However, the root of `Z` must be black (if it exists) so it is colored black. We will use a similar convention throughout our discussion.

Okay, so the problem with our red-black tree is that we now have a red node `P` with a red child `N`. Well, this is an easy fix; just color `P` black. We do so, but this creates an imbalance in the black nodes. So we trade one problem with the red-black structure for another. So we are now in the following situation:

![Parent Colored Black]({{site . url}}/assets/2018-01-20-pics/redParent2.svg)

Again, I repeat: the black nodes are now UNBALANCED. The fix for this now depends on the color of the sibling of `P`. We will take a look at this in the following sections.

## `P`'s Sibling is Also Red

For now, forget the color of the inserted node `N`. For this case, the fix does NOT actually depend on its color.

Since `P` was originally red, its parent `G` (i.e. the grandparent of `N`) must be black. We label `P`'s sibling (i.e. `N`'s uncle) as `U`. Since `U` is red, its child sub-trees must have black nodes (if they exist). So we are in the following situation:

![Unbalanced P Sibling Red]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack2B.svg)

Note that the black box and "+1" indicates that the left paths below `G` contain one extra black node. We have also chosen to label the sub-trees on the level of `P` and `G` using `A`, `B`, `C`, and `D`.

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

Now consider the case that `U` is black. Also, assume that `P` is the left child of `G`; we will assume this for the rest of post. For the other case, see the brief discussion in the last section at the bottom of this post.

Note that we now don't know the color of `U`'s children. For now, forget the color of `N`; let's see if we can fix the imbalance in black nodes without using this information. So we have the following situation:

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
* Paths through sub-trees `C` and `D` gain one of the blue node.

Next, let's try using a rotation as the next step in fixing the unbalance for the case of `P`'s sibling being a black node.

## Second Step for `P`'s Sibling is Black

In our last step, our last action was to change `G` to a red node, resulting in the right child paths through `G` missing one black node. Paths that go through sub-tree `B` are already balanced; so sub-tree `B` is a good candidate for a rotation. Since `G` is red, the rotation doesn't affect the black balance of paths going through sub-tree `A`. 
The rightward paths going through node `U` will then pick up the black node they are missing. So we get:

![Rotated P's Sibling Black]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack3B3.svg)

Now all of the black nodes are balanced, BUT the left child of `G` could potentially create a problem with the red-black property that no red node should have a red child. This is highlighted with the magenta connection in the diagram. 
So we need to know more information about the root of sub-tree `B` in order to make sure that we fix the red-black property; we need to make sure this root node is black (if it exists). So our recursion/inductive step needs to be a little bit stronger than simply inserting a red node. Let's take a look at the correct version in the next section. 

## True Recursion/Inductive Step

For our inductive step, we need to make sure we know information about the children of `N`. So we assume that:

* We are inserting a red node `N`.
* The child sub-trees of `N` are red-black trees (allow the possibility of empty trees). In particular, the children of `N` are black (when they exist). 
* The black height of `N`'s children matches the black height of `N`'s new sibling (i.e. `P`'s other child).

That is, we are in the following situation:

![True Recursion]({{site . url}}/assets/2018-01-20-pics/trueRecursion.svg)

Now, the cases we succesfully resolved before will be successfully resovled again; these don't take much effort to recheck, so we won't discuss them again. Let's return to the rest of the case where `P`'s sibling is a black node.

## Finishing Case of `P`'s Sibling is Black

Now, we return to the beginning of the case of `P`'s sibling `U` being black. We have two cases to consider.

### Node `N` is a Left Child of `P`

Consider the case that node `N` is a left child of `P`. In that case `P`'s right child is black (if it exists). After changing `P` to black we are in the following situation:

![Solvable One Rotation]({{site . url}}/assets/2018-01-20-pics/unbalancedBlack3BSolvable.svg)

Recall that the next two steps it to change `G` to red and then rotate. When considering the result of our rotation, we have that the child-subtree `B` of `G` has a black root. Therefore, the red-black properties are fixed and we are done.

### Node `N` is a Right Child of `P`

Consider the case that the node `N` is a right child of `P`. So we are in the following situation:

![N is Right Child]({{site . url}}/assets/2018-01-20-pics/wrongSide3B.svg)

In this case, if we continue as before, then `N` is the root node of the sub-tree `B` underneath `G` after the rotation. Then the red node `G` will have a red child, which violates one of the red-black properties.

So, before we proceed with the steps we used before, we need to somehow put a black node where `N` is currently.
We can do this without screwing up the black balance further by using a rotation about sub-tree `B`. After doing so we get:

![After Rotation]({{site . url}}/assets/2018-01-20-pics/wrongSide3B2.svg)

Now the node `N` is the red parent of the red node `P`, but aside from this role reversal, we are in the previous case we considered before. So we color node `N` to get:

![Black N on Top]({{site . url}}/assets/2018-01-20-pics/wrongSide3B3.svg)

Then we proceed as before by coloring `G` to be red and then performing a rotation to fix the black balance. The red-black properties will be fixed, and we are done.

## Other Relative Positions 

By symmetry we can consider the other possiblities for the positioning of the relative positions of `N`, `P`, and `G`. The important thing to note is that we need two rotations when `N` is inserted on the "inner most" possibility.
