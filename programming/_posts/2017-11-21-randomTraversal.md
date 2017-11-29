---
layout : post
title : Random Traversal of A Binary Tree
date : 2017-11-21
---

## [Download the Source Code for this Post]({{site . url}}/assets/2017-11-21-randomTraversal.py)

In this post, we will look at the probability distribution for the order of the nodes in a binary tree when the tree is traversed in a random order. If `nNodes` is the number of nodes in the tree, then as we are processing the nodes, we are marking them in the order they are processed. So the nodes are marked from 0 to `nNodes - 1`. We will be investigating the probability distribution for these marking for each node in the tree. We will derive a theoretical description of this formula, and we will verify it using numerical simulations in `python`. 

How is the traversal random? As we visit each node, we randomly choose between the three following methods of traversal: 
* preorder traversal - process this node, then the left sub-tree, and then the righ sub-tree.
* inorder traversal - process the left sub-tree, then this node, and then the right sub-tree. 
* postorder traversal - process the left sub-tree, then the right sub-tree, and then this node.

To make sure this is clear, let's consider an example. First consider the following tree where we have labeled the nodes using the letters `A` to `G`.

![Example tree, not processed yet]({{site . url}}/assets/2017-11-21-basicExample.svg)

Let us do a random traversal of this tree. The random traversal starts at the root, node `A`. Suppose as we randomly traverse the tree, we get the following results (a picture of the results is below):
1. `A` : We roll Postorder Traversal. Don't mark the root yet. First process the left and right sub-trees.
2. `B` : We roll Preorder Traversal. So we mark node `B` as the first visited as `0`. Then process sub-trees.
3. `D` : We roll Inorder Traversal, however it doesn't matter. At the last level of the tree, the roll doesn't affect anything. Mark node `D` as `1`.
4. `E` : We roll Preorder Traversal, but it doesn't matter. Mark node `E` as `2`.
5. `C` : We roll Inorder Traversal. First process left-subtree.
6. `F` : We roll Postorder Traversal, but it doesn't matter. Mark node `F` as `3`.
7. `C` : We now return to node `C` and mark it as `4`.
8. `G` : We roll Postorder Traversal, but it doesn't matter. Mark as node `G` as `5`.
9. `A` : Finally we return to node `A` since we are done processing its sub-trees. We mark node `A` as `6`.

![Random Traversal Results]({{site . url}}/assets/2017-11-21-basicExample2.svg)

We will consider a binary tree with a given number of levels `nLevels`. Every node on these levels will exist (i.e. the tree will not be missing any nodes). This is for simplicity, but our theoretical conversation can be adapted to the case where nodes are missing.

We will be investigating the probability distribution of these order markings of the nodes when the traversal probabilities at each node are uniform (so 1/3), and the choices at each node are independent.

## Theoretical Formula

Let's consider the theoretical formula for the probability distribution of a given node. Let's consider an example; let's work with a tree that has five levels and a particular node in the tree. We will consider the distribution of node `D` in level `3` (recall that the root is at level `0`) as marked in the diagram below:

![Diagram of Example]({{site . url}}/assets/2017-11-21-choices.svg)

The probability distribution of the order markings depend on the following:
1. The number of ancestors of node `D` that have node `D` in their right sub-tree. For our example, these are nodes `A` and `C`. The right edges below them are marked `red` to indicate so.
2. For the above nodes in item 1, the size of their left sub-trees. As marked in the diagram, we see that node `A` has a left sub-tree of size 15 nodes. The node `C` has a left sub-tree of 3 nodes.
3. The number of ancestors of node `D` that have node `D` in their left sub-tree. For our example, this is only node `B` marked in our diagram. The left edge below `B` is marked `blue` to indicate this.
4. The sizes of the left sub-tree and right sub-tree of node `D` (the node we are computing the distribution for). We have marked the left and right edges below node `D` as `purple`. We see that both of these sub-trees of `D` have only 1 node each.

To understand the distribution at node `D`, there are a couple things that need to be understood:
* The nodes in the left sub-trees of the ancestors described in item 2 will always be processed before node `D`. This is due to the fact that no matter which choice of traversals in the ancestors are made, the left-subtree nodes will be processed before the right sub-tree nodes. The right sub-tree nodes of these ancestors always include node `D`.
* Similarly, all of the nodes in the right sub-trees of the ancestors described in item 3 will always be processed after node `D`. This is due to the fact that for such ancestors, node `D` is always in their left sub-tree.
* Every ancestor described in item 1 has the same probability of being processed before node `D`. Since node `D` is in their right sub-trees, these ancestors are processed before `D` if these ancestors roll a Preorder Traversal or an Inorder Traversal. So they have a `2/3` probability of being processed before `D`.
* Every ancestor described in item 3 has the same probabilty of being processed before node `D`. Since node `D` is in their left sub-tree, these ancestors are processed before `D` only if these ancestors roll a Preorder Traversal. So they have a `1/3` probability of being processed before `D`.
