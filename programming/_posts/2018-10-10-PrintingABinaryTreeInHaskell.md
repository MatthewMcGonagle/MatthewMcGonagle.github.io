---
layout: post
title: "Printing a Binary Tree Using the State Monad"
date: 2017-10-10
tag: Haskell
---

## [Download the Source Code for This Post]( {{ site . url }}/assets/2017-10-10-code.hs)


We will be discussing how to use the State monad to help print a binary search tree in a top to bottom manner.
That is, we will be printing the root at the top of the tree and the tree grows downward. Consider the binary search tree for the words in the sentence "haskell is a cool language but state monads are hard"; the words are simply inserted in the order they appear in the sentence. Here is a picture of this tree.

![Picture of Example Tree]( {{site . url}}/assets/2017-10-10-ExampleTree.svg)

The following is the output for printing this tree in a top to bottom manner using our functions. 

```
                                haskell
  a                                       is
_                 cool                  _      language
            but          hard                _                     state
      are       _      _      _                           monads         _
    _     _                                             _        _
```
Here, each `_` represents an empty node in our tree.

Let us make a remark about the method we will use in this post. It is possible to do the horizontal printing above using a simple one pass recursion with functions that return `[String]`. In fact, the State monad would be unnecessary in this case. However, we will be using a different method as described below for the following reasons:
* Our method will actually determine the positions of each node in the print out. In the future, we plan on making a post about using this to print out the tree in a `.svg` file. 
* For pedagogical reasons, our method will allow us to demonstrate how to use the `State` monad.

## Our Tree Data Type

First, we will be using the following recursively defined tree data type:

``` haskell
data Tree a = Node a (Tree a) (Tree a) | Empty
```

This data type defines either a node containing a value of type `a` and two children also of type `Tree a`, or it defines an empty node.

For simplicity of exposition, let's just manually define the instance of our tree:

``` haskell
let exampleTree = Node "haskell"
                       (Node "a"
                             Empty
                             (Node "cool"
                                    (Node "but"
                                          (Node "are"
                                                Empty
                                                Empty
                                          ) 
                                          Empty
                                    ) 
                                    (Node "hard"
                                          Empty
                                          Empty
                                    )
                             ) 
                       ) 
                       (Node "is"
                             Empty
                             (Node "language"
                                   Empty
                                   (Node "state"
                                         (Node "monads"
                                               Empty
                                               Empty
                                         ) 
                                         Empty
                                   )
                             )
                       )
```

## Simple Left to Right Printing

For starters and debugging purposes, let's write a function for printing our
tree in a left to right manner. That is, the root will be at the left and the leaf nodes will be on the right. This function will be much more simple than what we will need to print the tree in a top to bottom manner.
To do this, we will traverse the tree in a depth first manner and use the `State` monad to keep track of the indentation level as we go. Note, we write this function for type `Tree a` where `a` is any type for which `show :: a -> String` is defined. 

``` haskell
-- Simple left / right print.

simpleShow :: (Show a) => Tree a -> State Position String 
simpleShow Empty = do
    indent <- get
    let padding = replicate indent ' '
    return $ padding ++ "Empty\n"
simpleShow (Node x lChild rChild) = do
    indent <- get
    let padding = replicate indent ' '
        xString = padding ++ "Node " ++ (show x) ++ "\n"
    put $ indent + 5
    lString <- simpleShow lChild
    put $ indent + 5
    rString <- simpleShow rChild
    return $ xString ++ lString ++ rString

```
Now let's test our simple left to right print function on our example tree to make sure it is constructed properly. We run 
``` haskell
putStrLn . fst $ runState (simpleShow exampleTree) 0 
```
and get output
```
Node "haskell"
     Node "a"
          Empty
          Node "cool"
               Node "but"
                    Node "are"
                         Empty
                         Empty
                    Empty
               Node "hard"
                    Empty
                    Empty
     Node "is"
          Empty
          Node "language"
               Empty
               Node "state"
                    Node "monads"
                         Empty
                         Empty
                    Empty
```

## Strategy for Top to Bottom Printing

Now, our strategy will be to do two passes of traversing the tree in the following order: 

1. We calculate widths of subtrees for each node, going in the order of bottom to top (i.e. from leaves to root). This traversal is depth first.
2. We calculate the position of each node, going in the order of top to bottom (i.e. root to leaves). This traversal is also depth first.

## Calculating Widths of Subtrees

We think of this problem recursively. Imagine that we have a tree of strings (i.e. `Tree String`) that we need to out put. That is, a type `a` deriving `Show`, we have already converted a tree of type `Tree a` to `Tree String`. To make output easy, we will convert empty nodes of type `Tree a` to `Node "_" Empty Empty`. 

For each node, we will keep track of the following data types recording important size information:

``` haskell
type Width = Int
data WidthInfo = WidthInfo { nodeWidth :: Width 
                           , leftWidth ::  Width
                           , rightWidth ::  Width
                           }
```
The field `nodeWidth` records the length of the string at the current node.
 The field `leftWidth` records the width of the array of `String` necessary to print the left child subtree.
 Note, that for aesthetic reasons, we wish to separate the node from the left child subtree by using one space.
 Therefore, `leftWidth` is one more than the width necessary to print the left child subtree.
 The field `rightWidth` is similar, but corresponds to the right child subtree. Please see the figure below.

![Subtree Measurements]( {{ site . url }}/assets/2017-10-10-TreeSubwidth.svg)

Let `WidthInfo widths` store the width information for the current node.
 Note that the entire width necessary to print the current node and its subtrees is `(nodeWidth widths) + (leftWidth widths) + (rightWidth widths)`. Also note that since we have converted empty nodes of type `Tree a` to non-empty nodes, we have for empty nodes that `WidthInfo {nodeWidth = 0, leftWidth = 0, rightWidth = 0}`, however we will simply convert empty nodes to empty nodes. 

For computing the width information at each node, we will do a depth first traversal of the tree, and we will compute the information starting at the leaves of the tree and going up to the root. So to compute the width information, we will use the following function (no need to use the `State` monad yet):

``` haskell
computeWidths :: Tree String -> Tree WidthInfo
computeWidths Empty = Empty
computeWidths (Node str lChild rChild) = Node widths lChild' rChild' 
    where lChild' = computeWidths lChild
          rChild' = computeWidths rChild
          widths = WidthInfo { nodeWidth = length str
                             , leftWidth = lWidth
                             , rightWidth = rWidth
                             }
          lWidth = case lChild' of 
                Empty -> 0
                (Node w _ _) -> 1 + (nodeWidth w) + (leftWidth w) + (rightWidth w)
          rWidth = case rChild' of 
                Empty -> 0  
                (Node w _ _) -> 1 + (nodeWidth w) + (leftWidth w) + (rightWidth w)
```
 
## Calculating the Positions of the Nodes 

After calculating the width information for each node, we can calculate the horizontal position of each node. For this, we will traverse the tree in a depth first traversal and calculate the positions starting from the root, going downward to the leaves. The data type for the position of each node will be an integer:
``` haskell
type Position = Int
```

For this calculation, we will use the `State` monad; the state will keep track of the position of the current node, as assigned by the cacluation from the parent node. For each node, we calculate the positions of the left and right children using the width information of each child. So to compute the positions we will use the following function:

``` haskell
computeNodePositions :: Tree WidthInfo -> State Position (Tree Position)
computeNodePositions Empty = return Empty
computeNodePositions (Node width lChild rChild) = do
    myPos <- get
    let lPos = case lChild of 
            Empty -> myPos
            (Node w _ _) -> myPos - 1 - (rightWidth w) - (nodeWidth w) 
        rPos = case rChild of 
            Empty -> myPos
            (Node w _ _) -> myPos + 1 + (leftWidth w) + (nodeWidth width) 
    put lPos
    lChild' <- computeNodePositions lChild
    put rPos
    rChild' <- computeNodePositions rChild
    return $ Node myPos lChild' rChild'
``` 
Now we combine these positions with the original strings into a single tree. The state in our return value represents the position of the root node.

``` haskell
-- Function to combine two trees.
combine :: Tree a -> Tree b -> Tree (a, b)
combine Empty _ = Empty
combine _ Empty = Empty
combine (Node x lChildx rChildx) (Node y lChildy rChildy) = Node (x, y) lChildxy rChildxy
    where lChildxy = combine lChildx lChildy
          rChildxy = combine rChildx rChildy

-- The position state represents the position of the root node.
computePositions :: Tree String -> State Position (Tree (String, Position))
computePositions x = do
    let widths = computeWidths x 
    pos <- computeNodePositions widths 
    return $ combine x pos
```

## Printing a Tree of Strings and Positions 

For printing a tree of type `Tree (String, Position)` we will use a breadth first traversal of the tree. As we traverse each level of the tree, we store the next level in a list. We use this to convert our tree into a double list, a list of lists of nodes on each level, i.e. of type `[[(String, Position)]]`.

For this reformatting of the tree, we don't really need to use the type of the tree. So we will work with the general tree of type `Tree a`.

``` haskell
-- Accumulator function for folding over each level. State represents the nodes in the next level of the tree.
addNodeToLevel :: [a] -> Tree a -> State [Tree a] [a]
addNodeToLevel acc Empty = return acc
addNodeToLevel acc (Node x lChild rChild) = do
    nextLevel <- get
    put $ rChild : lChild : nextLevel
    return $ x : acc

-- Reformat sublevels of tree.
reformatSubLevels :: [Tree a] -> State [Tree a] [[a]]
reformatSubLevels [] = return []
reformatSubLevels nodes = do
    put [] -- initialize the next level as empty
    thisLevel <- foldM addNodeToLevel [] nodes
    nextLevel <- get
    subLevels <- reformatSubLevels (reverse nextLevel) 
    return $ (reverse thisLevel) : subLevels

-- Reformat tree.
reformatTree :: Tree a -> [[a]]
reformatTree x = fst $ runState (reformatSubLevels [x]) []
```

After we have reformatted our tree into lists of levels of type `[[(String, Position)]]`, we need to convert each level into a string. For this we will need to use the `State` monad to keep track of the last Position.

``` haskell
addNodeString :: String -> (String, Position) -> State Position String
addNodeString acc (str, pos) = do
    lastPos <- get
    let nSpaces = pos - lastPos - 1
        nSpaces' = case nSpaces > 0 of 
            True -> nSpaces 
            False -> 0
        spacing = replicate nSpaces' ' '
    put $ pos + (length str) - 1
    return $ (reverse str) ++ spacing ++ acc 

showLevel :: [(String, Position)] -> String
showLevel nodes = fst $ runState strState (-1) 
    where strState = foldM addNodeString "" nodes
```

So now we can show entire tree where each node holds the string and its horizontal position.

``` haskell
showTreeWPositions :: Tree (String, Position) -> String
showTreeWPositions x = concat levelStrings
    where levelStrings = map (reverse . combineLevel) levels
          levels = reformatTree x
          combineLevel y = (++ "\n") . fst $ runState (foldM addNodeString "" y) (-1) 
```
Finally, before we can work on showing the entire original tree, we first need to convert the original empty nodes to nodes holding strings "_" representing them.

``` haskell
convertEmpty :: Tree String -> Tree String
convertEmpty Empty = Node "_" Empty Empty
convertEmpty (Node x lChild rChild) = Node x lChild' rChild'
    where lChild' = convertEmpty lChild
          rChild' = convertEmpty rChild
```
So, now we can show the entire tree. We compute the position of the root node based on the first node of the tree of width information. 

``` haskell
showTree :: Tree String -> String
showTree x = showTreeWPositions mixedTree
    where mixedTree = combine x' posTree
          posTree = fst $ runState (computeNodePositions widthTree) rootPos 
          rootPos = case widthTree of 
                Empty -> 0
                (Node w _ _) -> (leftWidth w) + 1
          widthTree = computeWidths x'
          x' = convertEmpty x
```

To get the example given at the beginning of this post, we simply run
``` haskell
putStrLn $ showTree exampleTree
```

## [Download the Source Code for This Post]( {{ site . url }}/assets/2017-10-10-code.hs)

