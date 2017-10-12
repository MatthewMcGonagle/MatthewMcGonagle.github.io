module Main where

import Control.Monad.State

main :: IO ()
main = do
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
  putStrLn . fst $ runState (simpleShow exampleTree) 0 
  putStrLn $ showTree 40 exampleTree

data Tree a = Node a (Tree a) (Tree a) | Empty
    deriving (Show)

-- Simple print.

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

type Width = Int
data WidthInfo = WidthInfo { nodeWidth :: Width 
                           , leftWidth ::  Width
                           , rightWidth ::  Width
                           }

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

type Position = Int

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

showTreeWPositions :: Tree (String, Position) -> String
showTreeWPositions x = concat levelStrings
    where levelStrings = map (reverse . combineLevel) levels
          levels = reformatTree x
          combineLevel y = (++ "\n") . fst $ runState (foldM addNodeString "" y) (-1) 

convertEmpty :: Tree String -> Tree String
convertEmpty Empty = Node "_" Empty Empty
convertEmpty (Node x lChild rChild) = Node x lChild' rChild'
    where lChild' = convertEmpty lChild
          rChild' = convertEmpty rChild

-- First argument is the position of the root node.
showTree :: Position -> Tree String -> String
showTree rootPos x = showTreeWPositions mixedTree
    where mixedTree = combine x' posTree
          posTree = fst $ runState (computeNodePositions widthTree) rootPos 
          widthTree = computeWidths x'
          x' = convertEmpty x
