{- |
Module  :  MakeFolds.hs 

Meant for benchmarking the different combinations of composing two fold operations that build up lists in terms of the possibilities
for their fold direction, their lazyness/strictness, and whether the computations of the mapping of each element is independent
or dependent on the previous computations in the fold.
-}

module Main where

import System.Environment
import Data.List
import Data.Int

-- | Encode accumulator function for a fold for which type of fold it is meant for.
--   Note that this doesn't specify the level of lazyness/strictness of the accumulation
---  function; it only specifies the level for the fold itself.
data FoldMap acc b = LMap (acc -> b -> acc) -- ^ Lazy left fold, i.e. foldl. 
                   | RMap (b -> acc -> acc) -- ^ Lazy right fold, i.e. foldr. 
                   | LMap' (acc -> b -> acc) -- ^ Strict left fold, i.e. foldl'. 

-- | Apply a specified type of fold map to a list and initial accumulator.
fold :: FoldMap acc b -- ^ The type of fold and the accumulation map.
     -> acc  -- ^ Initial accumluator
     -> [b]  -- ^ List to fold over. 
     -> acc  -- ^ Returns the final accumulator.

fold (LMap f) = foldl f 
fold (RMap f) = foldr f 
fold (LMap' f) = foldl' f

-- | Accumulator function to lazily map addition to each element of a list. This builds
--   the list in reverse order using ":" and is meant for left folds.
addl :: (Num a) => a -- ^ The shift to apply to every element
                -> [a]  -- ^ The accumulation list
                -> a  -- ^ The next element to shift.
                -> [a] -- ^ Returns a list where next value was shifted and is
                       --   now on the left side of the list. 

addl shift acc x = (shift + x) : acc

-- | Strict left fold accumulator similar to the function addl. The order of the resulting list 
--   will be in the opposite order that it was originally. 
addl' :: (Num a) => a -- ^ Shift value to add to each element.
                 -> [a] -- ^ The accumulation list.
                 -> a -- ^ The next element.
                 -> [a] -- ^ Returns the shifted element added to the left side of the list. 
addl' shift acc x = let x' = shift + x 
                    in x' `seq` x' : acc

-- | Lazy right fold accumulator similar to the function addl. However, the transformed list will 
--   in the same order that it was originally. 
addr :: (Num a) => a -- ^ The shift to apply to each element.
                -> a -- ^ The next element.
                -> [a] -- ^ The accumulation list.
                -> [a] -- ^ Returns the shifted value of the element added to the left side of the list. 
addr shift x acc = (shift + x) : acc

-- | Strict right accumulation function similar to the function addl. However, the transformed list will 
--   be in the same order that it was originally.
addr' :: (Num a) => a -- ^ The shift to add to each element of the list.
                 -> a -- ^ The next element. 
                 -> [a] -- ^ The accumulation list.
                 -> [a] -- ^ Returns the shifted element added to the left side of the accumulation list. 
addr' shift x acc = let x' = shift + x
                    in x' `seq` x' : acc

-- | For fold maps that return accumulators that are lists, this function automatically
--   composes them where the initializors are empty lists.
compose :: FoldMap [c] b -- ^ The outer fold map in the composition.
        -> FoldMap [b] a -- ^ The inner fold map in the composition.
        -> [a] -- ^ The list to apply the composition to.
        -> [c] -- ^ Returns a list where we have applied the composition of the folds.
compose g f = (fold g []) . (fold f [])

-- | For looking at folds where the computation of each step depends on the previous
--   state, this function will automatically compose the folds. We need to specify the 
--   initializers for both the inner and outer folds. That is, the initializer for the outer
--   fold will be independent of the result of the inner fold. 
compose2 :: FoldMap ([c], c) b -- ^ Outer fold map.
         -> FoldMap ([b], b) a -- ^ Inner fold map.
         -> ([c], c) -- ^ Outer initializer.
         -> ([b], b) -- ^ Inner initializer. 
         -> [a] -- ^ The list to apply the composition to.
         -> [c] -- ^ Returns the result of applying the compositon of the folds.
compose2 g f initialg initialf = fst . (fold g initialg) . fst . (fold f initialf)

-- | Lazy left fold accumulation function where the result depends on the last processed element.
--   The accumulator is the every element added by a constant shift and then modded by
--   the last element. The initializer should specify the value to mod by for the first element. 
--   The accumulator will be in the opposite order that it was originally.
addmodl :: (Integral a) => a -- ^ The constant shift to apply to each element.
                        -> ([a], a) -- ^ The initializer consists of an initial list
                                    --   and the value to mod the first operation by.
                        -> a -- ^ The next element.
                        -> ([a], a) -- ^ Returns the fully constructed list and the last value seen.
addmodl shift (acc, last) x = let x' = (shift + x) `mod` (1 + last)
                                  last' = x 
                              in (x' : acc, last') 

-- | Lazy right fold function similar to the function addmodl, except now the returned list will
--   be in the same order that it was originally. 
addmodr :: (Integral a) => a -- ^ The constant shift to add to each element.
                        -> a -- ^ The next element.
                        -> ([a], a) -- ^ The accumulator contains a list of transformed values and
                                    --   the last value processed which will be used to mod out the
                                    --   the next shift. 
                        -> ([a], a) -- ^ Returns the transformed list and the newest last value processed.

addmodr shift x (acc, last) = let x' = (shift + x) `mod` (1 + last)
                                  last' = x
                              in (x' : acc, last')

-- | Strict left fold function similar to the function addmodl. The resulting list will be in the 
--   opposite order from that it was originally in.
addmodl' :: (Integral a) => a -- ^ The constant shift.
                        -> ([a], a) -- ^ The accumulator consists of a list of transformed values
                                    --   and the last value seen which will be used to mod out
                                    --   the next shift.
                        -> a -- ^ The next value.
                        -> ([a], a) -- ^ Returns the transformed values and the next value to use for modding.
addmodl' shift (acc, last) x = let x' = (shift + x) `mod` (1 + last)
                                   acc' = x' : acc
                               in x' `seq` acc' `seq` (acc', x)

-- | Strict right fold function similar to the function addmodl. However, the resulting list will be in the 
--   same order from that it was originally in.
addmodr' :: (Integral a) => a -- ^ The constant shift.
                         -> a -- ^ The next value.
                         -> ([a], a) -- ^ The accumulator consists of a list of transformed values
                                     --   and the value to use for modding.
                         -> ([a], a) -- ^ Returns a list of transformed values and the value to use
                                     --   to mod out the next shift.
addmodr' shift x (acc, last) = let x' = (shift + x) `mod` (1 + last)
                                   acc' = x' : acc
                               in x' `seq` acc' `seq` (acc', x)

-- | Use the function parameters to run a composition of two folds a on a list of numbers. To force the folds
--   to run over the entire list, we print out the head and last element of the list.
--   Arguments should be 
--          testStr = "ind", "dep", or otherwise
--              Specifies whether to use accumulation functions where the computation is independent
--              or dependent on previous values. For otherwise, no folds are performed. Use this to benchmark
--              just creating the initial list and printing out its head and last elemetns.
--          gDirection = "L", "R", "L'", or "R'". For the outer fold, specifies the direction and whether
--              it is strict or lazy.
--          fDirection = "L", "R", "L'", or "R'". For the inner fold, specifies the direction and whether
--              it is strict or lazy.
--          numPts :: Int
--              The number of elements in the list to fold over. The list is simply [1 .. numPts].

main :: IO ()
main = do

    -- Get the parameters and build the list to fold over.
    args <- getArgs :: IO [String]
    let [testStr, gDirection, fDirection, numPts] = args
        n = read numPts :: Int64
        nums = [1..n] :: [Int64]
       
    -- Print out the parameters for the user to double check. 
    print $ "Fold types : " ++ gDirection ++ fDirection ++ testStr
    print "Number of Points = "
    print n

    print "Head and Last Elements of Untransformed List = "
    print $ head nums
    print $ last nums 

    case testStr of 
         -- Transformed values are independent on transformations of other values in the list.
         "ind" -> do
                  let gFunc = case gDirection of
                                   "L" -> LMap $ addl 2
                                   "R" -> RMap $ addr 2 
                                   "L'" -> LMap' $ addl' 2 
                                   "R'" -> RMap $ addr' 2
                      fFunc = case fDirection of
                                   "L" -> LMap $ addl 1
                                   "R" -> RMap $ addr 1 
                                   "L'" -> LMap' $ addl' 1
                                   "R'" -> RMap $ addr' 1
                      mapped = (gFunc `compose` fFunc) nums
                  print $ head mapped
                  print $ last mapped

         -- Transformed values are dependent on transformations of other values in the list.
         "dep" -> do
                 let gFunc = case gDirection of 
                                  "L" -> LMap $ addmodl 2
                                  "R" -> RMap $ addmodr 2 
                                  "L'" -> LMap' $ addmodl' 2 
                                  "R'" -> RMap $ addmodr' 2
                     fFunc = case fDirection of 
                                  "L" -> LMap $ addmodl 1
                                  "R" -> RMap $ addmodr 1 
                                  "L'" -> LMap' $ addmodl' 1 
                                  "R'" -> RMap $ addmodr' 1
                     initial = ([], 100)
                     mapped = (gFunc `compose2` fFunc) initial initial nums
                 print $ head mapped
                 print $ last mapped

         -- Don't do any folds. Use this to benchmark just creating the untransformed list.
         otherwise -> do
                      print "NO TEST"
