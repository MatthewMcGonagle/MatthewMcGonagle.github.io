---
layout : post
title : Profiling Compositions of Folds in Haskell
date : 2018-08-30
---

## [Download MakeFolds.hs here]({{site . url}}/assets/2018-09-01-files/MakeFold.hs).
## [Download AggregateStats.hs here]({{site . url}}/assets/2018-09-01-files/AggregateStats.hs)
## [Download analysis.py here]({{site . url}}/assets/2018-09-01-files/analysis.py)

In this post we will be looking at profiling the composition of two fold operations in Haskell for different
conditions based on lazyness/strictness and whether the computation at each step is dependent on previous
computations. In particular, we will be looking at fold operations where the accumulator is a list that 
is built up at each step.

Each fold will be one of four types: a lazy left fold "L", a lazy right fold "R", a strict left fold "L'",
or a lazy right fold with a strict evaluation on its accumulator function "R'".

We will compose two folds. Each fold will build up a list using the ":" operator. Note that the folds
aren't designed to have the same outputs. We are only interested in the how the effects of lazyness/strictness
come into play for the effects on how optimal the fold operations are.

These types of folds are run on a list of one million integers for two different scenarios. In one scenario, both
accumulation functions simply shift each integer by a constant and connects it to the new list using ":". This
scenario is similar to doing `map (shift + ) list`, except we don't necessarily keep the list in the same order. 
Also note that for this scenario, the computation of the shift is independent of previous computations of the
shift.

In the second scenario, in addition to adding by a shift, we mod out (i.e. modulo arithmetic) by the previously
seen value. This means that the compuation for the number we are adding to our new list depends on the computation
of previous values.

We also compare to a benchmark of not doing any fold. That is, we just construct the original list of numbers
and look at the head and last members to make sure the list is constructed. 

We will look at summaries of the run-time statistics for these different scenarios. For example, here are
graphs of the execution times for the different scenarios. The no test scenario is indicated by "None".

![Graph of Independent Execution Times]({{site . url}}/assets/2018-09-01-files/Analysis/indTimes.svg)
![Graph of Dependent Execution Times]({{site . url}}/assets/2018-09-01-files/Analysis/depTimes.svg)

As we can see, for the independent operations, the composition of two right folds (lazyness of the shift seems
like a non-issue) gives the best execution times. However, for the dependent computations, the composition of
two strict left folds is the clear winner.

# Composing the Independent Folds 

Now we look at making the composition of our folds in `MakeFolds.hs`. We will need to import the following
modules:
``` haskell
import System.Environment
import Data.List
import Data.Int
```
To automate our composition process, no matter what the type of fold, we will define a data type `FoldMap` and
a function `fold`.
``` haskell
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
```
Next, let's get our independent shift operations for each of the fold types.
``` haskell
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
```
Then we need a function to compose our independent operation folds:
``` haskell
-- | For fold maps that return accumulators that are lists, this function automatically
--   composes them where the initializors are empty lists.
compose :: FoldMap [c] b -- ^ The outer fold map in the composition.
        -> FoldMap [b] a -- ^ The inner fold map in the composition.
        -> [a] -- ^ The list to apply the composition to.
        -> [c] -- ^ Returns a list where we have applied the composition of the folds.
compose g f = (fold g []) . (fold f [])
```

# Composing the Dependent Folds

First we need to make our accumulator functions where the transformation of each list value depends on the
previous values in a non-trivial way. We accomplish this by keeping track of a state of the previously seen
value. Then we use this to mod out the shift.

``` haskell
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
```

Next, we write a function to make a composition of the dependent folds based on any direction or strictness.
``` haskell
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
```

# Main Execution of Folds

Next we construct our function `Main` for running our fold operations. The program will run in three states:

1. Independent folds.
2. Dependent folds.
3. No test. We use this to get a benchmark on run-time statistics for simply constructing the list 
of integers without transforming them. To make sure the list is constructed, we print out the head and last
elements of the list. 

Our parameters will also give the information the direction and strictness ("L", "R", "L'", or "R'") of the
inner and outer folds. For example, "L" is a lazy left fold and "L'" is a strict left fold. Now, "R'" isn't
entirely  strict; it just makes sure that the transformation is strict.

So the parameters need to specify the outer and inner properties of the folds (in that order); for example "L R"
is an outer lazy left fold and an inner lazy right fold.

The final parameter is the number of points to put in the initial list.

Our main function is the following:
``` haskell
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
```
Note that in order to make sure we construct the entire mapped list, we print out the head and last elements;
although it is possible that some of the intermediate elements could be left as thunks.

# Getting the Run-time Statistics

We will write instructions for when using `stack` to build and execute the Haskell application. A nice reference
for profiling Haskell programs is provided by 
["Real World Haskell" Chapter 25: Profiling and Optimization] 
(http://book.realworldhaskell.org/read/profiling-and-optimization.html). A detailed list the run-time statistics
reports generated by GHC can be found in the documentation at [Section 6.12.2: Run-time Control of the Users Guide
for GHC](https://downloads.haskell.org/~ghc/6.12.2/docs/html/users_guide/runtime-control.html) (this is actually
a little separated and hidden from the profiling section of the user's guide).

We want stack to recognize the executable as `MakeFolds-exe`, so we need to add a new executable section to
our .cabal file. So add the following to the project .cabal file (the name of the project is `FoldProfile`):
``` cabal
executable MakeFolds-exe
  main-is: MakeFolds.hs
  other-modules:
      Paths_FoldProfile
  hs-source-dirs:
      app
  ghc-options: -threaded -rtsopts -with-rtsopts=-N
  build-depends:
      FoldProfile
    , base >=4.7 && <5
  default-language: Haskell2010
```

Next, we need to make sure that stack has compiled the executable to allow for profiling, so run the following
in a terminal:
``` 
stack build --profile
```
Next, as an example, let us run the executable to get a profile of the run-time statistics for 
one million points, independent folds, a lazy left outer fold, a lazy right inner fold, and we will save the run-time statistics
report in a sub-directory `Analysis` as `Analysis\stderrorindLR`. To accomplish this, we run
```
stack exec MakeFolds-exe ind L R 1000000 --rts-options "-sAnalysis\stderrorindLR"
```
Note the use of the `-s` run-time statistics flag and how what follows (without any space) is where to
save the report. 

The report looks like the following
``` text
...\bin\MakeFolds-exe.EXE ind L R 1000000 +RTS -N -sAnalysis\stderrorindLR
     392,351,456 bytes allocated in the heap
     628,445,664 bytes copied during GC
     114,861,768 bytes maximum residency (9 sample(s))
      32,725,304 bytes maximum slop
             294 MB total memory in use (0 MB lost due to fragmentation)

                                     Tot time (elapsed)  Avg pause  Max pause
  Gen  0       368 colls,   368 par    1.109s   0.274s     0.0007s    0.0021s
  Gen  1         9 colls,     8 par    1.250s   0.384s     0.0426s    0.1471s

  Parallel GC work balance: 0.00% (serial 0%, perfect 100%)

  TASKS: 10 (1 bound, 9 peak workers (9 total), using -N8)

  SPARKS: 0 (0 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  INIT    time    0.000s  (  0.008s elapsed)
  MUT     time    0.203s  (  0.130s elapsed)
  GC      time    2.359s  (  0.658s elapsed)
  RP      time    0.000s  (  0.000s elapsed)
  PROF    time    0.000s  (  0.000s elapsed)
  EXIT    time    0.000s  (  0.000s elapsed)
  Total   time    2.562s  (  0.796s elapsed)

  Alloc rate    1,931,576,398 bytes per MUT second

  Productivity   7.9% of total user, 16.4% of total elapsed

gc_alloc_block_sync: 9516
whitehole_spin: 0
gen[0].sync: 0
gen[1].sync: 0
```

To get the statistics for all of the possiblities, we need to run a script to cover the many numerous 
possibilities. For example, the following is a Windows batch script for running all of them and then printing the
results to the terminal:
``` batch
set n=1000000
set tests=(ind, dep)
set actions=(L,R,L',R')
set baseName=Analysis\stderror

stack exec MakeFolds-exe NOTEST NULL NULL %n% --rts-options "-s%baseName%NOTEST"

for %%k in %tests% do (
    for %%i in %actions% do (
        for %%j in %actions% do stack exec MakeFolds-exe %%k %%i %%j %n% --rts-options "-s%baseName%%%i%%j%%k"
    )
)

type stderrorNOTEST

for %%k in %tests% do (
    for %%i in %actions% do (
        for %%j in %actions% do type %baseName%%%i%%j%%k
    )
)
``` 

Next, we will make a simple parser to pull out the information we are interested in from each of these
reports.

# Aggregating the Results

Next, we create a new executable using `AggregateStats.hs`. The purpose of this program is gather all of the
statistics we are interested in from the different run-time statistics reports generated by running
`MakeFolds-exe` for all of the different possibilities of folds.

We will be using the Parsec library to build our parsers. First, let us import the appropriate modules.
``` haskell
import Text.ParserCombinators.Parsec 
import System.IO
```
We will keep the information for a particular run-time statistics report in a data type `RunStatistics`:
``` haskell
-- | Encode the name of the statistics run with the information it contains.

data RunStatistics = RunStatistics String [(String, String)]
    deriving (Show)
```

We will need a parser to skip lines of the file that we are uninterested in.
``` haskell
-- | Parser to skip a line of the run time statistics file. 
line = do
       many $ noneOf "\n"
       char '\n'
       return ()
```
Next, let's make a parser for the section giving heap memory allocation details in terms of bytes.
``` haskell
-- | Parser for the number of bytes (comma separated every three digits) in the heap allocation section of the run-time statistics. 
nBytes = do
         digitGroups <- spaces >> (many digit) `sepBy` char ','
         return $ concat digitGroups

-- | Parser for the heap allocation bytes section of the run time statistics file.
--   Parser returns a list of pairs of strings.

byteStatsSection = 
        do
        let byteLine = do
                       numBytes <- nBytes
                       many $ noneOf "\n"
                       char '\n'
                       return numBytes
        totalHeap <- byteLine 
        gcHeap <- byteLine
        maxResidency <- byteLine
        maxSlop <- byteLine
        line
        return [ ("totalHeap (b)", totalHeap) 
               , ("gcHeap (b)", gcHeap)
               , ("maxResidency (b)", maxResidency)
               , ("maxSlop (b)", maxSlop)
               ]
```
Then let's make some parsers to handle the section giving timing details:
``` haskell
-- | Parser for a line of the timing information section of the run time statistics file.
--   Parser returns a pair of strings.

timeLine =
    do
    spaces
    title <- many alphaNum
    spaces 
    string "time"
    spaces
    time <- many $ noneOf "s" 
    char 's'
    many $ noneOf "\n"
    char '\n'
    return (title ++ " (s)", time)

-- | Parser for the timing information section of the run time statistics file.
--   Parser returns a list of pairs of strings.

timeStatsSection = 
    do
    line
    mut <- timeLine
    gc <- timeLine
    count 3 line 
    total <- timeLine
    return $ [mut, gc, total]
```

Next, we need to parse the section on productivity:
``` haskell
-- | Parser for the productivity section of the run time statistics file.
--   Parser returns a list of pairs of strings.           

productivitySection = 
    do
    spaces >> string "Productivity"
    spaces
    user <- many $ noneOf "%"
    string "% of total user," >> spaces
    total <- many $ noneOf "%"
    char '%'
    many $ noneOf "\n"
    char '\n'
    return [ ("user Prod (%)", user)
           , ("total Prod (%)", total) 
           ]
```

Finally, let's put it all together to build the parser for the entire file:
``` haskell
--   pairs of strings, The first element gives the name of the value parsed, and
--   the second is the value as a string. 

statsFile :: GenParser Char st [(String, String)]
statsFile = do
            line
            byteStats <- byteStatsSection
            count 11 line
            timeStats <- timeStatsSection
            count 3 line
            productivity <- productivitySection
            count 5 line
            eof
            return $ byteStats ++ timeStats ++ productivity 

-- | Parse a run time statistics file.
parseStats :: String -- ^ The file as a string (or more accurately as a character stream)
           -> Either ParseError [(String, String)] -- ^ Returns either a ParseError or a list of
                                                   --   of the name of each item parsed and a string
                                                   --   representation of its value.
parseStats = parse statsFile "(unknown)" 
```

Now, let's write a function to handle collecting the statistics from one of the reports.
``` haskell
-- | Collect the statistics in a run-time statistics file. 

getStats :: String -- ^ The directory of the file relative to the current directory.
         -> String -- ^ The name of the file.
         -> IO RunStatistics  -- ^ Returns the statistics collected, the name given to it is the filename.
getStats directory fileName =
    do
    fileHandle <- openFile (directory ++ fileName) ReadMode
    contents <- hGetContents fileHandle
    let stats = case parseStats contents of 
                     Right s -> RunStatistics fileName s
                     Left error ->  RunStatistics fileName []
    stats `seq` hClose fileHandle -- make sure to compute stats before file is closed.
    return stats 
```

Next, we will want to double check the results we have collected, so let's write some functions that
will allow us to print them nicely to the terminal (this isn't prettiest way to print them, it is more
quick and dirty).
``` haskell
-- | For creating a table of all results, we extract the headers of each column using
--   the first element of each pair in the list of statistics.

makeHeader :: RunStatistics -> [String]
makeHeader (RunStatistics _ pairs) = 
    let pairTitles = map fst pairs
    in  "name" : pairTitles

-- | Create a table row for the statistics results. Just use the name and the second values in each pair. 

makeRow :: RunStatistics -> [String]
makeRow (RunStatistics name pairs) = 
    let pairData = map snd pairs
    in name : pairData 

-- | Organize a list of run-time statistics into a table with column headers. It is assumed that all of the run-time
--   statistics have the same type of information in the same order.
makeTable :: [RunStatistics] -- ^ List of run-time statistics for different runs.
          -> [[String]] -- ^ When the run statistics is non-empty, returns a list of row information. The head  
                        --   element should be the column header names. 
makeTable [] = []               
makeTable runs = 
    let headerRow = makeHeader $ head runs
    in headerRow : (map makeRow runs)
```

Finally, we are going to choose to do the analysis of the results in `python` using `pandas`; the only reason
being that I am familiar with it. So we need a function that will allow us to write our results to a comma
separated file.
``` haskell
-- | Turn table information into a string representing a comma separated file. Exporting this will allow us
--   to use the python pandas module to easily investigate the outcomes. 
makeCSV :: [[String]] -> String
makeCSV = concat . (map makeRow) 
    where makeRow = tail . combineCols . insertCommas 
          insertCommas = map (',' : )
          combineCols = foldr (++) "\n"
```
Finally, we make our main executable to collect the statistics from the different reports.

``` haskell
-- | Collect the statistics contained in the run-time statistics files located in the Analysis sub-directory.
--   Save the collected statistics into a comma separated file.

main :: IO ()
main = do
    
    let exts1 = ["L", "R", "L'", "R'"] 
        exts2 = ["ind", "dep"]
        -- Compute all possibilities of extensions. Manually add in the NOTEST case later.
        allExts =
            do
            ext1 <- exts1
            ext2 <- exts1
            ext3 <- exts2
            return $ ext1 ++ ext2 ++ ext3
        fileNames = map ("stderror" ++ ) ( "NOTEST" : allExts) 
    print "Aggregating Stats"
    fileStats <- mapM (getStats "Analysis\\") fileNames 
    let table = makeTable fileStats
        pad width x
            | length x < width = replicate (width - (length x)) ' ' ++ x
            | otherwise = x
        convertRow = foldr (\x acc -> (pad 20 x) ++ acc) "" 
        table' = map convertRow table 
        csv = makeCSV table
    sequence_ $ map putStrLn table' 
    print csv
    writeFile "Analysis\\stats.csv" csv
```

Next, let's use `python` to graph our different results.

# Graphing the Results

We make the following `python` program for opening the `.csv` file, formatting the results, and making our
graphs.
``` python
'''
analysis.py

Graphs the outcomes of the run-time statistics.
'''

import pandas as pd
import matplotlib.pyplot as plt

stats = pd.read_csv('Analysis\\stats.csv', index_col = 'name')

# Remove the prefix 'stderror' from all of the names.
nSkip = len("stderror")
index = [name[nSkip:] for name in stats.index]

# Change the no test case to have the name 'None' + some padding that will let it be correctly
# processed with the other names later when we remove the dependence state from the names in
# the index..

noTestName = "stderrorNOTEST"[nSkip:]
index[index == noTestName] = "None123" # Pad the end by 3 characters that will be removed later. 

# Let's check up on the indices we have made.
print(index)

# Give stats the new indices.
stats.index = index

# The state of the dependence is given by the last three characters, except in the no test case.
# Also remove the state of dependence from the index names, this is where the padding for the
# no test index was needed.

stats['dependence'] = [name[-3:] for name in stats.index]
stats.index = [old[:-3] for old in stats.index] 

# Manually change the no test case.

stats.loc['None', 'dependence'] = 'None'

# Now convert the heap allocations from being in bytes to being in Megabytes, appropriately
# change the names of the columns too.

byteCols = ["totalHeap", "gcHeap", "maxResidency", "maxSlop"]
rename = {}
for col in byteCols:
    oldName = col + ' (b)'
    rename[oldName] = col + " (Mb)"
    stats[oldName] = stats[oldName] * 1e-6 

stats = stats.rename(index = str, columns = rename)

print(stats)

# We will be splitting our graphs in to two cases. Graph all of the dependent and no test case together.
# Second case is graphing all of the independent and no test case together.

indMask = (stats.dependence == "ind") | (stats.dependence == "None") 
depMask = (stats.dependence == "dep") | (stats.dependence == "None") 

stats = { 'ind' : stats.loc[indMask, :]
        , 'dep' : stats.loc[depMask, :]
        }

# Now make the graphs.

directory = "Analysis\\"
for split, name in zip(['ind', 'dep'], ['Independent', 'Dependent']):

    stats[split].plot.bar(y = "maxResidency (Mb)")
    plt.title('Max Residency in Heap for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Size (Megabytes)')
    plt.savefig(directory + split + 'Residency.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = ["totalHeap (Mb)", "gcHeap (Mb)"])
    plt.title('Heap Allocations for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Size (Megabytes)')
    plt.savefig(directory + split + 'Allocations.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = ["user Prod (%)", 'total Prod (%)'])
    plt.title('Productivities for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Productivity by Time (%)')
    plt.savefig(directory + split + 'Productivities.svg', bbox_inches = 'tight')
    plt.show()

    stats[split].plot.bar(y = 'Total (s)' )
    plt.title('Total Times for ' + name + ' Folds')
    plt.xlabel('Fold Types (Outer Inner)')
    plt.ylabel('Times (s)')
    plt.savefig(directory + split + 'Times.svg', bbox_inches = 'tight')
    plt.show()
```
Next, let's take a look at the graphs of the results.

# The Results

## [Download MakeFolds.hs here]({{site . url}}/assets/2018-09-01-files/MakeFold.hs).
## [Download AggregateStats.hs here]({{site . url}}/assets/2018-09-01-files/AggregateStats.hs)
## [Download analysis.py here]({{site . url}}/assets/2018-09-01-files/analysis.py)


