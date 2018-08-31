module Main where

import Text.ParserCombinators.Parsec 
import System.IO

-- | Encode the name of the statistics run with the information it contains.

data RunStatistics = RunStatistics String [(String, String)]
    deriving (Show)

-- | Parser to skip a line of the run time statistics file. 
line = do
       many $ noneOf "\n"
       char '\n'
       return ()

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

-- | Run time statistics file parser. Parser returns a list of
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

-- | Parser for the number of bytes (comma separated every three digits) in the heap allocation section of the run-time statistics. 
nBytes = do
         digitGroups <- spaces >> (many digit) `sepBy` char ','
         return $ concat digitGroups

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

-- | Turn table information into a string representing a comma separated file. Exporting this will allow us
--   to use the python pandas module to easily investigate the outcomes. 
makeCSV :: [[String]] -> String
makeCSV = concat . (map makeRow) 
    where makeRow = tail . combineCols . insertCommas 
          insertCommas = map (',' : )
          combineCols = foldr (++) "\n"

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
    
