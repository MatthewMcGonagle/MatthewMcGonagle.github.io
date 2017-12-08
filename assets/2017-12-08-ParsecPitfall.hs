module Main where

import Text.Parsec.Prim
import Text.Parsec.Char
import Text.Parsec.Combinator

main :: IO ()
main = do

    let ab = char 'a' >> char 'b' 

    putStrLn "Parsing 'abababCD' with (many ab)"
    -- Will parse successfully with consumption. 
    parseTest (many ab) "abababCD"

    putStrLn "\nParsing 'abababaD' with (many ab)"
    -- Will have a consumption error.
    parseTest (many ab) "abababaD"

    putStrLn "\nParsing 'abababaD' with (many $ try ab)"
    -- Will parse successfully with consumption.
    parseTest (many $ try ab) "abababaD"

    let spacesAb = spaces >> ab

    putStrLn "\nParsing ' ab ab ab CD' with (many spacesAb)"
    -- Will have a consumption error (will consume spaces before making error at "C").
    parseTest (many spacesAb) " ab ab ab CD"

    putStrLn "\nParsing ' ab ab ab cd' with (many $ try spacesAb)"
    -- Will parse successfully with consumption.
    parseTest (many $ try spacesAb) " ab ab ab cd"
   
    let cd = char 'C' >> char 'D'

    putStrLn $  "\nParsing 'ab ab ab CD' with\n"
             ++ "(do\n"
             ++ "      many $ try spacesAB\n"
             ++ "      cd\n"
             ++ ")"
    -- Will have a consumption error at the space before "CD", BUT it will report an
    -- an error at "C" when looking for space or "a".
    parseTest ( do
                    many $ try spacesAb
                    cd 
              )
              "ab ab ab CD" 


    putStrLn $  "\nParsing 'ab ab ab CD' with\n"
             ++ "(do\n"
             ++ "      many $ try spacesAB\n"
             ++ "      spaces >> cd\n"
             ++ ")"
    -- Will successfully parse with consumption.
    parseTest ( do
                    many $ try spacesAb
                    spaces >> cd 
              )
              "ab ab ab CD" 

    -- The first example in the post, not necessarily written very neatly.
    putStrLn $  "\nParsing 'ab ab ab CD' with\n"
             ++ "(do\n"
             ++ "      many $ try (spaces >> char 'a' >> char 'b')\n"
             ++ "      char 'C'\n"
             ++ "      char 'D'\n"
             ++ ")"
    -- Will have a consumption error at the space before "CD", BUT it will report an
    -- an error at "C" when looking for space or "a".
    parseTest ( do
                    many $ try (spaces >> char 'a' >> char 'b') 
                    char 'C'
                    char 'D' 
              )
              "ab ab ab CD" 

