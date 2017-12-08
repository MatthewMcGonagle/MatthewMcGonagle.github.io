---
layout: post
title: A Pitfall of Errors in Parsec Package for Haskell
date: 2017-12-08
---

## [Download the Source Code for this Post]({{site . url}}/assets/2017-12-08-ParsecPitfall.hs)

For this post we will be looking a pitfall of the error reporting in the Parsec package. The Parsec package is a library that uses a monadic approach to parsing stream data (such as a `String`) to allow the user to combine primitive parsers to create more complex parsers. Ideally, when your parser encounters an error in the stream source (that is, an error in the format of the input data) that it can't deal with, then it should report where in the stream that error occurs.
However, the Parsec package is kind of notorious for doing a sloppy job of this. After spending some time diving into the source files parsec/Text/Parsec/Prim.hs, parsec/Text/Parsec/Char.hs, and parsec/Text/Parsec/Error.hs, I will explain why this occurs to the best of my understanding.

Essentially the source of the problem with error reporting stems from two sources:
* Parsec parsers keep track of errors even if the effect it doesn't stop the parser. For example, such an error occurs when `many (char 'a')` stops encountering the character `'a'`. 
* When using bind `>>=` (also called `parserBind` in the parsec package) to combine parsers, prior (non-critical) error messages are combined with more current error messages (possibly critical) by only reporting the error that occurs latest in the stream source (that is, the error with largest stream position). If they occur at the same time, then they are reported together.

The problem with `parserBind` choosing to use the error with largest position, is that this error can occur chronologically before the error that kills the parsing. Using examples, we will see that it is easy to create this behavior by not being careful when using `many`. For example, we will see that
``` haskell
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
``` 
will produce the following output:
```
Parsing 'ab ab ab CD' with
(do
      many $ try (spaces >> char 'a' >> char 'b')
      char 'C'
      char 'D'
)
parse error at (line 1, column 10):
unexpected "C"
expecting space or "a"
```
The error reporting seems to imply that the parser is getting stuck with an error when running `many $ try (spaces >> char 'a' >> char 'b')`, but this is NOT true. The parser is actually having a critical error when trying to run `char 'C'` at the space that occurs in " CD". However, the error at this space has smaller position than the error at 'C' that occurs when `many $ try (spaces >> char 'a' >> char 'b')` stops running. The latter occurs after the space when it tries to parse `char 'a'` at the position of 'C' in the string. Note that 'C' occurs after the space in the sub-string " CD".
## [Download the Source Code for this Post]({{site . url}}/assets/2017-12-08-ParsecPitfall.hs)
