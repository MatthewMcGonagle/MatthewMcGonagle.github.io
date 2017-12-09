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
The error reporting seems to imply that the parser is getting stuck with an error when running `many $ try (spaces >> char 'a' >> char 'b')`, but this is NOT true. The parser is actually having a critical error when trying to run `char 'C'` at the space that occurs in the sub-string " CD". However, the error at this space has smaller position than the error at 'C' that occurs when `many $ try (spaces >> char 'a' >> char 'b')` stops running. The latter occurs after the space when it tries to parse `char 'a'` at the position of 'C' in the string. Note that 'C' occurs after the space in the sub-string " CD".

Since the latter error occurs at a larger position in the string "ab ab ab CD", it is the one that is reported. Thus, the error report becomes confusing and can easily steer you off the right track. 

To see that the space is the issue, now consider the result of running the following parser that parses away spaces before parsing "CD".

``` haskell
    putStrLn $  "\nParsing 'ab ab ab CD' with\n"
             ++ "(do\n"
             ++ "      many $ try (spaces >> char 'a' >> char 'b')\n"
             ++ "      spaces\n"
             ++ "      char 'C'\n"
             ++ "      char 'D'\n"
             ++ ")"
    -- Will have a consumption error at the space before "CD", BUT it will report an
    -- an error at "C" when looking for space or "a".
    parseTest ( do
                    many $ try (spaces >> char 'a' >> char 'b') 
                    spaces
                    char 'C'
                    char 'D' 
              )
              "ab ab ab CD" 
```

When we run this, we get the following output
```
Parsing 'ab ab ab CD' with
(do
      many $ try (spaces >> char 'a' >> char 'b')
      spaces
      char 'C'
      char 'D'
)
'D'
```

The output of the parsing is `'D'`. This is correct, because we didn't tell our parser to keep track of what is parsing. So the output is the last thing to be correctly parsed, which in this case is the character 'D'.

Later in the post, we will use many simple examples to examine how this process occurs. We will also point to where in the source files for Parsec you can find the problem with error propagation.

## Necessary Remarks on How Parsec Works

## Simple Examples

Let us look at a simple example of running `many` on a parser of more than one character.

``` haskell
    let ab = char 'a' >> char 'b' 

    putStrLn "Parsing 'abababCD' with (many ab)"
    -- Will parse successfully with consumption. 
    parseTest (many ab) "abababCD"
```

When we run this, we get the following output:
```
Parsing 'abababCD' with (many ab)
"bbb"
```
We don't get any error, and the parsing was successful. Note that each successful parse using `ab` gives exaclty one 'b' character. Then `many ab` successfully parses `ab` three times to give "bbb". 

It is important to note that the parser `ab` fails to parse the sub-string "CD" WITHOUT consuming any input. This error is in fact internally recorded, but it isn't considered critical since nothing was consumed. Therefore, the parsing will be successful.

Now let's consider an example where we get failure.
``` haskell
    putStrLn "\nParsing 'abababaD' with (many ab)"
    -- Will have a consumption error.
    parseTest (many ab) "abababaD"
```

When we run this we get the following output:
```
Parsing 'abababaD' with (many ab)
parse error at (line 1, column 8):
unexpected "D"
expecting "b"
```

So now we have an error; the parsing was NOT successful. Why so? Now, when `ab` tries to parse the sub-string "aD", it will consume the first character 'a'. However, when `char 'b'` tries to parse the character 'D', it fails. Now `char 'b'` does NOT consume the character `D`, but this doesn't matter as `char 'a'` has already consumed the character 'a'. Therefore, `ab` fails while consuming input; such an error causes the parser `many ab` to fail too.

This problem was created by `ab` parsing more than one character; recall that a single character parsing such as `char 'b'` fails WITHOUT consuming input. We can fix this issue by making use of `try`.
``` haskell
    putStrLn "\nParsing 'abababaD' with (many $ try ab)"
    -- Will parse successfully with consumption.
    parseTest (many $ try ab) "abababaD"
```

When we run this, we get
```
Parsing 'abababaD' with (many $ try ab)
"bbb"
```
Now the parser runs succesfully. Again, the error encountered by `char 'b'` trying to parse 'D' is still kept track of internally. 

So far the error messages have been pretty transparent; they haven't misdirected our attention in any way. We will create misdirection by introducing parsing of spaces (a perfectly natural thing to do).

Our first example will fail and the error message will give us a slight misdirection. 
``` haskell
    let spacesAb = spaces >> ab

    putStrLn "\nParsing ' ab ab ab CD' with (many spacesAb)"
    -- Will have a consumption error (will consume spaces before making error at "C").
    parseTest (many spacesAb) " ab ab ab CD"
```

When we run this, we get the following output:
```
Parsing ' ab ab ab CD' with (many spacesAb)
parse error at (line 1, column 11):
unexpected "C"
expecting space or "a"
```

We haven't yet told our parser to look for "CD", but `spacesAB` is inside `many`. So why doesn't it just return "bbb" for the successfully parsed "ab ab ab"? Once again, we have a consumption issue. The parser `spacesAB` will consume the space in the sub-string "ab CD". Therefore, `spacesAB` with consumption when it tries to parse "CD".

The error message is complaining about "C", and at first sight you might be mislead to behave that `many` doesn't behave as you thought it did. However, the problem is again consuming input before failure. The fix to that is easy enough, use `try`. 

When we use `try` combined with `spacesAB`, we will get a successfull parsing:
```haskell
    putStrLn "\nParsing ' ab ab ab CD' with (many $ try spacesAb)"
    -- Will parse successfully with consumption.
    parseTest (many $ try spacesAb) " ab ab ab CD"
```

When we run this, we get the following output:
```
Parsing ' ab ab ab CD' with (many $ try spacesAb)
"bbb"
```

So, now the parsing is successful. Now, let's try adding in the parsing of "CD", but let us consider an example where we FORGET to parse the space infront of "CD".
``` haskell
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
```
When we run this, we get the following output:
```
Parsing 'ab ab ab CD' with
(do
      many $ try spacesAB
      cd
)
parse error at (line 1, column 10):
unexpected "C"
expecting space or "a"
```
Now, we see that we get a complete misdirection. The error message seems to be indicating that there is something wrong with the parsing `many $ try spacesAB`. However, this is not the case; the parsing `many $ try spacesAB` finishes correctly. The problem is that we aren't parsing the space in the sub-string " CD". So, then why does the error seem to be talking about parsing `many $ try spacesAB`? It is doing so, because it is the error that causes `many` to stop trying to parse `try spacesAB`.

So why is the parser reporting this error instead of the real error that occurs when we try to parse the space in " CD" as the character 'C'? The answer lies in how `bindParser` deals with combining errors from parsers. For some reason,
## [Download the Source Code for this Post]({{site . url}}/assets/2017-12-08-ParsecPitfall.hs)
