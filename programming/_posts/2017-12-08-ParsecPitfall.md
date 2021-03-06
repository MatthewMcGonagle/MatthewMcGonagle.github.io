---
layout: post
title: A Pitfall of Errors for Parsec Parser Combinators 
date: 2017-12-08
tags: [Haskell, Parsec]
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

Here let us make some remarks about how Parsec works internally, which at the time of this writing, isn't really clear from the documentation. We won't be going into great detail, but we will be discussing details that affects the behavior of Parsec parsers. We will be looking at some implementations inside `Text.Parsec.Prim.hs` and `Text.Parsec.Error.hs`. [Here is the version of Text.Parsec.Prim.hs]({{site . url}}/assets/2017-12-08-Prim.hs) we are looking at. [Here is the version of Text.Parsec.Error.hs]({{site . url}}/assets/2017-12-08-Error.hs) we are looking at.

So what is a parser in Parsec? It is something that has a function to produce a new output based on the following parameters:
1. Current stream state (and a user state which we will be ignoring).
2. A function that provides instructions on the output to make if the parser is succesful AND consumes input.
3. Instructions on the output to make if the parser has an error AND consumes input.
4. Instructions on the output to make if the parser is successful AND does NOT consume input.
5. Instructions on the output to make if the parser has an error AND does NOT consume input. 

Explicitly, that `ParsecT` type is defined by the following inside Prim.hs:
``` haskell
-- This example code is from
-- Module      :  Text.Parsec.Prim
-- Copyright   :  (c) Daan Leijen 1999-2001, (c) Paolo Martini 2007
-- License     :  BSD-style (see the LICENSE file)

-- | ParserT monad transformer and Parser type

-- | @ParsecT s u m a@ is a parser with stream type @s@, user state type @u@,
-- underlying monad @m@ and return type @a@.  Parsec is strict in the user state.
-- If this is undesirable, simply use a data type like @data Box a = Box a@ and
-- the state type @Box YourStateType@ to add a level of indirection.

newtype ParsecT s u m a
    = ParsecT {unParser :: forall b .
                 State s u
              -> (a -> State s u -> ParseError -> m b) -- consumed ok
              -> (ParseError -> m b)                   -- consumed err
              -> (a -> State s u -> ParseError -> m b) -- empty ok
              -> (ParseError -> m b)                   -- empty err
              -> m b
             }
```

As you can see, the functions for instructing what the parser what to do if it succeeds (in either the case of input consumed or NO input consumed) depends on a parameter of type `ParseError`. This is coming from the fact that parsers keep track of errors even if they are not stopping the operation of the parser. 

For example, the parser `many (char 'a')` will keep parsing `char 'a'` until it doesn't find an `'a'` character. At this stopping point, the parser records an error despite the fact that it doesn't terminate as being in error. 

So when we are analyzing the parsers we write, we need to think about what will happen in four scenarios:
1. The parser succeeds while consuming input.
2. The parser fails and consumes input.
3. The parser succeeds and did not consume input. For example, this happens when the parser is happy to parse 0 or more occurences of some character.
4. The parser fails and did not consume input. 

It is important to note that for a given parser `p`, the parser `many p` will succeed if this last error is `p` fails WITHOUT consuming input. If during the process of repeatedly parsing `p`, if the parser `p` ever fails while consuming input, then `many p` is considered to have failed while consuming input. Understanding this point is important to understanding how to use `many` properly.

Now, inorder to understand how errors reported by Parsec can be misdirecting you to the wrong place in the input stream, it helps to understand how errors are combined when using `parserBind` to combine parsers. The operation `parserBind` combines errors (including internal non-critical errors and critical parsing stopping errors) using function `mergeError` from Error.hs. Here is how it is defined inside Error.hs:

``` haskell
-- This example code is from
-- Module      :  Text.Parsec.Error
-- Copyright   :  (c) Daan Leijen 1999-2001, (c) Paolo Martini 2007
-- License     :  BSD-style (see the LICENSE file)

mergeError :: ParseError -> ParseError -> ParseError
mergeError e1@(ParseError pos1 msgs1) e2@(ParseError pos2 msgs2)
    -- prefer meaningful errors
    | null msgs2 && not (null msgs1) = e1
    | null msgs1 && not (null msgs2) = e2
    | otherwise
    = case pos1 `compare` pos2 of
        -- select the longest match
        EQ -> ParseError pos1 (msgs1 ++ msgs2)
        GT -> e1
        LT -> e2
``` 
So the parser only keeps the error that occurs in the largest position in the input stream. If they occur at the same location, then it combines them into one error. This method of deciding which error is correct is does NOT necessarily produce the real error. This is especially true if we aren't careful about how we write our parsers.

Finally, I would just like to remark the the parser `char`, such as `char 'a'`, does NOT consume input if it fails.

## Simple Examples Without Misdirecting Error Messages

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

## Simple Examples with Misdirecting Error Messages

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

So why is the parser reporting this error instead of the real error that occurs when we try to parse the space in " CD" as the character 'C'? The answer lies in how `bindParser` deals with combining errors from parsers. For some reason, when `bindParser` makes a decision to keep either a previously found error or a newly created error. It doesn't do so based on the chronological order they are created. It does this by choosing the error that occurs at a larger position in the input stream. If they have the same position, then the errors are combined into one error.

Okay, so let's now add in the parsing of spaces before we parse "CD".
``` haskell
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
```

Now, when we run this we get the following output:
```
Parsing 'ab ab ab CD' with
(do
      many $ try spacesAB
      spaces >> cd
)
'D'
```

The parser finishes succesfully, and the output is the last thing parsed, which is simply the character 'D'.

## Reducing the Chance of this Pitfall

So how can we reduce the chance that we have to deal with this error misdirection? In this case, the problem is created by running a parser like `many $ try (spaces >> ab)`. The problem here is that `spaces` is run before `ab`. Spaces will always successfully parse. So in the presence of spaces, we will always have an error occurring at a stream position ahead of where `many` returns to. That is, `many` returns to where the first space occurs, but the error will be after this.

We can keep this from having by putting the `spaces` parsing behind the parsing of `ab`.
``` haskell
    putStrLn $  "\nParsing 'ab ab ab CD' with\n"
             ++ "(do\n"
             ++ "      spaces\n"
             ++ "      many $ try (ab >> spaces)\n"
             ++ "      char 'C'\n"
             ++ "      char 'D'\n"
             ++ ")"
    -- Will succesfully parse.
    parseTest ( do
                    spaces
                    many $ try (ab >> spaces) 
                    char 'C'
                    char 'D'
              )
              "ab ab ab CD" 
```
When we run this, we get the following output
```
Parsing 'ab ab ab CD' with
(do
      spaces
      many $ try (ab >> spaces)
      char 'C'
      char 'D'
)
'D'
```

So the parser actually parses it correctly. Now let's try it on a different input, one that will result in failure.
``` haskell
    putStrLn $  "\nParsing 'ab ab ab aCD' with\n"
             ++ "(do\n"
             ++ "      spaces\n"
             ++ "      many $ try (ab >> spaces)\n"
             ++ "      char 'C'\n"
             ++ "      char 'D'\n"
             ++ ")"
    -- Will fail to parse at 'C', but this time it is possible that the sub-string " aCD"
    -- was supposed to be " abCD" or " CD". So there is no midirection, and the user 
    -- should be able to decide which. 
    parseTest ( do
                    spaces
                    many $ try (ab >> spaces) 
                    char 'C'
                    char 'D'
              )
              "ab ab ab aCD" 

```
We get the following output:
```
Parsing 'ab ab ab aCD' with
(do
      spaces
      many $ try (ab >> spaces)
      char 'C'
      char 'D'
)
parse error at (line 1, column 11):
unexpected "C"
expecting "b"
```

So now we get an error at 'C' where the parser says it is expecting 'b'. Now, the parser didn't really fail looking for a 'b'. It failed when it tried to parse `char 'C'` at character 'a' in the sub-string "aCD". However, the error message isn't a misdirection. It is reasonable that the input sub-string "aCD" should be either "abCD" or just "CD". The user can reasonable decide which it should be.

## [Download the Source Code for this Post]({{site . url}}/assets/2017-12-08-ParsecPitfall.hs)
