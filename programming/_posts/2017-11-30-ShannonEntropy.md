---
layout: post
title: What is Shannon Entropy?
date: 2017/11/30
---

The purpose of this post is to record some of my musings on Shannon's viewpoint of entropy. This is based on two references:

* Claude Shannon's classic original paper "A Mathematical Theory of Communication". 
* The classic textbook "An Introduction to Probability and Random Processes" by Gian-Carlo Rota and Kenneth Baclawski, which at the time of this writing is [available in the open source collection on archive.org](https://archive.org/details/GianCarlo_Rota_and_Kenneth_Baclawski__An_Introduction_to_Probability_and_Random_Processes).

Shannon's view of entropy is framed in terms of sending signals from one party to another. For example, consider if the source wants to send messages consisting entirely of symbols from the set {A, B}. So the source will want to send stuff like "AABBA", "ABAA", "AAAAAAA", etc...
Furthermore, much like the English language has certain statistics for different letters, there are certain statistics for A and B. Let us consider the case that the probability that the symbol is A, P(A) = 3/4. So P(B) = 1/4.

Now, when we send our message, we will be using bits, each one is either 0 or 1. So, the question becomes on average, how many bits per symbol do we need to send? This is answered by the Shannon entropy. Now, you may be asking, the possible state of the symbol is either A or B, so don't we need to on average send 1 bit per symbol? (Recall that one bit represents two states) The answer is no, because if we consider encoding blocks of symbols then we can leverage the statistics of A and B to do better. Let us explore that in the next section.

## Example of Encoding Blocks of Symbols  

Recall that we are sending a message consisting of symbols from the two element set {A, B}; the statistics are P(A) = 3/4 and P(B) = 1/4. 

A first approach might be to send "0" when we wish to send "A" and send "1" to send "B". This is always 1 bit per symbol sent (not even just on average). However, we can do better if we consider encoding 3 symbols at once. For example, we could try using "00" to represent "AAA", and use "110" to represent "BAA". 
The main point is that "AAA" is the symbol block we send most often, so we will benefit from encoding it with less bits. For a real world example of variable length encoding, you need look no further than the short and long dashes of Morse code.

So let us try to create such an encoding. How should we proceed? Let us first write down some probabilities (these are simply computed using a binomial distribution).
* P(AAA) = 27/64
* P(2 A's and 1 B) = 27/64
* P(1 A and 2 B's) = 9/64
* P(BBB) = 1/64

So a large majority of our symbol blocks are from the set {AAA, AAB, ABA, BAA}. This is four states and can easily be encoded with two bits. However, we need to encode more states and need to know when our encoding is terminated. So we pick one of the 2A1B cases to leave out of the two bit encoding. So far we have:
* 00 for AAA
* 01 for AAB
* 10 for ABA
* 11 for inside set {BAA, BBA, BAB, ABB, BBB}.

Now we just need to decide how to encode the remaining possibilities inside {BAA, BBA, BAB, ABB, BBB}. This has 5 states, so we can't use only 2 bits. For some of the symbols we will need 3 bits. Since P(BAA) = 9/64 is larger than the probabilities of the others, let's use one bit to distinguish it from the rest. So:
* 110 for BAA
* 111 for inside set {BBA, BAB, ABB, BBB}.

The remaining set of possibilities {BBA, BAB, ABB, BBB} can now just be represented using 2 more bits. So to be clear, our final encoding is:
* 00 for AAA
* 01 for AAB
* 10 for ABA
* 110 for BAA
* 11100 for BBA
* 11101 for BAB
* 11110 for ABB
* 11111 for BBB

Let us compute how many bits on average are needed for this block of 3 symbols. First, we note that
* P(2 bits) = P({AAA, AAB, ABA}) = 27/64 + 2 * 9/64 = 45/64
* P(3 bits) = P(BAA) = 9/64
* P(5 bits) = P({BBA, BAB ABB, BBB}) = 9/64 + 1/64 = 10/64. 
* P(other bits) = 0

So, we get that the average number of bits is 2 * 45/64 + 3 * 9/64 + 5 * 10/64 bits; which is 167/64 = 2.61 bits. Since we are encoding 3 symbols at a time, this amounts to 0.87 bits per symbol on average! So we can beat 1 bit per symbol on average.

Our method for deriving our encoding seemed sort of ad hoc; can we do better?
