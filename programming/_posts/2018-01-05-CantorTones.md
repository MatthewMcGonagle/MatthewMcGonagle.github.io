---
layout: post
title: Making an Audio .wav File of Cantor Tones
data: 2018-01-05
---

## [Download the Source Code for this Post]({{site . url}}/assets/2018-01-05-CantorTones.py) 

In this post we will use the `wave.py` module to create audio .wav files that contain tones arranged in the pattern of the iterates of a Cantor set. A good reference for using `wave.py` is [this stack overflow answer](https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps/33913403#33913403), but this post is designed to be self-contained. 
If you want to learn more about the format of .wav files, then [this reference from McGill University](http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html) may be helpful. However, it isn't really necessary for you to know the details of the .wav file format to use the `wave.py` module and follow this post.

First, we will review the Cantor set and the iterations that go into its construction. Then we will discuss the code to make the .wav file containing the cantor arrangement of tones.  

## The Cantor Set

![Pic of Cantor Set]({{site . url}}/assets/2018-01-05-CantorSet.svg)

![Cantor Tones for C Chord]({{site . url}}/assets/2018-01-05-CantorCChord.svg)

## [Download the Source Code for this Post]({{site . url}}/assets/2018-01-05-CantorTones.py)
