---
format: post
date: 2019-07-24
title: Local Minimum For TSP Annealing
tags: Python MonteCarlo Optimization
---

{% capture local_min_py %}
    {% include 2019-07-25-files/local_minimum.py %}
{% endcapture %}
{% assign local_min_lines = local_min_py | newline_to_br | split: '<br />' %}

{% capture my_src_py %}
    {% include 2019-07-25-files/my_src.py %}
{% endcapture %}
{% assign my_src_lines = my_src_py | newline_to_br | split: '<br />' %}

## [Download local_minimum.py Here]({{ site . url}}/assets/2019-07-25-files/local_minimum.py)
## [Download my_src.py Here]({{ site . url}}/assets/2019-07-25-files/my_src.py)

In this post we will look at an example of a cycle that is a local minimum for the moves that are allowed
for the standard simulated annealing approach to the Travelling Salesman Problem 
([see this post]({{site . url}}/blog/2018/04/15/TSPArtWithAnnealing.html)). Note that the Travelling
Salesman Problem is actually a convex integer programming problem; so how do we encounter a local
minimum? The answer is that in some sense the simulated annealing process relies on moving in "directions"
of the cycle state space that are limited enough that relative to these restricted moves, we encounter
local minima. Now, actually finding an explicit example of how this occurs is a little tricky.

In particular, we show that the following cycle is a local minimum:

![Local Minimum Cycle]({{site . url}}/assets/2019-07-25-files/local_min.svg)

The dimensions of this cycle have been carefully chosen to give a local minimum;
e.g. if we chose the top vertices to have a
y-value of 3.0, then the cycle is no longer a local minimum.

Note, that this cycle is not a global minimum. In fact, the following cycle actually has smaller
length:

![Shorter Cycle]({{site . url}}/assets/2019-07-25-files/shorter_cycle.svg)

However, it takes MORE than one of our simulated annealing flips to go from the local minimum to
the smaller cycle.

In the following sections, we will use a brute force check to verify that the local minimum is in fact
a local minimum. That is, we will check every cycle that is obtained from our local minimum by one flip
has a cycle length at least as large as the local minimum. 

# Creating the Local Minimum

Now let's look at creating the local minimum. We use the following from 'my_src.py'.

{% highlight python %}
# From my_src.py.
{% for line in my_src_lines offset: 4 limit: 62 %}{{ line }}{% endfor %}
{% endhighlight %}

Let's get the local minimum and the shorter cycle.

{% highlight python %}
# From local_minimum.py.
{% for line in local_min_lines offset: 10 limit: 19 %}{{ line }}{% endfor %}
{% endhighlight %}

We saw the graphs of the two cycles and their lengths at the beginning of the post.

# Verify the Local Min is a Local Min

Now let's verify that the local minimum is actually a local minimum. We will simply compute
every possible flip, and find its length. For example, here is a graph of random flip of
the local minimum:

![A Random Local Move]({{site . url}}/assets/2019-07-25-files/local_move.svg)

Let's now get all of the lengths of the local moves. We will be making use of the following
function from `my_src.py`:

{% highlight python %}
# From my_src.py.
{% for line in my_src_lines offset:91 limit:19 %}{{ line }} {% endfor %}
{% endhighlight %}

Let's now get the lengths of the local moves, and let's make some graphs of some interesting
examples. Also, let's raise an exception if we see any local move that has a length less than
that of the local minimum.

{% highlight python %}
# From local_minimum.py
{% for line in local_min_lines offset: 30 limit:51 %}{{ line }}{% endfor %}
{% endhighlight %}

We get the following output:
{% highlight text %}
{% include 2019-07-25-files/output.txt %}
{% endhighlight %}

So we see that all of the local moves are at least as long as the local minimum. Some have the same
length; graphing these we infact see that these cycles have the same shape but are an order preserving
permutation of the original vertices. Here are their graphs:

![Example one of Same Length]({{ site . url }}/assets/2019-07-25-files/same_0_6.svg)
![Example one of Same Length]({{ site . url }}/assets/2019-07-25-files/same_0_7.svg)
![Example one of Same Length]({{ site . url }}/assets/2019-07-25-files/same_1_7.svg)

Now, let's take a look at all of the local move lengths.
![All Local Move Lengths]({{ site . url }}/assets/2019-07-25-files/local_lengths.svg)

So we see that the local minimum is in fact a local minimum when we restrict to the flips
used for simulated annealing in the travelling salesman problem.

## [Download local_minimum.py Here]({{ site . url}}/assets/2019-07-25-files/local_minimum.py)
## [Download my_src.py Here]({{ site . url}}/assets/2019-07-25-files/my_src.py)

