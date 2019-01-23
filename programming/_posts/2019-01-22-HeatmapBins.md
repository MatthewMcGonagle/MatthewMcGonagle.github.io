---
layout: post
date: 2019-01-22
title: Heatmap of Mean Values in 2D Histogram Bins
tags: Pandas Python
---

{% capture codeContent %}
    {% include 2019-01-22-files/heatmapBins.py %}
{% endcapture %}
{% assign codeLines = codeContent | newline_to_br | split: '<br />' %}

{% capture outputContent %}
    {% include 2019-01-22-files/output.txt %}
{% endcapture %}
{% assign outputLines = outputContent | newline_to_br | split: '<br />' %}

## [Download heatmapBins.py Here]({{ site . url }}/assets/2019-01-22-files/heatmapBins.py)

In this post we will look at how to use the `pandas` python module and the `seaborn` python module to 
create a heatmap of the mean values of a response variable for 2-dimensional bins from a histogram.

The final product will be

![Final heatmap of z vs Features 0 and 1]({{ site . url}}/assets/2019-01-22-files/graphs/means2.svg)

Let's get started by including the modules we will need in our example.

{% highlight python %}
{% for line in codeLines offset: 6 limit: 8 %}{{ line }}{% endfor %}
{% endhighlight %}

# Simulate Data

Now, we simulate some data. We will have two features, which are both pulled from normalized gaussians. The 
response variable `z` will simply be a linear function of the features: `z = x - y`.

{% highlight python %}
{% for line in codeLines offset: 15 limit: 12 %}{{ line }}{% endfor %}
{% endhighlight %}

Here is the output of the data's information.
{% highlight text %}
{% for line in outputLines offset: 2 limit: 9 %}{{ line }}{% endfor %}
{% endhighlight %}

Let's take a look at a scatter plot.

{% highlight python %}
{% for line in codeLines offset: 28 limit: 7 %}{{ line }}{% endfor %}
{% endhighlight %}

Here is the output.

![Scatter Plot of Features]({{site . url}}/assets/2019-01-22-files/graphs/features.svg)

Let's also take a look at a density plot using `seaborn`.

{% highlight python %}
{% for line in codeLines offset: 36 limit: 5 %}{{ line }}{% endfor %}
{% endhighlight %}

Here is the output.

![Density Plot of Features]({{site . url}}/assets/2019-01-22-files/graphs/density.svg)

# Make Cuts for Using Pandas Groupby

Next, let us use `pandas.cut()` to make cuts for our 2d bins.

{% highlight python %}
{% for line in codeLines offset: 42 limit: 4 %}{{ line }}{% endfor %}
{% endhighlight %}

The bin values are of type `pandas.IntervalIndex`. Here is the head of the `cuts` dataframe.
{% highlight text %}
{% for line in outputLines offset: 12 limit: 6 %}{{ line }}{% endfor %}
{% endhighlight %}

Here is the information on the `cuts` dataframe.
{% highlight text %}
{% for line in outputLines offset: 18 limit: 8 %}{{ line }}{% endfor %}
{% endhighlight %}

Note, that the types of the bins are labeled as `category`, but one should use methods from `pandas.IntervalIndex`
to work with them.

# Do the Groupby and Make Heatmap

Now, let's find the mean of `z` for each 2d feature bin; we will be doing a groupby using both of the bins
for Feature 0 and Feature 1.

{% highlight python %}
{% for line in codeLines offset: 49 limit: 7 %}{{ line }}{% endfor %}
{% endhighlight %}

Let's take a look at the head of `means`.
{% highlight text %}
{% for line in outputLines offset: 34 limit: 10 %}{{ line }}{% endfor %}
{% endhighlight %}

As we an see, we need to specify `means['z']` to get the means of the response variable `z`. This gives
{% highlight text %}
{% for line in outputLines offset: 44 limit: 14 %}{{ line }}{% endfor %}
{% endhighlight %}

Let's now graph a heatmap for the means of `z`.
{% highlight python %}
{% for line in codeLines offset: 57 limit: 5 %}{{ line }}{% endfor %}
{% endhighlight %}

This gives the graph:

![Heatmap with Interval Labels]({{ site.url }}/assets/2019-01-22-files/graphs/means1.svg)

As we can see, the x and y labels are intervals; this makes the graph look cluttered. Let us
now use the left endpoint of each interval as a label. We will use `pandas.IntervalIndex.left`.
{% highlight python %}
{% for line in codeLines offset: 63 limit: 6 %}{{ line }}{% endfor %}
{% endhighlight %}

This gives our final graph:

![Final graph]({{ site.url }}/assets/2019-01-22-files/graphs/means2.svg)

## [Download heatmapBins.py Here]({{ site . url }}/assets/2019-01-22-files/heatmapBins.py)

