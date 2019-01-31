---
layout: post
date: 2019-01-25
title: Using a CUDA GPU to do Jacobi Iterations
source: assets/2019-01-25-files
tags: C++ CUDA GPU NumericalAnalysis
---

{% capture source %}{{site . url}}/{{page . source}}{% endcapture %}

{% capture makeContent %}
{% include 2019-01-25-files/makefile %}
{% endcapture %}

{% capture exampleContent %}
{% include 2019-01-25-files/example.cu %}
{% endcapture %}
{% assign exampleLines = exampleContent | newline_to_br | split: '<br />' %} 

## [Download Source Files Here (tar.gz)]({{ site . url}}/assets/tarballs/2019-01-25-CUDAJacobi.tar.gz)

In this post we will look at using CUDA programming to get a GPU to compute Jacobi iterations for approximations
to harmonic functions satisfying Dirichlet boundary conditions (i.e. specifying the values of the function along
the boundary). Our domain will simply be the unit square in the xy-plane. Using a GPU will allow us to implement
every step of a single iteration in parallel.

As an example, here are the results for a PDE discretization grid of size 20x20.

![Values Surface]({{ source }}/graphs/Values_surface_20_20.svg)
![Values Heatmap]({{ source }}/graphs/Values_heat_20_20.svg)

Let's take a look at the errors and the logarithms base 10 of the relative errors.
![Errors Heatmap]({{ source }}/graphs/Errors_heat_20_20.svg)
![Log10 Relative Errors Heatmap]({{ source }}/graphs/Log10RelErrors_heat_20_20.svg)

For this post we will be making use of software inaddition to compiling C++.
* For making CUDA code:
    * We need a CUDA capable device, e.g. a CUDA capable GPU.
    * [CUDA compiler (NVCC)](https://developer.nvidia.com/cuda-llvm-compiler).
    * Something to run GNUMake (e.g. on windows, a [MinGW Shell](http://www.mingw.org/wiki/getting_started)).
* For graphing our results:
    * [GNUPlot](http://www.gnuplot.info/) to make graphs. We will need to be able to run it from a terminal.
    * Something to run Bash scripts (e.g. on windows, a [MinGW Shell](http://www.mingw.org/wiki/getting_started)).

# Building and Running the Code

We can simply use our `makefile` to build the CUDA C++ code.
{% highlight makefile %}
{{ makeContent }}
{% endhighlight %}

Our directory structure will be: 
* Main Directory
    * `data/`
    * `graphs/`
    * `makefile`
    * `example.cu`
    * `jacobi.cuh`
    * `jacobi.cu`
    * `fromFile.p`
    * `makeGraphs.sh` 

To use it we simply run `make` from a terminal from the main directory.

To use CUDA to find the approximate function on a 20x20 grid, we simply need to run the following
from the main directory.
``` terminal
example.exe 20
```
We get the following output:
```
Making initial values and true values
Before Average Error = 0.00788951
Copying to Device
Doing Jacobi Iterations
Copying result to values
Copying to file 'values.dat'
Now getting errors
After Average Error = 0.000176011
Now getting relative errors
```

Then, to make the graphs, we need to run the Bash script:
```
sh makeGraphs.sh 20
```

# Reviewing `example.cu`

Now let's review `example.cu`. First let's include headers we will need.

{% highlight cpp %}
{% for line in exampleLines offset: 1 limit: 13 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, let's define the harmonic function that we will use to assign our boundary values and check
the validity of our results.
{% highlight cpp %}
{% for line in exampleLines offset: 14 limit: 11 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, let's deal with the parameters sent to `main()`.
{% highlight cpp %}
{% for line in exampleLines offset: 25 limit: 18 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, let's define the variables we will need.
{% highlight cpp %}
{% for line in exampleLines offset: 44 limit: 14 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, let's setup the intial values (including boundary values), find the true values, and find the
initial average error.
{% highlight cpp %}
{% for line in exampleLines offset: 59 limit: 10 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, we need to copy the initial values to the CUDA device.
{% highlight cpp %}
{% for line in exampleLines offset: 70 limit: 10 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, we use the host device to tell the CUDA device to do each iteration. We simply synchronize between
each iteration.
{% highlight cpp %}
{% for line in exampleLines offset: 81 limit: 15 %}{{ line }}{% endfor %}
{% endhighlight %}

Next, let's get the results and save the relevant data to file.
{% highlight cpp %}
{% for line in exampleLines offset: 97 limit: 23 %}{{ line }}{% endfor %}
{% endhighlight %}

Finally let's clean up and exit.
{% highlight cpp %}
{% for line in exampleLines offset: 121 limit: 11 %}{{ line }}{% endfor %}
{% endhighlight %}

# Graphing the Results

We will use the GNUPlot script `fromFile.p` and the Bash script `makeGraphs.sh`. First, let's take a look
at the GNUPlot script.
{% highlight gnuplot %}
{% include 2019-01-25-files/fromFile.p %}
{% endhighlight %}

Next, we use a Bash script to run the GNUPlot script on each data file.  
{% highlight gnuplot %}
{% include 2019-01-25-files/makeGraphs.sh %}
{% endhighlight %}
This makes graphs including those at the beginning of this post.

# Reviewing `jacobi.cuh`

Next, let's take a look at the functions we used in `jacobi.cuh`.
{% highlight cpp %}
{% include 2019-01-25-files/jacobi.cuh %}
{% endhighlight %}

# Reviewing `jacobi.cu`

Next, let's see how the functions are implemented.
{% highlight cpp %}
{% include 2019-01-25-files/jacobi.cu %}
{% endhighlight %}

## [Download Source Files Here (tar.gz)]({{ site . url}}/assets/tarballs/2019-01-25-CUDAJacobi.tar.gz)

