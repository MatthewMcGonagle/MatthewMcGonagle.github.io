---
layout: post
date: 2019-01-25
title: Using a CUDA GPU to do Jacobi Iterations
source: assets/2019-01-25-files
tags: C++ CUDA GPU NumericalAnalysis
---

{% capture source %}{{site . url}}/{{page . source}}{% endcapture %}

## [Download example.cu Here]({{ source }}/example.cu)
## [Download jacobi.cuh Here]({{ source }}/jacobi.cuh)
## [Download jacobi.cu Here]({{ source }}/jacobi.cu)
## [Download fromFile.p Here]({{ source }}/fromFile.p)
## [Download makeGraphs.sh Here]({{ source }}/makeGraphs.sh) 
## [Download makefile Here]({{ source }}/makefile)

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
    * [GNUPlot](http://www.gnuplot.info/) to make graphs.
    * Something to run Bash scripts (e.g. on windows, a [MinGW Shell](http://www.mingw.org/wiki/getting_started)).

# Building the Code

{% capture makeContent %}
{% include 2019-01-25-files/makefile %}
{% endcapture %}

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

## [Download example.cu Here]({{ source }}/example.cu)
## [Download jacobi.cuh Here]({{ source }}/jacobi.cuh)
## [Download jacobi.cu Here]({{ source }}/jacobi.cu)
## [Download fromFile.p Here]({{ source }}/fromFile.p)
## [Download makeGraphs.sh Here]({{ source }}/makeGraphs.sh) 
## [Download makefile Here]({{ source }}/makefile)


