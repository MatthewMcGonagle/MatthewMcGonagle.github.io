---
layout : post
title : "Learning Geometric Shapes with a Multi-layer Perceptron"
date : 2017-10-17
---

For this post, we will look at how a Multi-layer Perceptron learns data by looking at how it learns simple geometric shapes.

We will be using the standard logistic activation function, `logistic(x) = 1 / (1 + exp(-x)),` or in python:
``` python
# Define Logistic Function
def logistic(x):
    return 1 / (1 + np.exp(-x))
```
 Here is a graph of the logistic function:

![Graph of Logistic Function]({{site . url}}/assets/2017-10-17-graphs/logistic.png)

As you can see, the logistic function flattens out for values far from zero.

However, we will be looking at functions of two variables f(x,y). First, let's set up our x-values and y-values in 2d arrays.
``` python
# Set up domain value arrays.
X = np.arange(100)[np.newaxis, :]
X = np.full((100, 100), X, dtype = 'float32')

Y = np.arange(100)[::-1, np.newaxis]
Y = np.full((100, 100), Y, dtype = 'float32')
```
Now, let's set up a logistic function for 2d variables that really only depends on x. We also give it an offset by 50 in the x-direction.
``` python
 def f(x, y):
    return logistic(x - 50) 
```
Now let's take a look at its graph as a heat map. We will be using the module `seaborn` and call it by the standard convention `sns`.
``` python
import seaborn as sns
```
Now print out the heat map.
``` python
z = f(X, Y) 

plt.clf()
xticks = np.arange(0, 100, 10)
yticks = np.arange(0, 100, 10)
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
```
![Picture of One Logistic]({{site . url}}/assets/2017-10-17-graphs/manual_onelog.png) 

Note that we are graphing from x = 0 to x = 100, and at this scale, the graph of looks like a sharp transition from 0 to 1 at x = 50.

## Manual Combinations of Logistic Functions 

First let's set up our different logistic functions.
``` python
def f1(x, y):
    return logistic(-x +  35)
def f2(x, y):
    return logistic(y - 65)
def f3(x, y):
    return logistic(x - 65)
def f4(x, y):
    return logistic(-y + 35)
```

Now let's consider taking the average of two logistic functions.
``` python
z = 1/2.0 * f2(X,Y) + 1/2.0 * f3(X,Y) 

plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Two Logistic Functions')
plt.savefig('2017-10-17-graphs/manual_twolog.png')
```
![Picture of Two Logistics]({{site . url}}/assets/2017-10-17-graphs/manual_twolog.png)
