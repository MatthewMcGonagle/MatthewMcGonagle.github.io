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
def layer1_node1(x, y):
    return logistic(-x +  35)
def layer1_node2(x, y):
    return logistic(y - 65)
def layer1_node3(x, y):
    return logistic(x - 65)
def layer1_node4(x, y):
    return logistic(-y + 35)
```

Now let's consider taking the average of two logistic functions.
``` python
z = 1/2.0 * layer1_node2(X,Y) + 1/2.0 * layer1_node3(X,Y) 

plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Two Logistic Functions')
plt.savefig('2017-10-17-graphs/manual_twolog.png')
```
![Picture of Two Logistics]({{site . url}}/assets/2017-10-17-graphs/manual_twolog.png)

Here we observe a problem for using simple combinations of logistic functions (i.e. one hidden layer). They are incapable of learning a simple bounded geometric object; that is, the combination must be non-desirable values outside the shape. For example, let's try combining four logistic functions to make a square. 
``` python
z = 0.5 * layer1_node1(X,Y) + 0.5 * layer1_node2(X,Y) + 0.5 * layer1_node3(X,Y) + 0.5 * layer1_node4(X,Y)

plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Four Logistic Functions')
plt.savefig('2017-10-17-graphs/manual_fourlog.png')
```
![Picture of Four Logistics]({{site . url}}/assets/2017-10-17-graphs/manual_fourlog.png)

What we want is a square of 0 values in the center and all other values are close to 1. However, we see that there are large sections outside the square where we get other values. To fix this, we will need to add another hidden layer.

## Manually Adding a Second Hidden Layer

Now let's consider what happends when we manually add in a second hidden layer. First let's look at learning a corner shape. Before when we looked at a combination of two logistic functions, we had values outside the corner that weren't the desired value of 1. Now we may compose this (and an additional offset and scaling) with a logistic function to force all values outside the corner to be 1.
``` python
def layer2_node1(x, y):
    result = layer1_node2(X,Y) + layer1_node3(X,Y) - 1/2
    return logistic( result * 10)

z = layer2_node1(X,Y) 
plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Hidden Layer Sizes is (2, 1)')
plt.savefig('2017-10-17-graphs/manual_hidden(2,1).png')
```
![Picture of Corner with Manual Two Hidden Layers]({{site . url}}/assets/2017-10-17-graphs/manual_hidden(2,1).png)

We can also do something similar to obtain a square.
``` python
def layer2_node2(x, y):
    result = layer1_node1(X,Y) + layer1_node2(X,Y) + layer1_node3(X,Y) + layer1_node4(X,Y) - 1/2
    return logistic( result * 10)

z = layer2_node2(X, Y)
plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Hidden Layer Sizes is (4, 1)')
plt.savefig('2017-10-17-graphs/manual_hidden(4,1).png')
```
![Picture of Square with Manual Two Hidden Layers]({{site . url}}/assets/2017-10-17-graphs/manual_hidden(4,1).png)

In fact, we can now construct more complicated shapes. For example, we can make another larger square and then subtract it from our previous square.
``` python
def layer1_node5(x, y):
    return logistic(-x +  25)
def layer1_node6(x, y):
    return logistic(y - 75)
def layer1_node7(x, y):
    return logistic(x - 75)
def layer1_node8(x, y):
    return logistic(-y + 25)

def layer2_node3(x, y):
    result = layer1_node5(X,Y) + layer1_node6(X,Y) + layer1_node7(X,Y) + layer1_node8(X,Y) - 1/2
    return logistic(result * 10)

z = -layer2_node3(X,Y) + layer2_node2(X,Y) 
plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Hidden Layer Sizes is (8, 2)')
plt.savefig('2017-10-17-graphs/manual_hidden(8,2).png')

```
![Picture of Complex Square with Manual Two Hidden Layers]({{site . url}}/assets/2017-10-17-graphs/manual_hidden(8,2).png)

Now, let's take a look at using scikit-learn's `MLPRegressor()` to actually try to learn these shapes.

## Learning a Square with MLPRegressor()


