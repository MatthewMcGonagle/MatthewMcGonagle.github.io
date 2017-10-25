---
layout: post
title: "Learning Simple Shapes in Tensor Flow"
data: 2017-10-23
---

In this post we will be doing something similar to the post ["Learning Geometric Shapes with a Multi-layer Perceptron"]({{site . url}}/blog/2017/10/17/LearningShapesMLP). In fact, we will also be using a multi-layer perceptron. However that post uses multi-layer perceptrons constructed using the `sklearn` module; in this post we contruct our multi-layer perceptrons using the `tensorflow` module. 
Furthermore, in the other post we used logistic functions for our activation functions, but now we will use ReLu activation functions. We will also be comparing results for mini-batch gradient descent optimization versus adam optimization.

Let us recall what we mean by "learning shapes." We will consider the following shape to be represented by a scalar valued function f(x,y).

![Picture of the Original Shape]({{site . url}}/assets/2017-10-23-graphs/original.png)

For points on the shape, the value of f(x,y) is `1.0` and for other points it is `0.0`. So we will be using a neural network to try to learn this function f(x,y). The best result is given by training a neural network of 3 hidden layers using adam optimization. In this case, the result we get is:

![Picture of Adam for 3 Hidden Layers]({{site . url}}/assets/2017-10-23-graphs/hidden3Adam.png)

Before we get started, let's import the following modules, set up our random seeds, and adjust our plot sizes.
```python
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

np.random.seed(20171023)
tf.set_random_seed(20171023)
fig = plt.figure(figsize = (3.5,3))

```
## Very Simple Tensor Flow Example : A Linear Model

In this section, we will take a look at a basic `tensorflow` example given by a simple linear model.
