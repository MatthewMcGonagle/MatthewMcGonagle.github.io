import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

np.random.seed(20171023)
tf.set_random_seed(20171023)

print('Creating Shapes')

nX = 100
nY = 100
X = np.full((nY, nX), np.arange(nX)) / nX 
Y = nY - 1 - np.full((nY, nX), np.arange(nY)[:, np.newaxis])
Y = Y / nY 

shapes = np.zeros((nY, nX)) 
mask = (0.35 < X) & (X < 0.65) & (0.35 < Y) & (Y < 0.65)
mask = (.25 < X) & (X < .75) & (.25 < Y) & (Y < .75) & ~mask
shapes[mask] = 1.0 # 1.0 
features = np.stack([X.reshape(-1),Y.reshape(-1)], axis = -1)
print('features.shape = ', features.shape)


def plotheatmap(heats):
    plt.clf()
    xticks = np.arange(0, 1, .1)
    yticks = np.arange(0, 1, .1)
    ax = sns.heatmap(heats, xticklabels = xticks, yticklabels = yticks[::-1])
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_yticks(np.arange(0, 100, 10))

plotheatmap(shapes)    
plt.title('Original Shape')
plt.show()
shapes = shapes.reshape(-1, 1)

print('shapes.shape = ', shapes.shape)

# Create tensorflow graph.

# Set up features placeholder x and training input placeholder y_.
# Feature dimension is 2 for position in image. The output dimension is 1 for the color of the pixel.

x = tf.placeholder(tf.float32, shape = [None, 2]) 
y_ = tf.placeholder(tf.float32, shape = [None, 1]) 

W = tf.Variable(tf.zeros([2, 1]) + 0.1, dtype = tf.float32)
b = tf.Variable(tf.zeros([1]) - 0.1, dtype = tf.float32)
y = tf.add(tf.matmul(x, W), b)
l2_loss = tf.reduce_mean(tf.square(y - y_)) 

train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(l2_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    losses = []
    lossindex = []
    for i in range(3000):
        sess.run(train_step, {x: features, y_: shapes})
        if i % 100 == 0:
            newloss = sess.run(l2_loss, {x: features, y_:shapes})
            losses.append(newloss) 
            lossindex.append(i)
            print(i, ' loss = ', newloss)

    result = sess.run(y, {x: features})
    weights = sess.run(W)
    bias = sess.run(b)
    
print('weights = ', weights)
print('bias = ', bias)

plt.clf()
plt.title('Losses For Gradient Descent of Linear Model')
plt.plot(lossindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Gradient Descent for Linear Model')
plt.show()

model = LinearRegression()
model.fit(features, shapes)
print('regression coef = ', model.coef_)
print('intercept = ', model.intercept_)
result = model.predict(features).reshape((nY, nX))
plotheatmap(result)
plt.title('Regular Linear Regression')
plt.show()

# Now let's make a graph with a hidden layer. 

hiddensize1 = 50
tf.reset_default_graph()
tf.set_random_seed(20171023)

x = tf.placeholder(tf.float32, shape = [None, 2]) 
y_ = tf.placeholder(tf.float32, shape = [None, 1]) 

W1 = tf.Variable(tf.random_normal([2, hiddensize1], stddev = 0.25) )
b1 = tf.Variable(tf.zeros([hiddensize1]) )
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

Wout = tf.Variable(tf.random_normal([hiddensize1, 1], stddev = 0.25) ) 
bout = tf.Variable(tf.zeros([1]) ) 
out = tf.matmul(hidden1, Wout) + bout 

l2_reg = tf.contrib.layers.l2_regularizer(scale = 1e-4)
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, Wout])

l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term
train_step = tf.train.GradientDescentOptimizer(20e-2).minimize(l2_loss)

def runSession(nsteps):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        losses = []
        lossesindex = []
        for i in range(nsteps):
            batchi = np.random.randint(0, len(shapes), size = 200)
            batchx = features[batchi]
            batchy = shapes[batchi]
            sess.run(train_step, {x: batchx, y_: batchy})
            if i % 100 == 0:
                newloss = sess.run(l2_loss, {x: features, y_:shapes})
                losses.append(newloss) 
                lossesindex.append(i)
                print(i, ' loss = ', newloss)
        
        result = sess.run(out, {x: features})

    return lossesindex, losses, result

lossesindex, losses, result = runSession(3000)

plt.title('Losses for Gradient Descent of 1 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Gradient Descent for 1 Hidden')
plt.show()

# Do again for adam optimizer

train_step = tf.train.AdamOptimizer(3e-3).minimize(l2_loss)
lossesindex, losses, result = runSession(3000)

plt.title('Losses for Adam of 1 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Adam for 1 Hidden')
plt.show()

# Now let's add another layer.

hiddensize2 = 50

W2 = tf.Variable(tf.random_normal([hiddensize1, hiddensize2], stddev = 0.25) )
b2 = tf.Variable(tf.zeros(hiddensize2) )
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

Wout = tf.Variable(tf.random_normal([hiddensize2, 1], stddev = 0.25) )
bout = tf.Variable(tf.zeros([1]))
out = tf.matmul(hidden2, Wout) + bout 

# We redefined the node out, so we need to redefine the regularity function, 
# the loss function, and the training step.

l2_reg = tf.contrib.layers.l2_regularizer(scale = 1e-4)
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, W2, Wout])

l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term
train_step = tf.train.GradientDescentOptimizer(10e-2).minimize(l2_loss)

lossesindex, losses, result = runSession(6000)

plt.title('Losses for Gradient Descent of 2 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Gradient Descent for 2 Hidden')
plt.show()

train_step = tf.train.AdamOptimizer(3e-3).minimize(l2_loss)
lossesindex, losses, result = runSession(5000)

plt.title('Losses for Adam for 2 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Adam for 2 Hidden Layers')
plt.show()

# Add a third hidden layer.

hiddensize3 = 50
W3 = tf.Variable(tf.random_normal([hiddensize2, hiddensize3], stddev = 0.25) )
b3 = tf.Variable(tf.zeros(hiddensize3) )
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

Wout = tf.Variable(tf.random_normal([hiddensize3, 1], stddev = 0.25) )
bout = tf.Variable(tf.zeros([1]) )
out = tf.matmul(hidden3, Wout) + bout 

# We redefined the node out, so we need to redefine the regularity function, the loss function,
# and the training step

reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, W2, W3, Wout])
l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term

train_step = tf.train.GradientDescentOptimizer(15e-2).minimize(l2_loss)

lossesindex, losses, result = runSession(6000)

plt.title('Losses for Gradient Descent of 3 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Gradient Descent for 3 Hidden Layers')
plt.show()

train_step = tf.train.AdamOptimizer(2e-3).minimize(l2_loss)

lossesindex, losses, result = runSession(5000)

plt.title('Losses for Adam of 3 Hidden')
plt.plot(lossesindex[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
plotheatmap(result)
plt.title('Adam for 3 Hidden Layers')
plt.show()