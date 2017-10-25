import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

#sns.heatmap(shapes, vmin = 0.0, vmax = 1.0)
sns.heatmap(shapes)
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

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
# 
#     losses = []
#     for i in range(2500):
#         sess.run(train_step, {x: features, y_: shapes})
#         newloss = sess.run(l2_loss, {x: features, y_:shapes})
#         losses.append(newloss) 
# 
#     result = sess.run(y, {x: features})
#     print('[W, b] = ', sess.run([W, b]))
# 
# 
#     
# plt.plot(losses[1000:])
# plt.show()
# 
# print('result.shape = ', result.shape)
# print(shapes.shape)
# 
# result = result.reshape(nY, nX)
# sns.heatmap(result)
# plt.show()
# 
# model = LinearRegression()
# model.fit(features, shapes)
# print('regression coef = ', model.coef_)
# print('intercept = ', model.intercept_)
# result = model.predict(features).reshape((nY, nX))
# sns.heatmap(result)
# plt.show()

# Now add to the tensorflow graph.
hiddensize1 = 50

W1 = tf.Variable(tf.random_normal([2, hiddensize1], stddev = 0.25) )
b1 = tf.Variable(tf.zeros([hiddensize1]) )
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

Wout = tf.Variable(tf.random_normal([hiddensize1, 1], stddev = 0.25) ) 
bout = tf.Variable(tf.zeros([1]) ) 
out = tf.matmul(hidden1, Wout) + bout 

l2_loss = tf.reduce_mean(tf.square(out - y_))
train_step = tf.train.GradientDescentOptimizer(5e-2).minimize(l2_loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
# 
#     losses = []
#     for i in range(3000):
#         batchi = np.random.randint(0, len(shapes), size = 200)
#         batchx = features[batchi]
#         batchy = shapes[batchi]
#         sess.run(train_step, {x: batchx, y_: batchy, keep_prob: 0.5})
#         newloss = sess.run(l2_loss, {x: features, y_:shapes, keep_prob: 0.5})
#         losses.append(newloss) 
#     
#     result = sess.run(out, {x: features, keep_prob:1.0})
# 
# plt.plot(losses[100:])
# plt.show()
# 
# result = result.reshape(nY, nX)
# sns.heatmap(result)
# plt.show()

# Do again for adam optimizer

# train_step = tf.train.AdamOptimizer(1e-2).minimize(l2_loss)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
# 
#     losses = []
#     for i in range(3000):
#         batchi = np.random.randint(0, len(shapes), size = 200)
#         batchx = features[batchi]
#         batchy = shapes[batchi]
#         sess.run(train_step, {x: batchx, y_: batchy, keep_prob: 0.5})
#         newloss = sess.run(l2_loss, {x: features, y_:shapes, keep_prob: 0.5})
#         losses.append(newloss) 
#     
#     result = sess.run(out, {x: features, keep_prob: 1.0})
# 
# plt.plot(losses[100:])
# plt.show()
# 
# result = result.reshape(nY, nX)
# sns.heatmap(result)
# plt.show()


# Now let's add another layer.

# tf.reset_default_graph()
# 
# hiddensize1 = 50
hiddensize2 = 50

# x = tf.placeholder(tf.float32, shape = [None, 2]) 
# y_ = tf.placeholder(tf.float32, shape = [None, 1]) 
# 
# W1 = tf.Variable(tf.random_normal([2, hiddensize1], stddev = 0.25) ) 
# b1 = tf.Variable(tf.zeros([hiddensize1]) )
# hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_normal([hiddensize1, hiddensize2], stddev = 0.25) )
b2 = tf.Variable(tf.zeros(hiddensize2) )
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

Wout = tf.Variable(tf.random_normal([hiddensize2, 1], stddev = 0.25) )
bout = tf.Variable(tf.zeros([1]))
out = tf.matmul(hidden2, Wout) + bout 

# We redefined the node out, so we need to redefine the loss function and the training step
l2_reg = tf.contrib.layers.l2_regularizer(scale = 1e-4)
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, W2, Wout])

l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term
train_step = tf.train.GradientDescentOptimizer(2e-1).minimize(l2_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    losses = []
    for i in range(3000):
        batchi = np.random.randint(0, len(shapes), size = 200)
        batchx = features[batchi]
        batchy = shapes[batchi]
        sess.run(train_step, {x: batchx, y_: batchy})
        if i % 100 == 0:
            newloss = sess.run(l2_loss, {x: features, y_:shapes})
            losses.append(newloss) 
            print(i, ' loss = ', newloss)
    
    result = sess.run(out, {x: features})

plt.plot(losses)
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show()

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-2
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step,
                                           500, 0.95, staircase = True)

train_step = tf.train.AdamOptimizer(1e-2).minimize(l2_loss, global_step = global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    losses = []
    for i in range(5000):
        batchi = np.random.randint(len(shapes), size = 200)
        batchx = features[batchi]
        batchy = shapes[batchi]
        sess.run(train_step, {x: batchx, y_: batchy})
        if i % 100 == 0:
            newloss = sess.run(l2_loss, {x: features, y_:shapes})
            losses.append(newloss) 
            print(i, ' loss = ', newloss)

    
    result = sess.run(out, {x: features})

plt.plot(losses)
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show()

# Add a third hidden layer.
hiddensize3 = 50
W3 = tf.Variable(tf.random_normal([hiddensize2, hiddensize3], stddev = 0.25) )
b3 = tf.Variable(tf.zeros(hiddensize3) )
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

Wout = tf.Variable(tf.random_normal([hiddensize3, 1], stddev = 0.25) )
bout = tf.Variable(tf.zeros([1]) )
out = tf.matmul(hidden3, Wout) + bout 

# We redefined the node out, so we need to redefine the loss function and the training step
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, W2, W3, Wout])
l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-2
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step,
                                           500, 0.95, staircase = True)


train_step = tf.train.AdamOptimizer(1e-2).minimize(l2_loss, global_step = global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    losses = []
    for i in range(3000):
        batchi = np.random.randint(len(shapes), size = 200)
        batchx = features[batchi]
        batchy = shapes[batchi]
        sess.run(train_step, {x: batchx, y_: batchy})
        if i % 100 == 0:
            newloss = sess.run(l2_loss, {x: features, y_:shapes})
            losses.append(newloss) 
            print(i, ' loss = ', newloss)
    
    result = sess.run(out, {x: features})

plt.plot(losses)
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show()
