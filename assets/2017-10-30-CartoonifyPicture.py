import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys
from sklearn.preprocessing import StandardScaler

# Open the image we will try to turn into a cartoon.

img = Image.open('HilbertPic.jpg')
img = np.asarray(img, dtype = 'int32')
print('img.shape = ', img.shape)

# Function to turn image into an image of discrete number of grayscale levels.

def stratify(X, levels):
    nLevels = len(levels)
    X[X < levels[0]] = levels[0]
    for i in range(nLevels - 1):
        mask = (X > levels[i]) & (X < levels[i+1])
        X[mask] = levels[i]
    X[X > levels[nLevels - 1]] = levels[nLevels - 1]
    return X

stratlevels = np.arange(0, 256, int(256 / 8))
stratimg = stratify(img, stratlevels)
plt.imshow(255 - stratimg)
plt.title('Simple Discrete Levels')
plt.show()

# Set up features for X and Y coordinates.

nY, nX = img.shape
X = np.full(img.shape, np.arange(nX))
X = X / nX
Y = np.full(img.shape, np.arange(nY)[:, np.newaxis])
Y = nY - 1 - Y
Y = Y / nY

img = img.reshape(-1, 1)
print('img.shape = ', img.shape)

# First let's just try learning the picture using the X and Y coordinates as features.

features = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis = -1)

nSamples, nFeatures = features.shape
print('features.shape = ', features.shape)

# Set up layers. 

print('Setting up layers.')

tf.set_random_seed(20171026)
np.random.seed(20171026)

init_stddev = 0.5
init_mean = 0.0
init_bias = 0.1
hidden1size = 50
hidden2size = 50
hidden3size = 50

x = tf.placeholder(tf.float32, shape = [None, nFeatures])
y_ = tf.placeholder(tf.float32, shape = [None, 1])

W1 = tf.Variable(tf.truncated_normal([nFeatures, hidden1size], stddev = init_stddev, mean = init_mean) )
b1 = tf.Variable(tf.zeros([hidden1size]) + init_bias)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hidden1size, hidden2size], stddev = init_stddev, mean = init_mean))
b2 = tf.Variable(tf.zeros([hidden2size]) + init_bias)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2) 

W3 = tf.Variable(tf.truncated_normal([hidden2size, hidden3size], stddev = init_stddev, mean = init_mean))
b3 = tf.Variable(tf.zeros([hidden3size]) + init_bias)
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

Wout = tf.Variable(tf.truncated_normal([hidden3size, 1], stddev = init_stddev, mean = init_mean))
bout = tf.Variable(tf.zeros([1]) + init_bias)
out = tf.matmul(hidden3, Wout) + bout

# Set up regularizer and loss.

print('Setting up loss.')

l2_loss = tf.reduce_mean(tf.square(out - y_))
train_step = tf.train.AdamOptimizer(1e-2).minimize(l2_loss)

# Train network on image.

batchsize = 200

def runSession(nsteps):
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
    
        losses = []
        losses_index = []
        for i in range(nsteps):
            batchi = np.random.randint(0, len(img), size = batchsize)
            batchx = features[batchi]
            batchy = img[batchi]
            sess.run(train_step, {x: batchx, y_: batchy})
            if i % 100 == 0:
                newloss = sess.run(l2_loss, {x: features, y_: img})
                losses.append(newloss)
                losses_index.append(i)
                print(i, ' loss = ', newloss)
        
        result = sess.run(out, {x: features})
        result = 255 - result
        
        return losses_index, losses, result

losses_index, losses, result = runSession(10000)

plt.plot(losses_index[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show() 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
sns.heatmap(result)
plt.show() 

# Try decaying learning rate.

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

losses_index, losses, result = runSession(10000)

plt.plot(losses_index[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show() 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
sns.heatmap(result)
plt.show() 

def getNeighbors(X, nbrRadius):
    rowsX, colsX = X.shape
    cols = np.arange(-nbrRadius, nbrRadius + 1)[np.newaxis, :]
    rows = np.arange(-nbrRadius, nbrRadius + 1)[:, np.newaxis]
    cols = np.arange(colsX)[np.newaxis, :, np.newaxis, np.newaxis] + cols
    rows = np.arange(rowsX)[:, np.newaxis, np.newaxis, np.newaxis] + rows 
    cols = np.amax([cols, np.zeros(cols.shape)], axis = 0)
    rows = np.amax([rows, np.zeros(rows.shape)], axis = 0)
    cols = np.amin([cols, np.full(cols.shape, colsX - 1)], axis = 0)
    rows = np.amin([rows, np.full(rows.shape, rowsX - 1)], axis = 0)
    rows = rows.astype('int32')
    cols = cols.astype('int32')
  
    nbrs = X[rows, cols] 
    # mask = np.ones(nbrs.shape, dtype = bool)
    # mask[:, :, nbrRadius, nbrRadius] = 0
    # nbrs = nbrs[mask].reshape(rowsX, colsX, -1)
    nbrs = nbrs.reshape(rowsX, colsX, -1)
    rowsX, colsX, nNbrs = nbrs.shape
    nbrs = nbrs.reshape(-1, nNbrs)
    return nbrs

nbrs = getNeighbors(img, 4)
print('nbrs.shape = ', nbrs.shape)

nbr_weights= nbrs.std(axis = -1, keepdims = True)
nbr_weights = nbr_weights / np.amax(nbr_weights)

plt.imshow(nbr_weights.reshape(nY, nX))
plt.show()

features = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), nbr_weights.reshape(-1,1)], axis = -1)

nSamples, nFeatures = features.shape
print('features.shape = ', features.shape)

# Set up layers. 

print('Setting up layers.')
tf.reset_default_graph()
tf.set_random_seed(20171026)
np.random.seed(20171026)

init_stddev = 0.5
init_mean = 0.0
init_bias = 0.1
hidden1size = 50
hidden2size = 50
hidden3size = 50

x = tf.placeholder(tf.float32, shape = [None, nFeatures])
y_ = tf.placeholder(tf.float32, shape = [None, 1])

W1 = tf.Variable(tf.truncated_normal([nFeatures, hidden1size], stddev = init_stddev, mean = init_mean) )
b1 = tf.Variable(tf.zeros([hidden1size]) + init_bias)
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([hidden1size, hidden2size], stddev = init_stddev, mean = init_mean))
b2 = tf.Variable(tf.zeros([hidden2size]) + init_bias)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2) 

W3 = tf.Variable(tf.truncated_normal([hidden2size, hidden3size], stddev = init_stddev, mean = init_mean))
b3 = tf.Variable(tf.zeros([hidden3size]) + init_bias)
hidden3 = tf.nn.relu(tf.matmul(hidden2, W3) + b3)

Wout = tf.Variable(tf.truncated_normal([hidden3size, 1], stddev = init_stddev, mean = init_mean))
bout = tf.Variable(tf.zeros([1]) + init_bias)
out = tf.matmul(hidden3, Wout) + bout

# Set up loss.

print('Setting up loss.')

l2_loss = tf.reduce_mean(tf.square(out - y_)) 

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

losses_index, losses, result = runSession(10000)

plt.plot(losses_index[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show() 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
sns.heatmap(result)
plt.show() 

# Set up regularizer and loss.

print('Setting up loss.')

l2_reg = tf.contrib.layers.l2_regularizer(scale = 10**1.5) 
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1[2, :]]) 
l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

losses_index, losses, result = runSession(10000)

plt.plot(losses_index[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show() 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
sns.heatmap(result)
plt.show() 

