import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys
from sklearn.preprocessing import StandardScaler

img = Image.open('HilbertPic.jpg')
#img = Image.open('pitbull.jpg').convert('L')

img = np.asarray(img, dtype = 'int32')
print('img.shape = ', img.shape)

def stratify(X, levels):
    nLevels = len(levels)
    X[X < levels[0]] = levels[0]
    for i in range(nLevels - 1):
        mask = (X > levels[i]) & (X < levels[i+1])
        X[mask] = levels[i]
    X[X > levels[nLevels - 1]] = levels[nLevels - 1]
    return X

stratlevels = np.arange(0, 256, int(256 / 3))
stratimg = stratify(img, stratlevels)
plt.imshow(255 - stratimg)
plt.show()
stratimg = stratimg.reshape(-1, 1)
print('stratimg.shape = ', stratimg.shape)
nbr_weights = np.full(stratimg.shape, 1.0)

# Set up features.

nY, nX = img.shape
X = np.full(img.shape, np.arange(nX))
X = X / nX
Y = np.full(img.shape, np.arange(nY)[:, np.newaxis])
Y = nY - 1 - Y
Y = Y / nY

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
    mask = np.ones(nbrs.shape, dtype = bool)
    mask[:, :, nbrRadius, nbrRadius] = 0
    nbrs = nbrs[mask].reshape(rowsX, colsX, -1)
    rowsX, colsX, nNbrs = nbrs.shape
    nbrs = nbrs.reshape(-1, nNbrs)
    return nbrs
  
# nbrs = getNeighbors(img, 1)
# print('nbrs.shape = ', nbrs.shape)
# 
# print('Check corners of X')
# print(X[0,0], X[-1,-1])
# print('Check corners of Y')
# print(Y[0,0], Y[-1, -1])
# 
# nbr_weights= nbrs.std(axis = -1, keepdims = True)
# nbr_weights = nbr_weights / np.amax(nbr_weights)
# 
# mask = (nbr_weights > 0.2) & (nbr_weights < 1.0)
# nbr_weights[mask] = 1.0 
# nbr_weights[~mask] = 2**-2
# print(np.unique(nbr_weights, return_counts = True))
# plt.imshow(nbr_weights.reshape(nY, nX))
# plt.show()

# nbrs = getNeighbors(img, 5)
# print('nbrs.shape = ', nbrs.shape)
# 
# print('Check corners of X')
# print(X[0,0], X[-1,-1])
# print('Check corners of Y')
# print(Y[0,0], Y[-1, -1])
# 
# nbr_weights= nbrs.std(axis = -1, keepdims = True)
# nbr_weights = nbr_weights / np.amax(nbr_weights)
# 
# mask = (nbr_weights > 0.1) & (nbr_weights < 1.0)
# nbr_weights[mask] = 1.0 
# nbr_weights[~mask] = 2**-2
# print(np.unique(nbr_weights, return_counts = True))
# plt.imshow(nbr_weights.reshape(nY, nX))
# plt.show()

regions = Image.open('HilbertFeature2.png').convert('L')
#regions = Image.open('pitbullfeature.png').convert('L')
regions = np.asarray(regions, dtype = 'int32')
print('regions.shape = ', regions.shape)
vals, counts = np.unique(regions, return_counts = True)
print('regions values = \n', np.stack([vals, counts], axis = -1)) 
mask = np.full(regions.shape, True)
for val, newval in zip(vals, range(len(vals))):
    masktochange = (regions == val)
    regions[masktochange & mask] = newval
    mask = mask & ~masktochange

vals, counts = np.unique(regions, return_counts = True)
print('regions new values = \n', np.stack([vals, counts], axis = -1)) 

regions = regions / np.amax(regions)
vals, counts = np.unique(regions, return_counts = True)
print('regions new values = \n', np.stack([vals, counts], axis = -1)) 

fig = plt.gcf()
ax = plt.imshow(regions.reshape(nY, nX))
cbar = fig.colorbar(ax)
print('region values are ', np.unique(regions, return_counts = True))
plt.show()
regions = regions.reshape(-1, 1)

# nbr_weights = Image.open('HilbertWeights.png').convert('L')
# 
# nbr_weights = np.asarray(nbr_weights, dtype = 'int32')
# print('img.shape = ', nbr_weights.shape)
# nbr_weights = nbr_weights / np.amax(nbr_weights)
# 
# fig = plt.gcf()
# ax = plt.imshow(nbr_weights.reshape(nY, nX))
# cbar = fig.colorbar(ax)
# print('weight values are ', np.unique(nbr_weights, return_counts = True))
# plt.show()
# nbr_weights = nbr_weights.reshape(-1, 1)
# 
# weight_vals = np.unique(nbr_weights)
# nbr_weights[nbr_weights == weight_vals[0]] = 2**-5
# nbr_weights[nbr_weights == weight_vals[1]] = 2**-3
# nbr_weights[nbr_weights == weight_vals[2]] = 1.0
# print('weight values are ', np.unique(nbr_weights, return_counts = True))

img = img.reshape(-1, 1)
features = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), regions], axis = -1)

nSamples, nFeatures = features.shape
print('features.shape = ', features.shape)
print('img.shape = ', img.shape)

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

print('Setting up regularizer and loss.')

l2_reg = tf.contrib.layers.l2_regularizer(scale = 10**4.8)
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1[2, :]]) #[W1, W2, W3, Wout])

# loss_weights = tf.placeholder(tf.float32, shape = [None, 1]) 
# l2_loss = tf.reduce_mean(loss_weights * tf.square(out - y_)) #+ reg_term

loss_weights = tf.placeholder(tf.float32, shape = [None, 1]) 
l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/30 * 0.85), staircase = True)
# 10**(-1/12 * 0.75)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

# Nice result.
# l2_reg = tf.contrib.layers.l2_regularizer(scale = 1e-1)
# reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [W1, W2, W3, Wout])
# 
# loss_weights = tf.placeholder(tf.float32, shape = [None, 1]) 
# l2_loss = tf.reduce_mean(tf.square(out - y_)) + reg_term
# global_step = tf.Variable(0, trainable = False)
# learning_rate_init = 1e-1
# learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/12 * 0.65), staircase = True)
# # 10**(-1/12 * 0.75)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)


# Train network on image.

batchsize = 200

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    losses = []
    losses_index = []
    for i in range(10000): #25000):
        batchi = np.random.randint(0, len(img), size = batchsize)
        batchx = features[batchi]
        batchy = img[batchi]
        batchweight = nbr_weights[batchi]
        sess.run(train_step, {x: batchx, y_: batchy, loss_weights : batchweight})
        if i % 100 == 0:
            newloss = sess.run(l2_loss, {x: features, y_: img, loss_weights : nbr_weights})
            losses.append(newloss)
            losses_index.append(i)
            print(i, ' loss = ', newloss)
    
    result = sess.run(out, {x: features})
    result = 255 - result

plt.plot(losses_index[1:], losses[1:])
plt.show()

result = result.reshape(nY, nX)
sns.heatmap(result)
plt.show() 


# img = img.reshape(nY, nX)
# edges = img - result
# stratlevels = [-100, 0, 100]
# edges = stratify(edges, stratlevels)
# sns.heatmap(edges)
# plt.show()   

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
sns.heatmap(result)
plt.show() 

# sns.heatmap(result + edges)
# plt.show()

