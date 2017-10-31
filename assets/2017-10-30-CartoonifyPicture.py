import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys

fig = plt.figure(figsize = (3.5,3))

# Open the image we will try to turn into a cartoon.

img = Image.open('HilbertPic.jpg')
img = np.asarray(img, dtype = 'int32')
print('img.shape = ', img.shape)

# Function for plotting images with tick marks.

def plotimage(data):
    nY, nX = data.shape
    plt.clf()
    plt.imshow(data, cmap = plt.get_cmap('gray'))
    labels = np.arange(0, 1, .1)
    yticks = np.arange(0, 1, .1)
    ax = plt.gca()
    ax.grid(False)
    plt.xticks(np.arange(0, nX, nX / 10), labels) 
    plt.yticks(np.arange(0, nY, nY / 10), 1.0 - labels) 


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

plotimage(stratimg)
plt.title('Simple Discrete Levels')
plt.savefig('2017-10-30-graphs/simple.png')

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

def createLayers(nFeatures):

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

    layers = { 'x':x, 'y_':y_, 'W1':W1, 'out':out }
    return layers 
   
layers = createLayers(nFeatures) 

# Set up the loss function and training step. 

print('Setting up loss.')

l2_loss = tf.reduce_mean(tf.square(layers['out'] - layers['y_']))
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
            sess.run(train_step, {layers['x']: batchx, layers['y_']: batchy})
            if i % 100 == 0:
                newloss = sess.run(l2_loss, {layers['x']: features, layers['y_']: img})
                losses.append(newloss)
                losses_index.append(i)
                print(i, ' loss = ', newloss)
        
        result = sess.run(layers['out'], {layers['x']: features})
        
        return losses_index, losses, result

print('\nTraining for Constant Rate')
losses_index, losses, result = runSession(10000)

plt.clf()
plt.plot(losses_index[1:], losses[1:])
plt.title('Losses for Constant Training Rate')
plt.savefig('2017-10-30-graphs/constantRateLosses.svg')

result = result.reshape(nY, nX)
plotimage(result)
plt.title('Constant Training Rate')
plt.savefig('2017-10-30-graphs/constantRate.png') 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
plotimage(result)
plt.title('Constant Rate After Stratifying')
plt.savefig('2017-10-30-graphs/constantRateStrat.png') 

# Try decaying learning rate.

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

print('\nTraining for Variable Training Rate')
losses_index, losses, result = runSession(10000)

plt.clf()
plt.plot(losses_index[1:], losses[1:])
plt.title('Losses for Decaying Training Rate')
plt.savefig('2017-10-30-graphs/decayRateLosses.svg')

result = result.reshape(nY, nX)
plotimage(result)
plt.title('Decaying Training Rate')
plt.savefig('2017-10-30-graphs/decayRate.png') 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
plotimage(result)
plt.title('Decaying Rate after Stratification')
plt.savefig('2017-10-30-graphs/decayRateStrat.png') 

# Now look at creating another feature based on standard deviation in neighborhood of point.

def getNeighborhood(X, nbrRadius):
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
    nbrs = nbrs.reshape(rowsX, colsX, -1)
    rowsX, colsX, nNbrs = nbrs.shape
    nbrs = nbrs.reshape(-1, nNbrs)
    return nbrs

nbrs = getNeighborhood(img, 4)
print('nbrs.shape = ', nbrs.shape)

nbrs = nbrs.std(axis = -1, keepdims = True)
nbrs = nbrs / np.amax(nbrs)

plotimage(nbrs.reshape(nY, nX))
plt.title('Standard Deviation of Neighborhoods')
plt.savefig('2017-10-30-graphs/nbrFeature.png')

# Create new features.

features = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), nbrs.reshape(-1,1)], axis = -1)

nSamples, nFeatures = features.shape
print('features.shape = ', features.shape)

# Recreate layers. 

print('Setting up layers.')
tf.reset_default_graph()
tf.set_random_seed(20171026)
np.random.seed(20171026)

layers = createLayers(nFeatures)

# Set up loss.

print('Setting up loss.')

l2_loss = tf.reduce_mean(tf.square(layers['out'] - layers['y_'])) 

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

print('Training for Including Neighborhood Stddev')
losses_index, losses, result = runSession(10000)

plt.clf()
plt.plot(losses_index[1:], losses[1:])
plt.title('Losses for Including Neighborhood Stddev')
plt.savefig('2017-10-30-graphs/nbrLosses.svg')

result = result.reshape(nY, nX)
plotimage(result)
plt.title('Using Neighborhood Stddev')
plt.savefig('2017-10-30-graphs/nbr.png') 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
plotimage(result)
plt.title('Using Neighborhood Stddev After Stratification')
plt.savefig('2017-10-30-graphs/nbrStrat.png') 

# Set up regularizer and loss.

print('Setting up loss.')

l2_reg = tf.contrib.layers.l2_regularizer(scale = 10**1.5) 
reg_term = tf.contrib.layers.apply_regularization(l2_reg, weights_list = [layers['W1'][2, :]]) 
l2_loss = tf.reduce_mean(tf.square(layers['out'] - layers['y_'])) + reg_term

global_step = tf.Variable(0, trainable = False)
learning_rate_init = 1e-1
learning_rate = tf.train.exponential_decay(learning_rate_init, global_step, 500, 10**(-1/20 * 1.50), staircase = True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(l2_loss, global_step = global_step)

print('Training for Regularity Applied to Neighborhood Feature')
losses_index, losses, result = runSession(10000)

plt.clf()
plt.plot(losses_index[1:], losses[1:])
plt.title('Losses for Using Regularity Neighborhood Stddev')
plt.savefig('2017-10-30-graphs/regularityLosses.svg')

result = result.reshape(nY, nX)
plotimage(result)
plt.title('Using Regularity of Neighborhood Stddev')
plt.savefig('2017-10-30-graphs/regularity.png') 

result[result > 255] = 255 
result[result < 0] = 0
stratlevels = np.arange(0, 256, int(256 / 8))
result = stratify(result, stratlevels)
plotimage(result)
plt.title('Regularity Stddev After Stratification')
plt.savefig('2017-10-30-graphs/regularityStrat.png') 

