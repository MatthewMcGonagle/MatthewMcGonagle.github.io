from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define Logistic Function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Get a graph of logistic function
X = np.arange(-7, 7, 0.1)
Y = logistic(X)
fig = plt.figure(figsize = (3.5,3))
plt.plot(X, Y)
plt.title('Logistic Function')
plt.savefig('2017-10-17-graphs/logistic.png')

# Set up domain value arrays.
X = np.arange(100)[np.newaxis, :]
X = np.full((100, 100), X, dtype = 'float32')

Y = np.arange(100)[::-1, np.newaxis]
Y = np.full((100, 100), Y, dtype = 'float32')

# Let's get a look at the graph of the logistic function.
def f(x, y):
    return logistic(x - 50) 

z = f(X, Y) 

plt.clf()
xticks = np.arange(0, 100, 10)
yticks = np.arange(0, 100, 10)
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)

plt.title('One Logistic Function')
plt.savefig('2017-10-17-graphs/manual_onelog.png')

# Now let's look at hidden layer.
def layer1_node1(x, y):
    return logistic(-x +  35)
def layer1_node2(x, y):
    return logistic(y - 65)
def layer1_node3(x, y):
    return logistic(x - 65)
def layer1_node4(x, y):
    return logistic(-y + 35)

z = 1/2.0 * layer1_node2(X,Y) + 1/2.0 * layer1_node3(X,Y) 

plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Two Logistic Functions')
plt.savefig('2017-10-17-graphs/manual_twolog.png')

z = 0.5 * layer1_node1(X,Y) + 0.5 * layer1_node2(X,Y) + 0.5 * layer1_node3(X,Y) + 0.5 * layer1_node4(X,Y)

plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Four Logistic Functions')
plt.savefig('2017-10-17-graphs/manual_fourlog.png')

# Start looking at adding a second hidden layer.

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

mask = (25 < X) & (X < 75) & (25 < Y) & (Y < 75)
z = np.zeros(X.shape)
z[mask] = 1.0
plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Original Shape')
plt.savefig('2017-10-17-graphs/square_train.png')

hidden_layer_sizes = [(1), (2), (3), (4), (10), (100), (5, 2), (8, 2), (10, 10), (10, 10, 10), (10, 10, 10, 10)]
X_train = np.stack([X.reshape(-1), Y.reshape(-1)], axis = -1)
y_train = z.reshape(-1)
mlp = MLPRegressor( activation = 'logistic',
                    learning_rate_init = 1e-1,
                    random_state = 2017 )
model = Pipeline([ ('scaler', StandardScaler()),
                   ('mlp', MLPRegressor( activation = 'logistic',
                                         learning_rate_init = 1e-1,
                                         random_state = 2017 ) ) ])

for sizes in hidden_layer_sizes:
    model.set_params(mlp__hidden_layer_sizes = sizes)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    z_predict = y_predict.reshape(X.shape)
    plt.clf()
    ax = sns.heatmap(z_predict, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.title('hidden_layer_sizes = ' + str(sizes))
    plt.savefig('2017-10-17-graphs/square' + str(sizes) + '.png')
    print('Finished ', sizes)

# Train on layer sizes for more complicated shape.

mask = (25 < X) & (X < 75) & (25 < Y) & (Y < 75)
mask2 = (35 < X) & (X < 65) & (35 < Y) & (Y < 65)
z = np.zeros(X.shape)
z[mask & ~mask2] = 1.0
plt.clf()
ax = sns.heatmap(z, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
ax.set_xticks(xticks)
ax.set_yticks(yticks)
plt.title('Original Shape')
plt.savefig('2017-10-17-graphs/annular_train.png')

y_train = z.reshape(-1)
model = Pipeline([ ('scaler', StandardScaler()),
                   ('mlp', MLPRegressor(hidden_layer_sizes = (3),
                                        activation = 'logistic',
                                        learning_rate_init = 1e-1,
                                        random_state = 2017) ) ])
                                        
for sizes in hidden_layer_sizes:
    model.set_params(mlp__hidden_layer_sizes = sizes)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    z_predict = y_predict.reshape(X.shape)

    plt.clf()
    ax = sns.heatmap(z_predict, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.title('hidden_layer_sizes = ' + str(sizes))
    plt.savefig('2017-10-17-graphs/annular' + str(sizes) + '.png')
    print('Finished ', sizes)

# Now let's train on regularity for large layer sizes.

model = Pipeline([ ('scaler', StandardScaler()),
                   ('mlp', MLPRegressor(hidden_layer_sizes = (50, 50, 50),
                           activation = 'logistic',
                           learning_rate_init = 1e-1,
                           random_state = 2017) ) ])
alphas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for alpha, i in zip(alphas, range(len(alphas))):
    model.set_params(mlp__alpha = alpha)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    z_predict = y_predict.reshape(X.shape)

    plt.clf()
    ax = sns.heatmap(z_predict, vmin = 0.0, vmax = 1.0, xticklabels = xticks, yticklabels = yticks[::-1]) 
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.title('alpha = ' + str(alpha))
    plt.savefig('2017-10-17-graphs/annular_regularity' + str(i) + '.png')
    print('Finished ', alpha)
   
