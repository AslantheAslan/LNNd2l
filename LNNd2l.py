from d2l import mxnet as d2l
import math
from mxnet import gluon, np, autograd, npx
import time
import random
import matplotlib.pyplot as plt

n = 10000
a = np.ones(n)
b = np.ones(n)

#Let's define a timer because we well benchmark the running time

class Timer: #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

c = np.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
#Here I wanted to benchmark the workload of summing every single element of a and b into a single array.
print(f'{timer.stop():.5f} sec')

timer.start()
d = a + b
print(f'{timer.stop():.8f} sec')

print("Generating the Dataset")

def synthetic_data(w, b, num_examples): #@save
    """Generate y = Xw + b + noise."""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
#You can also try d2l.numpy(features[:, 0])

print("Reading the Dataset")
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, num_examples)])
    yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

print("Initializing Model Parameters")

w = np.random.normal(0, 0.01, (2, 1))
#notice that (2,1) is the size that we want to get
#we just create a weight matrix randomly
# Please note that if we took the weight as zero by the fololowing command: w = np.zeros((2, 1)), the code would work
# the output would be as successfull as when we take the w as a random normal distribution
b = np.zeros(1)
#let's make the data noiseless with the above line
w.attach_grad()
b.attach_grad()
#Attribution of gradient has been added to both matrices
print("Weight tensor",w)
print(b)

print("Defining the Model")

def linreg(X, w, b): #@save
    """The linear regression model."""
    """Basically, we want to create a model such as y=Xw+b"""
    return np.dot(X, w) + b

print("Defining the Loss Function")
def squared_loss(y_hat, y): #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    #the reason why we reshaped y as y.reshape(y_hat.shape) is that we need y as in the same shape as y_hat

print("Defining the Optimization Algorithm")
def sgd(params, lr, batch_size): #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
#The above code applies the minibatch stochastic gradient descent
#update, given a set of parameters, a learning rate, and a batch size.

print("Training")
#In summary, we will execute the following loop:
# Initialize parameters (w; b)
# Repeat until done
#    Compute gradient g
#    Update parameters (w; b)

lr = 0.4
num_epochs = 10
#Here I used diff hyperparameters with the book "d2l". You can choose different learning rates
#and number of epochs but the process of choosing better hyperparameters are not easy and depends on experience.
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y) # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size) # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')

#Up to this part, we have seen how a deep network can be implemented and optimized from scratch, using just tensors and
#auto differentiation, without any need for defining layers or fancy optimizers.