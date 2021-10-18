from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
 #so far we just generated a synthetic datatable like how we did on LNNd2l

def load_array(data_arrays, batch_size, is_train=True): #@save
    """Construct a Gluon data iterator."""
    #Here we called a data iterator rather than using the one that we created on LNNd2l example
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
    #Here you can enable shuffling on each epoch when shuffle is true.

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))
#As it is obvious, we wanted to iterate first few items in the range of batch size

#Here, we don't actually need to call a model for training but I wanted to show how a layer can be called from a
#specific library. In linear neural network case, it may not be useful, because LNNs consist of only one layer. But in
#the upcoming examples, we'll be using this specific model operation and it will save us more time to focus on focusing
#especially on the layers used to construct the model rather than having to focus on the implementation.

# nn is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
# In Gluon, the fully-connected layer is defined in the Dense class. Since we only want to generate a single scalar
# output, we set that number to 1.

from mxnet import init
net.initialize(init.Normal(sigma=0.01))
# mxnet also allows as to initialize model parameters quickly.
# Beware that something strange occurs here. We are initializing parameters for a network even though Gluon does not yet
# know how many dimensions the input will have! Gluon lets us get away with this because behind the scene, the
# initialization is actually deferred. We can only access or manipulate parameters after the first time attempt to pass
# data through the network.

loss = gluon.loss.L2Loss()

# In Gluon, the loss module defines various loss functions. In this example, we will use the Gluon implementation of
# squared loss (L2Loss).

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# The line above implies an optimization algorithm by itself. we will specify the parameters to optimize over
# (obtainable from our model net via net.collect_params()), the optimization algorithm we wish to use (sgd), and
# a dictionary of hyperparameters required by our optimization algorithm. Minibatch stochastic
# gradient descent just requires that we set the value learning_rate, which is set to 0.03 here.

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')

# Even if we have used high-level APIs, the training loop above is interestingly similar to the one that we used on LNNd2l.py

w = net[0].weight.data()
print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')