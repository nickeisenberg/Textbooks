# Section 3.5.4 example 

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# create training data

num_samples_per_class = 2000

neg_samples = np.random.multivariate_normal(
        mean = [0,3],
        cov = [[1,.5], [.5,1]],
        size = num_samples_per_class)

pos_samples = np.random.multivariate_normal(
        mean = [3,0],
        cov = [[1,.5], [.5,1]],
        size = num_samples_per_class)

inputs = np.vstack((neg_samples, pos_samples)).astype(np.float32)

targets = np.vstack(
        (np.zeros((num_samples_per_class,1), dtype='float32'),
    np.ones((num_samples_per_class,1), dtype='float32'))
    )

# create linear classifier variables 
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim,output_dim)))
b = tf.Variable(initial_value=tf.random.uniform(shape=(output_dim,)))

# Forward pass function 
def model(inputs):
    return tf.matmul(inputs, W) + b

# Loss function
def square_loss(targets, predictions):
    dif = tf.square(targets - predictions)
    return tf.reduce_mean(dif)

# Training step
learningrate = 0.1

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learningrate)
    b.assign_sub(grad_loss_wrt_b * learningrate)
    return loss

# Training
for step in range(40):
    loss = training_step(inputs, targets)
    # print(loss)

# Plot predicitons 
x = np.linspace(-1,4,100)
y1 = - W[0] / W[1] * x + (.5 - b) / W[1]
plt.plot(x,y1,'-r')

predictions = model(inputs)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(inputs[:,0], inputs[:,1], c=targets[:,0])
ax1.set_title('Targets')

ax2.scatter(inputs[:,0], inputs[:,1], c=predictions[:,0] > .5)
ax2.set_title('Predictions')
x = np.linspace(-1,4,100)
y1 = - W[0] / W[1] * x + (.5 - b) / W[1]
ax2.plot(x,y1,'-r')


plt.show()

