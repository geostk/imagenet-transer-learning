import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load dataset
dataset = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True)

# Split dataset into train, test and validation set.
X_train = dataset.train
X_test = dataset.test
X_valid = dataset.validation

# define hyperparameters
image_size = 28
image_shape = (image_size, image_size)
image_shape_flat = image_size * image_size
num_classes = 10

learning_rate = 1e-3
batch_size = 500
num_iter = 2000

# define model's placeholder variables
X = tf.placeholder(tf.float32, [None, image_shape_flat])
y = tf.placeholder(tf.float32, [None, num_classes])
y_true = tf.argmax(y, axis=1)

W = tf.Variable(tf.truncated_normal(shape=[image_shape_flat, num_classes]))
b = tf.Variable(tf.zeros(shape=[num_classes]))

# Build the model
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)
y_pred_true = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = optimizer.minimize(cost)

# Evaluate the model
correct = tf.equal(y_true, y_pred_true)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(num_iter):
    X_batch, y_batch = X_train.next_batch(batch_size=batch_size)
    _ = sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
    if step % 500 == 0:
        _accuracy = sess.run(accuracy, feed_dict={X: X_valid.images, y: X_valid.labels})
        print('{:.2%}'.format(_accuracy))

overall_accuracy = sess.run(accuracy, feed_dict={X: X_test.images, y: X_test.labels})
print("Overall accuracy on test set = {:.2%}".format(overall_accuracy))

# Close tensorflow's Session
sess.close()
