########################################################
#               Written by: Yih Kai Teh                #
########################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate  = 0.005							# Learning rate for optimizer
batch_size     = 250							# size of mini batch for each training
number_epochs  = 80							# number of desired epochs for training
number_input   = 1							# number of input (1 pixel at a time)
number_steps   = 784							# number of total step (since input is 1, so 784 steps)
number_classes = 1							# number of class (since it is binary so 1 class)
number_hidden  = 128							# number of hideen layer units
train_accuracy = test_accuracy = train_loss = test_loss = np.array([])	# store the details for plotting the graph


# binarize the values of pixels to be either 0 or 1
def binarize(images, threshold=0.1):
	return (threshold < images).astype('float32')


# evaluation is done in batches because the full batches is memory exhausting
def batch_by_batch_evalution(data_set):
	# intialize the batch accuracy and loss into 0 for each evaluation
	batch_accuracy = batch_loss = 0

	# compute the number of total batches is needed (depend on the specified batch size)
	total_batches = int(data_set.num_examples / batch_size)

	# pass all the images batch by batch to compute the accuracy and loss
	for _ in range(total_batches):
		images, _ = data_set.next_batch(batch_size)
		images    = binarize(images)
		acc, loss = sess.run([average_accuracy, cross_entropy_loss], feed_dict={input_x: images, true_y: images[:,1:]})
		batch_accuracy += acc
		batch_loss     += loss

	# calculate the average loss and accuracy for the entire batch
	batch_average_loss     = batch_loss / total_batches
	batch_average_accuracy = batch_accuracy / total_batches
	return batch_average_accuracy, batch_average_loss

# initialize the weights and biases
weights = {'out': tf.Variable(tf.truncated_normal([number_hidden, number_classes]))}
biases  = {'out': tf.Variable(tf.truncated_normal([number_classes]))}

# placeholder for the input which is the entire image
input_x = tf.placeholder(tf.float32, shape=[None, 784])

# placeholder for the true pixel to compare except the first pixel 
true_y = tf.placeholder(tf.float32, shape=[None, 783])

# reshaped the image into right dimension to be passed into RNN cell
reshaped_x = tf.reshape(input_x, [-1, number_steps, number_input])

# compute the output of the RNN cell states and output with GRU cell
with tf.variable_scope("encoder"):
	gru_cell        = tf.contrib.rnn.GRUCell(number_hidden)
	outputs, states = tf.nn.dynamic_rnn(gru_cell, reshaped_x, dtype=tf.float32)

all_outputs = tf.reshape(outputs,[-1, number_hidden])

# use the last rnn outputs as input and compute the output of linear layer then reshape
linear_layer          = tf.matmul(all_outputs, weights['out']) + biases['out']
reshaped_linear_layer = tf.reshape(linear_layer, [-1,784])

# remove the last output because there is nothing to compare 
predict_y = reshaped_linear_layer[:,:-1]

# compute the cross-entropy loss and optimize with ADAM to reduce the loss
cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_y, labels=true_y))
train_op           = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

# compute the average accuracy by counting the number of correct predicted label
accuracy         = tf.equal(tf.rint(tf.nn.sigmoid(predict_y)), true_y)
average_accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	saver             = tf.train.Saver()
	previous_test_loss = 10000

	# run training for 80 Epochs
	for epoch in range(number_epochs):
		for _ in range(int(mnist.train.num_examples / batch_size)):
			images, _ = mnist.train.next_batch(batch_size)
			images    = binarize(images)
			sess.run(train_op, feed_dict={input_x: images, true_y: images[:,1:]})

		# evaluate the loss and accuracy from both training and testing set 
		batch_train_accuracy, batch_train_loss = batch_by_batch_evalution(mnist.train)
		batch_test_accuracy, batch_test_loss = batch_by_batch_evalution(mnist.test)

		# display and store all the result for plotting the graph at the end of training
		print("Epoch %4d -- Training Loss: %10f -- Testing Loss: %10f -- Train Accuracy: %f -- Test Accuracy: %f" % (epoch, batch_train_loss, batch_test_loss, batch_train_accuracy*100, batch_test_accuracy*100))
		train_accuracy = np.append(train_accuracy, batch_train_accuracy * 100)
		test_accuracy  = np.append(test_accuracy, batch_test_accuracy * 100)
		train_loss     = np.append(train_loss, batch_train_loss)
		test_loss      = np.append(test_loss, batch_test_loss)

		# only save the best model when the test loss is the lowest 
		if batch_test_loss < previous_test_loss:
			previous_test_loss = batch_test_loss
			saver.save(sess, './model/gru_128')

	# plot the graph for the accuracy and loss throughout training
	x = np.linspace(0, number_epochs - 1, num=number_epochs)
	plt.figure(0)
	plt.plot(x, train_accuracy, 'r', label='Train')
	plt.plot(x, test_accuracy, 'b', label='Test')
	plt.title('Plot of Train and Test Accuracy Over %d Epochs for GRU 128 units' % (number_epochs))
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy (Percent)')
	plt.legend()
	ax  = plt.subplot(111)
	box = ax.get_position()
	lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig('Plot of Train and Test Accuracy Over %d Epochs.png' % (number_epochs), bbox_extra_artists=(lgd,), bbox_inches='tight')

	plt.figure(1)
	plt.plot(x, train_loss, 'r', label='Train')
	plt.plot(x, test_loss, 'b', label='Test')
	plt.title('Plot of Train and Test Loss Over %d Epochs for GRU 128 units' % (number_epochs))
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	ax  = plt.subplot(111)
	box = ax.get_position()
	lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig('Plot of Train and Test Loss Over %d Epochs.png' % (number_epochs), bbox_extra_artists=(lgd,), bbox_inches='tight')
