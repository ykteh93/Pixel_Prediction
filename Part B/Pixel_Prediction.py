########################################################
#               Written by: Yih Kai Teh                #
########################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

number_input      = 1		# number of input (1 pixel at a time)
number_steps      = 484		# number of total step (since input is 1, so 784 steps)
number_classes    = 1		# number of class (since it is binary so 1 class)
number_hidden     = 128		# number of hideen layer units
number_test_image = 100		# number of test image to be evaluated
image_dimension   = 28		# size of the image dimension which is 28 x 28


# binarize the values of pixels to be either 0 or 1
def binarize(images, threshold=0.1):
	return (threshold < images).astype('float32')

# initialize the weights and biases
weights = {'out': tf.Variable(tf.truncated_normal([number_hidden, number_classes]))}
biases  = {'out': tf.Variable(tf.truncated_normal([number_classes]))}

ground_truth = tf.placeholder(tf.float32, shape=[None, 784])	# placeholder for original image
mask = tf.placeholder(tf.float32, shape=[None, 484])		# placeholder for masked image (484 pixels unmasked, 300 pixels masked)

reshaped_x = tf.reshape(mask, [-1, number_steps, number_input])	# reshaped masked image into right dimension to be passed into RNN cell

# compute the output of the RNN cell states and output with GRU cell
with tf.variable_scope("encoder"):
	gru_cell        = tf.contrib.rnn.GRUCell(number_hidden)
	outputs, states = tf.nn.dynamic_rnn(gru_cell, reshaped_x, dtype=tf.float32)

# use the last rnn outputs as input and compute the output of linear layer
linear_layer = tf.matmul(outputs[:, -1, :], weights['out']) + biases['out']

# store the linear layer to be used for cross entropy loss calculation with logits
prediction = linear_layer

# predict the first masked pixel by setting output of sigmoid to 1 if >0.5
next_pixel = tf.reshape(tf.rint(tf.nn.sigmoid(linear_layer)),[1,1])

# concat the first predicted pixel with masked image of 484 pixels to become 485 pixels
incremental_input = tf.concat([mask, next_pixel], 1)

# predict the next 299 masked pixels (this process is similar as above)
# can't combine with above because of the different RNN states 
with tf.variable_scope("encoder", reuse=True):
	for i in range(299):
		reshaped_next_pixel = tf.reshape(next_pixel, [-1, 1, number_input])

		predict_outputs, states = tf.nn.dynamic_rnn(gru_cell, reshaped_next_pixel, dtype=tf.float32, initial_state=states)

		linear_layer = tf.matmul(predict_outputs[:, -1, :], weights['out']) + biases['out']
		prediction   = tf.concat([prediction, linear_layer], 1)

		next_pixel        = tf.reshape(tf.rint(tf.nn.sigmoid(linear_layer)), [1, 1])
		incremental_input = tf.concat([incremental_input, next_pixel], 1)

# compute the cross entropy loss between the ground truth and prediction
cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=ground_truth[:, 484:784]), 1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	saver = tf.train.Saver()
	saver.restore(sess, "./model/gru_128")

	# sample batch of test images and binarize it
	images, _ = mnist.test.next_batch(number_test_image)
	images    = binarize(images)

	for i in range(number_test_image):
		im           = np.reshape(images[i] , (1,784))
		loss, result = sess.run([cross_entropy_loss,incremental_input], feed_dict={ground_truth: im, mask: im[:, 0:484]})

		print("\rSaving Image {}".format(i), end="")

		# plot the first image which is the original image
		fig            = plt.figure(i, figsize=(15, 11))
		original_image = images[i, :]
		original_image = np.reshape(original_image, (image_dimension, image_dimension))
		ax = plt.subplot(1, 3, 1)
		ax.set_xticks(())
		ax.set_yticks(())
		plt.imshow(original_image)
		plt.xlabel("Ground Truth", fontsize=18)

		# plot the second image which is the masked image
		masked_image          = np.tile(images[i, :], 1)
		masked_image[484:784] = 0.5
		masked_image          = np.reshape(masked_image, (image_dimension, image_dimension))
		ax = plt.subplot(1, 3, 2)
		ax.set_xticks(())
		ax.set_yticks(())
		plt.imshow(masked_image, cmap='Greys')
		plt.xlabel("Masked Image", fontsize=18)

		# plot the third image which is the predicted image
		predicted_image = np.reshape(result,(image_dimension, image_dimension))
		ax = plt.subplot(1, 3, 3)
		ax.set_xticks(())
		ax.set_yticks(())
		plt.imshow(predicted_image, cmap='Greys')
		plt.xlabel("Prediction (Loss: %f)" % (loss), fontsize=18)

		plt.savefig('./image/Image %d.png' % (i))
