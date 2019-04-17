import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gzip, pickle, random

#--------------------------------------------------------------------------------------------------

batch = 50
max_epochs = 50
learning_rate = 0.0005

in_height = 97
in_width = 157
in_channels = 1
hidden_units = 128
num_predictions = 1
stride_height = 4
stride_width = 4
frames = 5
filter_depth = frames
filter_height = 21
filter_width = 21
filter2_depth = 1
filter2_height = filter_height
filter2_width = filter_width
#reg_parameter = pow(10.0,-5.5)

# SET THE REGULARIZATION STRENGTH YOU WANT TO USE

reg_parameters = np.logspace(start=-4.5, stop=-6.5, num=9, base=10.0)

#scales = np.logspace(start=0.0, stop=2.0, num=3, base=10.0)
out_height = 77
out_width = 137

#--------------------------------------------------------------------------------------------------

#Load data

save_path = '/set/path/to/save/folder/'

with gzip.open("set/path/to/folder/with/data") as f:
	data = pickle.load(f)
print()
print(">>Images have been decompressed and loaded. Normalizing dataset...")

#normalize sets. suffle_training_set is a function to break data into input/target

training_set_prev = data[:900]
training_set_prev = training_set_prev[:,3:,3:,...]
validation_set_prev = data[900:1000]
validation_set_prev = validation_set_prev[:,3:,3:,...]
training_set_normal = (training_set_prev-np.mean(training_set_prev))/np.std(training_set_prev)
validation_set_normal = (validation_set_prev-np.mean(validation_set_prev))/np.std(validation_set_prev)

def suffle_training_set (training_set_normal,frames):
	np.random.shuffle(training_set_normal)
	start = np.random.randint(0,44,size=training_set_normal.shape[0])
	for i in range(training_set_normal.shape[0]):	
		x_training_set = training_set_normal[:,:,:,start[i]:start[i]+frames]
		x_training_set = np.rollaxis(x_training_set,3,1)
		x_training_set = np.expand_dims(x_training_set,axis=4)

		y_training_set = training_set_normal[:,10:-10,10:-10,start[i]+frames+1]
		y_training_set = np.expand_dims(y_training_set,axis=4)
		y_training_set = np.expand_dims(y_training_set,axis=1)
	return x_training_set, y_training_set

x_training_set, y_training_set = suffle_training_set(training_set_normal,frames)
x_validation_set, y_validation_set = suffle_training_set(validation_set_normal,frames)

print()
print(">>Training set prepared.")
print()

#---------------------------------------------------------------------------------------------------
loss_val_all = []
loss_train_all = []
loss_val_reg_all = []
loss_train_reg_all = []

for reg_parameter in reg_parameters:
	
	# Set Graph nodes

	batch_input = tf.placeholder(dtype=tf.float32,shape=[None,frames,in_height,in_width,in_channels])

	W1 = tf.Variable(tf.random_normal([filter_depth,filter_height,filter_width,in_channels,hidden_units], stddev=0.01))
	b1 = tf.Variable(tf.zeros([hidden_units]))
	W2 = tf.Variable(tf.random_normal([filter2_depth, filter2_height, filter2_width, in_channels, hidden_units], stddev=0.01))
	b2 = tf.Variable(tf.zeros([num_predictions,out_height,out_width,in_channels]))

	batch_output = tf.placeholder(dtype=tf.float32,shape=[None,num_predictions,out_height,out_width,in_channels])

	#--------------------------------------------------------------------------------------------------

	#Define the operations inside the Graph

	convolution = tf.sigmoid(tf.add(tf.nn.conv3d(batch_input,W1,strides=[1,1,stride_height,stride_width,1], padding="VALID"), b1))
	output = tf.add(tf.nn.conv3d_transpose(convolution,W2,output_shape=[batch,num_predictions,out_height,out_width,in_channels],
		strides=[1,1,stride_height,stride_width,1], padding="SAME"),b2)

	#---------------------------------------------------------------------------------------------------

	#Define cost funciton and training algorithm

	loss = tf.reduce_mean(tf.square(tf.subtract(output,batch_output))) + reg_parameter*(tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)))
	loss_standard = tf.reduce_mean(tf.square(tf.subtract(output,batch_output)))
	train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	#---------------------------------------------------------------------------------------------------

	#Make iterators to make batches and feed the data. There are 4. 2 for the training set and 2 for the validation set, because they cannot vary
	#in size but both sets have different sizes. Also, there are two for each set because the input and output shapes are also different.

	iterator_placeholder_x =tf.placeholder(dtype="float32", shape=x_training_set.shape)
	dataset_x=tf.data.Dataset.from_tensor_slices(iterator_placeholder_x)
	dataset_x=dataset_x.batch(batch)
	iterator_x = dataset_x.make_initializable_iterator()
	next_x=iterator_x.get_next()

	iterator_placeholder_y =tf.placeholder(dtype="float32", shape=y_training_set.shape)
	dataset_y=tf.data.Dataset.from_tensor_slices(iterator_placeholder_y)
	dataset_y=dataset_y.batch(batch)
	iterator_y = dataset_y.make_initializable_iterator()
	next_y=iterator_y.get_next()

	iterator_placeholder_x_val =tf.placeholder(dtype="float32", shape=x_validation_set.shape)
	dataset_x_val=tf.data.Dataset.from_tensor_slices(iterator_placeholder_x_val)
	dataset_x_val=dataset_x_val.batch(batch)
	iterator_x_val = dataset_x_val.make_initializable_iterator()
	next_x_val=iterator_x_val.get_next()

	iterator_placeholder_y_val =tf.placeholder(dtype="float32", shape=y_validation_set.shape)
	dataset_y_val=tf.data.Dataset.from_tensor_slices(iterator_placeholder_y_val)
	dataset_y_val=dataset_y_val.batch(batch)
	iterator_y_val = dataset_y_val.make_initializable_iterator()
	next_y_val=iterator_y_val.get_next()

	#--------------------------------------------------------------------------------------------------

	# Train the network, store the filters after each epoch and store loss both with and without regularization

	print(">>Learning the task...")
	print()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	epoch = 0
	loss_train = []
	loss_train_reg = []
	reg_string = np.array2string(reg_parameter)
	#scale_string = np.array2string(scale)
	while epoch < max_epochs:
		epoch +=1
		loss_epoch = 0
		loss_epoch_reg = 0
		batch_num = 0
		star = 0
		while star < 44:
			x_training_set, y_training_set = suffle_training_set(training_set_normal,frames)
			star = star+1
			sess.run(iterator_x.initializer, feed_dict={iterator_placeholder_x: x_training_set})
			sess.run(iterator_y.initializer, feed_dict={iterator_placeholder_y: y_training_set})
			while True:
				try:
					batch_x = sess.run(next_x)
					batch_y = sess.run(next_y)
					loss_batch, loss_batch_reg, _ = sess.run([loss_standard, loss, train], feed_dict={batch_input: batch_x, batch_output: batch_y})
					loss_epoch += loss_batch
					loss_epoch_reg += loss_batch_reg
					batch_num += 1
				except tf.errors.OutOfRangeError:
					break	
		filters = sess.run(W1)
		#reconstructions = sess.run(output, feed_dict={batch_input: batch_x})
		loss_epoch = loss_epoch/batch_num
		loss_epoch_reg = loss_epoch_reg/batch_num

		loss_train.append(loss_epoch) 
		loss_train_reg.append(loss_epoch_reg) 

		print("epoch:", epoch, "loss: {:.3f}".format(loss_epoch))

		epochs_to_save = np.array2string(np.array(epoch))

		string_to_save = "reg_"+reg_string+"_epoch_"+epochs_to_save

		np.save(save_path+string_to_save+"_filters.npy", filters)

	loss_train = np.stack(loss_train)
	loss_train_reg = np.stack(loss_train_reg)

	loss_train_all.append(loss_train)
	loss_train_reg_all.append(loss_train_reg)

	loss_val = 0
	loss_val_reg = 0
	batch_num = 0
	star = 0
	while star < 44:
		x_validation_set, y_validation_set = suffle_training_set(validation_set_normal,frames)
		star = star+1
		sess.run(iterator_x_val.initializer, feed_dict={iterator_placeholder_x_val: x_validation_set})
		sess.run(iterator_y_val.initializer, feed_dict={iterator_placeholder_y_val: y_validation_set})
		while True:
			try:
				batch_x = sess.run(next_x_val)
				batch_y = sess.run(next_y_val)
				loss_val_batch, loss_val_reg_batch = sess.run([loss_standard, loss], feed_dict={batch_input: batch_x, batch_output: batch_y})
				loss_val += loss_val_batch
				loss_val_reg += loss_val_reg_batch
				batch_num += 1
			except tf.errors.OutOfRangeError:
				break
	
	loss_val = loss_val/batch_num
	loss_val_reg = loss_val_reg/batch_num
	loss_val_all.append(loss_val)
	loss_val_reg_all.append(loss_val_reg)

loss_val_all = np.stack(loss_val_all)
loss_val_reg_all = np.stack(loss_val_reg_all)
loss_train_all = np.stack(loss_train_all)
loss_train_reg_all = np.stack(loss_train_reg_all)

np.save(save_path+"loss_train.npy", loss_train_all)
np.save(save_path+"loss_train_reg.npy", loss_train_reg_all)
np.save(save_path+"loss_validation.npy", loss_val_all)
np.save(save_path+"loss_validation_reg.npy", loss_val_reg_all)

print()
print(">> Hyperparameter search completed.")
print()
		#------------------------------------------------------------------------------------------------