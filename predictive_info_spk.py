from __future__ import print_function

import tensorflow as tf
import numpy as np
import gzip, pickle, random

FLOATX = np.float32

class SpikingLayer(object):

    def __init__(self, n=128, input_shape=(150, 400, 500), batch_size=100, **kwargs):
        # input_shape=(batch_size,width,height,nb_steps)
        # n: Number of neurons
        self.n = n
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.datapoints = self.input_shape[0]
        self.m = self.input_shape[1]
        self.l = self.input_shape[2]

        self.set_default_attributes()

        for key, value in kwargs.items():
            if key in self.diff_eq_pars:
                self.diff_eq_pars[key] = value
            elif key in self.unit_settings:
                self.unit_settings[key] = value

        self.W_in = self.unit_settings['W_in']
        self.E_in = self.unit_settings['E_in']
        self.tau = self.unit_settings['tau']

        self.E = np.zeros(self.n, dtype=FLOATX)

        # Instantiate a specific TensorFlow graph for the feedforward layer
        self.graph = tf.Graph()

        ################################
        # Build the neuron model graph #
        ################################
        with self.graph.as_default():

            ##############################
            # Variables and placeholders #
            ##############################    
            self.get_vars_and_ph()

            ##############
            # Operations #
            ##############
            
            # Operations to evaluate the membrane response (potential v and recovery u)
            self.potential, self.recovery = self.get_response_ops()

            #self.v_op_max = tf.assign(self.v_history, tf.maximum(self.potential, self.v_history))

            self.get_cost_funcs()

            self.get_iterator()

            self.restart_units()

    def set_default_attributes(self):

        # PARAMETERS OF DIFFERENTIAL EQUATION
        # dv = diff_eq_A*v*v + diff_eq_B*v + diff_eq_C + I - u (default: 0.04, 5, 140)
        self.diff_eq_pars = {'diff_eq_A' : 0.04,
                             'diff_eq_B' : 5.0,
                             'diff_eq_C' : 140.0,
                             'dt' : 1}

        # SETTINGS OF INDIVIDUAL UNITS
        # A = Scale of the membrane recovery (lower values lead to slow recovery)
        # B = Sensitivity of recovery towards membrane potential (higher values lead to higher firing rate)
        # C = Membrane voltage reset value
        # D = Membrane recovery 'boost' after spike
        self.unit_settings = {'A' : np.full((self.batch_size, self.n), 0.02, dtype=FLOATX),
                              'B' : np.full((self.batch_size, self.n), 0.2, dtype=FLOATX),
                              #'C' : np.full((self.batch_size, self.n), -50.0, dtype=FLOATX),
                              #'D' : np.full((self.batch_size, self.n), 2.0, dtype=FLOATX),
                              'C' : np.full((self.batch_size, self.n), -65.0, dtype=FLOATX),
                              'D' : np.full((self.batch_size, self.n), 8.0, dtype=FLOATX),
                              'SPIKING_THRESHOLD' : 35.0,
                              'RESTING_POTENTIAL' : -70.0,
                              'W_in' : np.full((self.n,self.m), 0.07, dtype=FLOATX),
                              'E_in' : np.zeros((self.m), dtype=FLOATX),
                              'tau' : 10.0,
                              'input_scale' : 1,#0.05
                              'recurrent_conductance_update' : 0.02}

        self.network_settings = {'random_seed': 49,
                                 'kernel_size' : 1
                                 }

        self.cost_settings = {'update_func': 'adam',
                              'regularization': 'l1',
                              'reg_factor': 0.0,
                              'lr': 0.001,
                              'lr_decay': 0.5,
                              'act_reg': None,
                              'act_reg_factor': 0.0, 
                              'elastic_alpha': None, 
                              'output_distribution': None}

    ###############################################
    # Define the graph Variables and placeholders #
    ###############################################

    def get_vars_and_ph(self):

            self.init_val = float(10 / np.sqrt(self.m * self.network_settings['kernel_size']))
            #self.init_val = 2.0

            # MEMBRANE POTENTIAL
            # All neurons start at the resting potential
            self.init_v = tf.constant_initializer(dtype=FLOATX, value=self.unit_settings['RESTING_POTENTIAL'])
            self.v = tf.get_variable("v",shape=[self.batch_size, self.n], dtype=FLOATX, initializer=self.init_v, trainable=False)

            #self.v_history = tf.get_variable("v_history",shape=[self.batch_size, self.n], dtype=FLOATX, initializer=self.init_v, trainable=False)

            self.threshold = tf.constant(self.unit_settings['SPIKING_THRESHOLD'], shape=[self.batch_size, self.n])

            # MEMBRANE RECOVERY
            # All neurons start with a value of B * C
            self.init_u = tf.constant_initializer(dtype=FLOATX, value=self.unit_settings['B']*self.unit_settings['C'])
            self.u = tf.get_variable("u",shape=[self.batch_size, self.n], dtype=FLOATX, initializer=self.init_u, trainable=False)

            #self.u = tf.Variable(self.unit_settings['B']*self.unit_settings['C'], name='u')

            # FEEDFORWARD INPUT
            #self.I = tf.placeholder(FLOATX, shape=[self.batch_size, self.m], name="Input")
            self.I = tf.placeholder(FLOATX, shape=[self.batch_size, self.m], name="Input")

            # Weights:
            #self.W_ff = tf.get_variable("W_ff", [self.m, self.n], dtype=FLOATX,
            #                        initializer=tf.random_uniform_initializer(minval=-self.init_val,
            #                                                                  maxval=self.init_val,
            #                                                                  dtype=FLOATX,
            #                                                                  seed=self.network_settings['random_seed']))

            # Weights:
            self.W_ff = tf.get_variable("W_ff", [self.m, self.n], dtype=FLOATX,
                                    initializer=tf.random_uniform_initializer(minval=-self.init_val,
                                                                              maxval=self.init_val,
                                                                              dtype=FLOATX,
                                                                              seed=self.network_settings['random_seed']),
                                                                              trainable=True)

            # And biases:
            self.b_ff = tf.get_variable("b_ff", [self.batch_size, self.n],
                                    initializer=tf.constant_initializer(dtype=FLOATX,
                                                                        value=1.0),
                                                                        trainable=True)

            #self.b_in = tf.get_variable("b_in", [self.batch_size,self.n],
            #                        initializer=tf.constant_initializer(dtype=FLOATX,
            #                                                            value=1.0),
            #                                                            trainable=False)
                    

            # Weighted input
            # Weighted input
            self.I_mat = tf.add(tf.matmul(self.I, self.W_ff), self.b_ff)

            #self.I_mat = tf.nn.relu(self.I_mat, name="hidden_activation")
            # RECURRENT INPUT

            # Weights
            self.W_rec = tf.get_variable("W_rec", [self.n, self.n], dtype=FLOATX,
                                    initializer=tf.random_uniform_initializer(minval=-self.init_val,
                                                                              maxval=self.init_val,
                                                                              dtype=FLOATX,
                                                                              seed=self.network_settings['random_seed']),
                                                                              trainable=False)
            self.W_rec = tf.cast(self.W_rec, dtype=FLOATX)

            #self.W_rec = tf.cast(self.W_rec, dtype=FLOATX)
            # XXX Make into a IRNN

            # OUTPUT

            # Weights:
            #self.W_out = tf.get_variable("W_out", [self.n, self.m], dtype=FLOATX,
            #                        initializer=tf.random_uniform_initializer(minval=-self.init_val,
            #                                                                  maxval=self.init_val,
            #                                                                  dtype=FLOATX,
            #                                                                  seed=self.network_settings['random_seed']),
            #                                                                  trainable=True)

            self.W_out = tf.get_variable("W_out", [self.n, self.m], dtype=FLOATX,
                                    initializer=tf.random_uniform_initializer(minval=-self.init_val,
                                                                              maxval=self.init_val,
                                                                              dtype=FLOATX,
                                                                              seed=self.network_settings['random_seed']),
                                                                              trainable=True)


            # And biases:
            self.b_out = tf.get_variable("b_out", [self.batch_size, self.m],
                                    initializer=tf.constant_initializer(dtype=FLOATX,
                                                                        value=1.0),
                                                                        trainable=True)

            # We also need a placeholder to pass the length of the time interval
            # Length of time interval should be the time between frames
            #self.dt = tf.placeholder(tf.float32, shape=(1))
            self.dt = tf.placeholder(tf.float32, shape=None, name="dt")
            #self.dt = self.diff_eq_pars['dt']

            # Input synapse conductance dynamics (increases on each synapse spike)
            #self.g_in = tf.Variable(tf.zeros(dtype=tf.float32, shape=[self.batch_size, self.m]),
            #                                 dtype=tf.float32,
            #                                 name='g_in')

            # Placeholder to pass the input synapses behaviour at each timestep
            #self.input_syn_has_spike = tf.placeholder(tf.bool, shape=[self.batch_size, self.m], name="has_spiked")

            # RECCURENCY
            # Recurrent synapse conductance dynamics (increases on each synapse spike)
            self.g = tf.get_variable("g", shape=[self.batch_size, self.n],dtype=FLOATX,
            	                    initializer=tf.zeros_initializer(),
            	                    trainable=False)

    #######################################################
    # Define the graph of operations to update v and u:   # 
    # has_fired_op                                        # 
    #   -> (v_reset_op, u_rest_op)      <- I              #
    #      -> (dv_op, du_op)          <- i_op             #
    #        -> (v_op, u_op)                              #
    # We only need to return the leaf operations as their #
    # graph include the others.                           #
    #######################################################

    def get_response_ops(self):

        has_fired_op, v_reset_op, u_reset_op = self.get_reset_ops()

        i_op = self.get_input_ops(has_fired_op, v_reset_op)
        
        v_op, u_op = self.get_update_ops(has_fired_op, v_reset_op, u_reset_op, i_op)

        self.out_spk = tf.sigmoid(v_op - self.threshold)
        
        return v_op, u_op

    def heaviside(self,x,threshold): return tf.clip_by_value(tf.sign(x-threshold),0,1.0)

    def surrogate_grad(self,x,threshold,scale=100e-3):

        out = tf.sigmoid(scale*(x)) - tf.stop_gradient(tf.sigmoid(x) - self.heaviside(x,threshold))

        return out

    def get_reset_ops(self):
        
        # Evaluate which neurons have reached the spiking threshold

        #surrogate gradient for heaviside non-linearity

        has_fired_op = tf.greater_equal(self.v, self.threshold)

        #uncomment the line below (and comment the one above) to run the network without surrogate gradient

        #self.has_fired_op = self.heaviside(self.v, self.threshold)

        # Neurons that have spiked must be reset, others simply evolve from their initial value

        # Membrane potential is reset to C
        v_reset_op = tf.where(has_fired_op, self.unit_settings['C'], self.v)

        # Membrane recovery is increased by D 
        u_reset_op = tf.where(has_fired_op, tf.add(self.u, self.unit_settings['D']), self.u)

        return has_fired_op, v_reset_op, u_reset_op

    # This method is for future use when we introduce synaptic currents

    def restart_units(self):

        self.restart_op = tf.initialize_variables([self.v, self.u])
        #self.restart_history = tf.initialize_variables([self.v_history])

    def get_input_ops(self, has_fired_op, v_op):


        # THIS WHOLE FUNCTION IS ONLY NECESSARY FOR A RECCURRENT LAYER. NON RECURRENTS LAYERS:
        # self.total_input = self.I_mat
        # First, update recurrent conductance dynamics
        # - Increment the current factor of synapses that fired
        # - Decrease by tau the conductance dynamics in any case
        conductance_update = self.unit_settings['recurrent_conductance_update'] * tf.ones(shape=self.g.shape)

        g_update_op = tf.where(has_fired_op,
                                 tf.add(self.g, conductance_update),
                                 tf.subtract(self.g, tf.multiply(self.dt, tf.divide(self.g, self.tau))))

        # Update the g variable
        g_op = tf.assign(self.g, g_update_op)

        # Evaluate the recurrent conductance (I_rec = Σ wjgj(Ej -v(t)))
        #i_rec_op = tf.einsum('ab,cd->ac', tf.multiply(g_op, tf.subtract(tf.constant(self.E), v_op)), self.W_rec)
        i_rec_op = tf.matmul(tf.multiply(g_op, tf.subtract(tf.constant(self.E), v_op)), self.W_rec)

        # The input is the sum of feedforward and recurrent input
        i_op = self.I_mat + i_rec_op

        # Store a reference to this operation for easier retrieval
        self.total_input = i_op

        return i_op
        
    def get_update_ops(self, has_fired_op, v_reset_op, u_reset_op, i_op):
        
        # Evaluate membrane potential increment for the considered time interval
        # dv = 0 if the neuron fired, dv = 0.04v*v + 5v + 140 + I -u otherwise (default)

        dv_op = tf.where(has_fired_op,
                         tf.zeros(self.v.shape),
                         tf.subtract(tf.add_n([tf.multiply(tf.square(v_reset_op), self.diff_eq_pars['diff_eq_A']),
                                               tf.multiply(v_reset_op, self.diff_eq_pars['diff_eq_B']),
                                               tf.constant(self.diff_eq_pars['diff_eq_C'], shape=[self.batch_size, self.n]),
                                               i_op]),
                                     self.u))
            
        # Evaluate membrane recovery decrement for the considered time interval
        # du = 0 if the neuron fired, du = a*(b*v-u) otherwise
        du_op = tf.where(has_fired_op,
                         tf.zeros([self.batch_size, self.n]),
                         tf.multiply(self.unit_settings['A'], tf.subtract(tf.multiply(self.unit_settings['B'], v_reset_op), u_reset_op)))

        # Increment membrane potential, and clamp it to the spiking threshold
        # v += dv * dt

        v_op = tf.assign(self.v, tf.minimum(tf.constant(self.unit_settings['SPIKING_THRESHOLD'], shape=[self.batch_size, self.n]),
                                                 tf.add(v_reset_op, tf.multiply(dv_op, self.dt))))

        # Decrease membrane recovery
        u_op = tf.assign(self.u, tf.add(u_reset_op, tf.multiply(du_op, self.dt)))

        return v_op, u_op

    def get_cost_funcs(self):

        #self.target = tf.placeholder(shape=[self.batch_size, self.h, self.w], dtype=FLOATX, name="target")
        self.target = tf.placeholder(shape=[self.batch_size, self.m], dtype=FLOATX, name="target")
        #linear mapping for the output

        #self.out_spk = self.surrogate_grad(self.v_op_max, self.threshold)

        self.net_output = tf.add(tf.matmul(self.out_spk, self.W_out), self.b_out)

        #self.cost = tf.losses.mean_squared_error(labels=self.target, predictions=self.net_output)
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.net_output,self.target)))
        self.cost_func = tf.reduce_mean(tf.square(tf.subtract(self.net_output,self.target))) + self.cost_settings['reg_factor']*(tf.reduce_sum(tf.abs(self.W_ff)) + tf.reduce_sum(tf.abs(self.W_out)))

        # Get trainable parameters
        #params = [self.W_ff,self.W_out]
        #params = tf.trainable_variables()

        # Apply regularization
        #if self.cost_settings['regularization'] == 'l1':
        #    l1_regularizer = tf.contrib.layers.l1_regularizer(self.cost_settings['reg_factor'],
        #                                                  scope='l1_regularizer')
        #    reg_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=params)
            #tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=params)
        #elif self.cost_settings['regularization'] == 'l2':
        #    l2_regularizer = tf.contrib.layers.l2_regularizer(self.cost_settings['reg_factor'],
        #                                                  scope='l2_regularizer')
        #    reg_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights_list=params)
            #tf.contrib.layers.apply_regularization(l2_regularizer, weights_list=params)
        #else:
        #    reg_penalty = 0

        #self.cost_func = self.cost + reg_penalty

        # OPTIMIZER
        opt = tf.train.AdamOptimizer(self.cost_settings['lr'])
        # Compute and clip gradients
        gradients, variables = zip(*opt.compute_gradients(self.cost_func))
        gradients, _ = tf.clip_by_global_norm(gradients, 50) # XXXXXXXXXXXXXX
        self.train_func = opt.apply_gradients(zip(gradients, variables))

    def get_iterator(self):

        self.iterator_placeholder =tf.placeholder(dtype="float32", shape=self.input_shape)
        dataset = tf.data.Dataset.from_tensor_slices(self.iterator_placeholder)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    def optimize(self, I_in, dt=5e-3, epochs=100):

        steps = range(self.l-1)

        ##############
        # Simulation #
        ##############

        with tf.Session(graph=self.graph) as sess:
            ## Initialize global variables to their default values
            sess.run(tf.global_variables_initializer())
            v_out = []
            for epoch in range(epochs):
                sess.run(self.iterator.initializer, feed_dict={self.iterator_placeholder:I_in})
                cost = 0
                runs = 0

                while True:
                    try:
                        this_batch = sess.run(self.next_batch)
                        v_out_local = []
                        sess.run(self.restart_op)
                        #sess.run(self.restart_history)
                        count_local = 0

                        for step in steps:
                            #sess.run(self.restart_history)
                            # Run the simulation at each time step
                            I_this_step = np.reshape(this_batch[:,:,step], newshape=[self.batch_size, self.m])
                            I_next_step = np.reshape(this_batch[:,:,step+1], newshape=[self.batch_size, self.m])
                            sim_feed = {self.I : I_this_step, self.target : I_next_step, self.dt:dt}
                            #for sim_step in range(40):
                            #v, u = sess.run([self.potential,self.recovery], feed_dict = {self.I : I_this_step, self.dt : dt})
                            #_ = sess.run([self.v_op_max], feed_dict = {self.I : I_this_step, self.dt : dt})
                            #v = sess.run(self.v)
                            #v_out_local.append(v)
                            v, u,_, cost_value = sess.run([self.potential,self.recovery, self.train_func, self.cost], feed_dict = sim_feed)
                            v_out_local.append(v)
                            cost += cost_value
                            runs += 1

                        v_out_local = np.stack(v_out_local)

                    except tf.errors.OutOfRangeError:
                        break
                v_out.append(v_out_local)
                print("epoch", epoch + 1, ", loss: {:.3f}".format(cost/runs))
            filters = sess.run(self.W_ff)
            reconstruction_W = sess.run(self.W_out)
            reconstruction = sess.run(self.net_output, feed_dict = {self.I: I_this_step, self.dt:dt})
        v_out = np.stack(v_out)
        np.save("/home/diego/Documents/predictive_info/spiking/reconstructions.npy",reconstruction)
        np.save("/home/diego/Documents/predictive_info/spiking/input_frames.npy",I_this_step)
        np.save("/home/diego/Documents/predictive_info/spiking/filters.npy",filters)
        np.save("/home/diego/Documents/predictive_info/spiking/W_out.npy",reconstruction_W)

        return v_out


################################################################################################################
#                                   NETWORK RUNNER. FEED INPUT DATA HERE.                                      #
################################################################################################################

################
#  Parameters  #
################

epochs = 20
batch_size = 100
dt = 1
num_units = 128

#load data
print("Loading data...")

with gzip.open("/home/diego/Documents/predictive_info/preprocessed_dataset.pkl.gz") as f:
    data = pickle.load(f)

print("Data loaded")
print("Creating layer of spiking units...")
data = data[:,80:,140:,...]
nb_inputs = data.shape[1]*data.shape[2]

#pick a training set
training_set = data[:1000]
training_set = training_set.reshape(training_set.shape[0],nb_inputs,training_set.shape[3])
x_train = (training_set-np.mean(training_set))/np.std(training_set)
#x_train = training_set/np.max(training_set)
#create a layer of Izhikevich integrate-and-fire units

spikinglayer = SpikingLayer(n=num_units, input_shape=x_train.shape, batch_size=batch_size)

#train the network
print("Training the network...")
v_out = spikinglayer.optimize(I_in=x_train, dt=dt, epochs=epochs)
np.save("/home/diego/Documents/predictive_info/spiking/mem_potential.npy",v_out)