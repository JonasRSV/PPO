import tensorflow as tf
import scipy.signal
import numpy as np
import random
import copy
from queue import Queue
import sys

class PPO(object):

    def __init__(self, state_dim, action_dim, gamma=0.99,
                 entropy_coefficient=0.01, traj=64, clip_param=0.2, 
                 optim_epoch=5, lr=0.01, lr_decay=0., 
                 value_hidden_layers=2, actor_hidden_layers=2, 
                 value_hidden_neurons=32, actor_hidden_neurons=32, 
                 scope="ppo", add_layer_norm=False, continous=True, 
                 training=True):

        if not continous:
            raise NotImplementedError("TODO")

        self.sess  = tf.get_default_session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.traj  = traj
        self.clip_param  = clip_param
        self.optim_epoch = optim_epoch

        self.add_layer_norm  = add_layer_norm
        self.training        = training
        self.continous       = continous

        #####################################
        # Create Object and Value Functions # 
        #####################################


        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, self.s_dim])

            with tf.variable_scope("pi"):
                self.obf = self.create_actor(self.state,
                                             actor_hidden_layers, 
                                             actor_hidden_neurons,
                                             trainable=True)
            with tf.variable_scope("old_pi"):
                    self.old_obf = self.create_actor(self.state,
                                                     actor_hidden_layers,
                                                     actor_hidden_neurons,
                                                     trainable=False)
            with tf.variable_scope("value"):
                    self.value_out = self.create_value(self.state,
                                                       value_hidden_layers,
                                                       value_hidden_neurons)


            ###################################
            # Define Target Network Update Op #
            ###################################

            pi_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/pi'.format(scope))

            old_pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/old_pi'.format(scope))

            self.equal_op = [tpv.assign(pv)
                                for tpv, pv in zip(old_pi_vars, pi_vars)]

            #####################
            # Learning Rate Ops #
            #####################
            ppo_train_step = tf.Variable(0, dtype=tf.int32)

            lr_scale    = tf.Variable(1, dtype=tf.float32)
            lr_decay    = tf.constant(1 - lr_decay, tf.float32)
            decay_lr_op = lr_scale.assign(lr_scale * lr_decay)

            self.update_cstate_op  = (ppo_train_step.assign(ppo_train_step + 1), decay_lr_op)

            ################
            # Training Ops #
            ################
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None])
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim])


            entropy_penalty = tf.reduce_mean(self.obf.entropy()) * (-entropy_coefficient)
            pi_div_piold    = self.obf.prob(self.actions) / (self.old_obf.prob(self.actions) + tf.constant(1e-6))


            #########################################
            #       Surrogate Objectives            #
            # https://arxiv.org/pdf/1707.06347.pdf  #
            #          (6)    and    (7)            #
            #########################################
            policy_advantage  = self.rewards - self.value_out 

            surrogate         = pi_div_piold * policy_advantage
            clipped_surrogate = tf.clip_by_value(pi_div_piold, 1.0 - clip_param, 1.0 + clip_param) * policy_advantage

            pessimistic_surrogate = -tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate))

            value_function_loss = tf.losses.mean_squared_error(self.rewards, self.value_out)
            #############################################
            # Minimize the Entropy of the policies,     #
            # their divergence and the value functions  #
            # mininterpretation of reality! :)          #
            #############################################
            self.total_loss = value_function_loss + pessimistic_surrogate + entropy_penalty
            policy_opt = tf.train.AdamOptimizer(learning_rate=lr * lr_scale).minimize(pessimistic_surrogate + entropy_penalty)
            value_opt  = tf.train.AdamOptimizer(learning_rate=lr * lr_scale).minimize(value_function_loss)

            self.optimizer  = (policy_opt, value_opt)




            #####################################
            # If training use stochastic policy #
            #####################################
            if training:
                self.policy = self.obf.sample()
                self._init_trajectory_sampling()
            else:
                self.policy = self.obf.mode()

    def create_value(self, state, layers, neurons):

        
        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.tanh)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.tanh)

            if self.add_layer_norm:
                x = tf.contrib.layers.layer_norm(x, trainable=False)


        out = tf.layers.dense(x, 
                              1, 
                              activation=None)

        return out[:, 0]

    def create_actor(self, state, layers, neurons, trainable=True):

        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.tanh,
                            trainable=trainable)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.tanh,
                                trainable=trainable)

            if self.add_layer_norm:
                x = tf.contrib.layers.layer_norm(x, trainable=False)


        #############################
        # Define objective function #
        #############################
        obf = None
        if self.continous:
            mean   = tf.layers.dense(x, self.a_dim, 
                                     activation=tf.nn.tanh,
                                     trainable=trainable)

            sigma  = tf.layers.dense(x, self.a_dim,
                                    activation=tf.nn.softplus,
                                    trainable=trainable)


            obf = tf.distributions.Normal(loc=mean, scale=sigma)

        else:
            raise NotImplementedError("TODO")

        return obf

    def predict(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})


    def train(self):
        losses   = 0
        episodes = 0
        while not self.traj_queue.empty():
            trajectory = self.traj_queue.get()

            discounted = scipy.signal.lfilter([1], [1, -self.gamma], trajectory["rewards"][::-1], axis=0)[::-1]
            self.sess.run(self.equal_op)


            loss = 0
            for _ in range(self.optim_epoch):
                _, epoch_l = self.sess.run((self.optimizer, self.total_loss),
                                feed_dict={self.state: trajectory["observations"],
                                           self.actions: trajectory["actions"],
                                           self.rewards: discounted})
                loss += epoch_l

            losses   += loss / self.optim_epoch
            episodes += 1




            self.sess.run(self.update_cstate_op)

            return losses / episodes

    def _init_trajectory_sampling(self):
        self.traj_queue = Queue()

        self.traj_template = {"observations": [],
                              "rewards": [],
                              "terminals": [],
                              "actions": []}

        self.trajectory = copy.deepcopy(self.traj_template)
        self.traj_step  = 0


    def add_experience(self, ob, rew, term, ac):
        self.trajectory["observations"].append(ob)
        self.trajectory["rewards"].append(rew)
        self.trajectory["terminals"].append(term)
        self.trajectory["actions"].append(ac)
        self.traj_step += 1

        if self.traj_step >= self.traj:
            self.trajectory["observations"] = np.array(self.trajectory["observations"])
            self.trajectory["rewards"]      = np.array(self.trajectory["rewards"])
            self.trajectory["terminals"]    = np.array(self.trajectory["terminals"])
            self.trajectory["actions"]      = np.array(self.trajectory["actions"])

            self.traj_queue.put(self.trajectory)
            self.trajectory = copy.deepcopy(self.traj_template)
            self.traj_step  = 0



