import tensorflow as tf
import scipy.signal
import numpy as np
import random
import copy
from queue import Queue
import sys
import pandas as pd

class PPO(object):

    def __init__(self, state_dim, action_dim, gamma=0.95,
                 traj=32, clip_param=0.2, optim_epoch=5, lr=0.001,
                 value_hidden_layers=0, actor_hidden_layers=0, 
                 value_hidden_neurons=100, actor_hidden_neurons=200, 
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

            pi_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           '{}/pi'.format(scope))

            old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           '{}/old_pi'.format(scope))

            self.equal_op = [tpv.assign(pv)
                                for tpv, pv in zip(old_pi_vars, pi_vars)]

            ################
            # Training Ops #
            ################
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None])
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim])


            pi_div_piold = tf.exp(self.obf.log_prob(self.actions) - tf.stop_gradient(self.old_obf.log_prob(self.actions)))


            #########################################
            #       Surrogate Objectives            #
            # https://arxiv.org/pdf/1707.06347.pdf  #
            #          (6)    and    (7)            #
            #########################################
            policy_advantage  = self.rewards - tf.stop_gradient(self.value_out)


            surrogate         = pi_div_piold * policy_advantage
            clipped_surrogate = tf.clip_by_value(pi_div_piold, 1.0 - clip_param, 1.0 + clip_param) * policy_advantage

            pessimistic_surrogate = -tf.reduce_mean(tf.minimum(surrogate, clipped_surrogate))

            value_function_loss = tf.reduce_mean(tf.square(self.value_out - self.rewards)) 

            self.total_loss = pessimistic_surrogate #(self.obf.prob(self.actions), self.old_obf.prob(self.actions))
            policy_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(pessimistic_surrogate)
            value_opt  = tf.train.AdamOptimizer(learning_rate=lr).minimize(value_function_loss)

            self.optimizer  = (policy_opt, value_opt)


            #####################################
            # If training use stochastic policy #
            #####################################
            if training:
                self.policy = self.obf.sample()
            else:
                self.policy = self.obf.mode()

    def create_value(self, state, layers, neurons):

        
        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.relu)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.relu)

            if self.add_layer_norm:
                x = tf.contrib.layers.layer_norm(x, trainable=False)


        out = tf.layers.dense(x, 
                              1, 
                              activation=None)

        return out[:, 0]

    def create_actor(self, state, layers, neurons, trainable=True):

        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.relu,
                            trainable=trainable)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.relu,
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

            obf = tf.distributions.Normal(loc=2 * mean, scale=sigma)

        else:
            raise NotImplementedError("TODO")

        return obf

    def predict(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})


    def train(self, trajectory):
        final_state = trajectory["final_state"].reshape(1, -1)
        final_value = self.sess.run((self.value_out), feed_dict={self.state: final_state})

        trajectory = pd.DataFrame(data={"observations":trajectory["observations"],
                                        "actions": trajectory["actions"],
                                        "rewards": trajectory["rewards"],
                                        "terminals": trajectory["terminals"]})


        rewards     = trajectory["rewards"] 
        rewards     = np.clip(rewards, -1, 1)
        rewards     = np.append(rewards, [final_value])
        is_terminal = 1 - trajectory["terminals"]

        trajectory_length = len(is_terminal)


        discounted = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            rewards[i] = discounted[i] = rewards[i] + self.gamma * rewards[i + 1] * is_terminal[i]


        trajectory["rewards"] = discounted



        self.sess.run(self.equal_op)

        total_loss = 0
        training_samples = 2 * trajectory.shape[0] // self.traj
        for _ in range(training_samples):
            sample = trajectory.sample(self.traj)
            obs  = np.vstack(sample["observations"])
            acs  = np.vstack(sample["actions"])
            rews = np.asarray(sample["rewards"])



            loss = 0
            for _ in range(self.optim_epoch):
                _, epoch_l = self.sess.run((self.optimizer, self.total_loss),
                                feed_dict={self.state: obs,
                                           self.actions: acs,
                                           self.rewards: rews})


                # print(epoch_l)
                loss += abs(epoch_l)


            total_loss += abs(loss / self.optim_epoch)

        return total_loss / training_samples

