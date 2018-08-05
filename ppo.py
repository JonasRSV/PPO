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
                 entropy_coefficient=0.01, traj=64, clip_param=0.2, 
                 optim_epoch=5, lr=0.0001, lr_decay=0., 
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

            pi_vars     = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                           '{}/pi'.format(scope))

            old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
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


            pi_div_piold    = tf.exp(self.obf.log_prob(self.actions) - self.old_obf.log_prob(self.actions))


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
            self.total_loss = pessimistic_surrogate + value_function_loss #(self.obf.prob(self.actions), self.old_obf.prob(self.actions))
            policy_opt = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(pessimistic_surrogate)
            value_opt  = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(value_function_loss)

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
                            activation=tf.nn.elu)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.elu)

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


    def train(self, trajectory):
        final_state = trajectory["final_state"].reshape(1, -1)
        final_value = self.sess.run((self.value_out), feed_dict={self.state: final_state})

        trajectory = pd.DataFrame(data={"observations":trajectory["observations"],
                                        "actions": trajectory["actions"],
                                        "rewards": trajectory["rewards"],
                                        "terminals": trajectory["terminals"]})


        rewards     = trajectory["rewards"] 
        rewards     = np.append(rewards, [final_value])
        is_terminal = 1 - trajectory["terminals"]

        trajectory_length = len(is_terminal)


        discounted = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            discounted[i] = rewards[i] + self.gamma * rewards[i + 1] * is_terminal[i]

        # trajectory["rewards"] = discounted

        
        total_loss = 0
        training_samples = 2 * trajectory.shape[0] // self.traj
        for _ in range(training_samples):
            sample = trajectory.sample(self.traj)
            obs  = np.vstack(sample["observations"])
            acs  = np.vstack(sample["actions"])
            rews = np.asarray(sample["rewards"])

            self.sess.run(self.equal_op)

            loss = 0
            for _ in range(self.optim_epoch):
                _, epoch_l = self.sess.run((self.optimizer, self.total_loss),
                                feed_dict={self.state: obs,
                                           self.actions: acs,
                                           self.rewards: rews})

                loss += epoch_l

            total_loss += loss / self.optim_epoch


        self.sess.run(self.update_cstate_op)

        return total_loss / training_samples

