import tensorflow as tf
import numpy as np
import random
import copy
import sys
import pandas as pd

class Normal(object):
    ###########################
    # Credit OpenAI Baselines #
    ###########################

    def __init__(self, mu, log_sigma, exp_scale=1):
        self.mu        = mu
        self.log_sigma = log_sigma
        self.sigma     = tf.exp(log_sigma)
        self.exp_scale = exp_scale


    def mode(self):
        return self.mu

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mu) / self.sigma), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.log_sigma, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.log_sigma + .5 * np.log(2.0 * np.pi * np.e), axis=-1)\
                * self.exp_scale

    def sample(self):
        return self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))\
                * self.exp_scale

    def prob(self, x):
        return tf.exp(-self.neglogp(x))


class Softmax():
    ###########################
    # Credit Baselines OpenAI #
    ###########################
    def __init__(self, logits, exp_scale):
        self.logits    = logits
        self.exp_scale = exp_scale

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=one_hot_actions)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1) * self.exp_scale

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=tf.float32)
        return tf.argmax(self.logits - self.exp_scale * tf.log(-tf.log(u)), axis=-1)

    def prob(self, x):
        return tf.exp(-self.neglogp(x))


class PPO(object):

    def __init__(self, state_dim, action_dim, gamma=0.95, lam=0.95,
                 entropy_coefficient=0.001, value_coefficient=0.5,
                 batch=64, clip_param=0.2, optim_epoch=5, lr=0.001,
                 lr_decay=0.0, exp_decay=0.0, storage=64,
                 value_hidden_layers=1, actor_hidden_layers=1, 
                 value_hidden_neurons=64, actor_hidden_neurons=64, 
                 scope="ppo", add_layer_norm=False, continous=True, 
                 training=True):

        self.sess  = tf.get_default_session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.gamma = gamma
        self.lam   = lam

        self.batch       = batch
        self.clip_param  = clip_param
        self.optim_epoch = optim_epoch

        self.add_layer_norm  = add_layer_norm
        self.training        = training
        self.continous       = continous

        self.storage            = storage
        self.trajectory_storage = pd.DataFrame()

        #############
        # Decay Ops #
        #############

        lr_scale  = tf.Variable(1, dtype=tf.float32, trainable=False)
        exp_scale = tf.Variable(1, dtype=tf.float32, trainable=False)

        lr_decay  = tf.constant(1 - lr_decay, dtype=tf.float32)
        exp_decay = tf.constant(1 - exp_decay, dtype=tf.float32)

        self.decay_u_op = (lr_scale.assign(lr_scale * lr_decay),
                          exp_scale.assign(exp_scale * exp_decay))

        #####################################
        # Create Object and Value Functions # 
        #####################################


        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [None, self.s_dim])

            with tf.variable_scope("pi"):
                self.obf = self.create_actor(self.state,
                                             actor_hidden_layers, 
                                             actor_hidden_neurons,
                                             exp_scale,
                                             trainable=True)
            with tf.variable_scope("old_pi"):
                    self.old_obf = self.create_actor(self.state,
                                                     actor_hidden_layers,
                                                     actor_hidden_neurons,
                                                     exp_scale,
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

            pi_train_vars    = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 '{}/pi'.format(scope))

            value_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 '{}/value'.format(scope))

            train_vars = pi_train_vars + value_train_vars

            ################
            # Training Ops #
            ################
            self.advantages = tf.placeholder(dtype=tf.float32, shape=[None])
            self.vtarget    = tf.placeholder(dtype=tf.float32, shape=[None])

            self.actions = None
            if self.continous:
                self.actions = tf.placeholder(dtype=tf.float32, shape=[None, self.a_dim])
            else:
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None, 1])

            pi_a_likelihood    = self.obf.prob(self.actions)
            oldpi_a_likelihood = self.old_obf.prob(self.actions)
            pi_div_piold       = pi_a_likelihood / oldpi_a_likelihood

            mean, variance = tf.nn.moments(self.advantages, axes=0)
            normalized_adv = (self.advantages - mean) / tf.sqrt(variance)

            #########################################
            #       Surrogate Objectives            #
            # https://arxiv.org/pdf/1707.06347.pdf  #
            #          (6)    and    (7)            #
            #########################################
            surrogate         = pi_div_piold * normalized_adv
            clipped_surrogate = tf.clip_by_value(pi_div_piold, 1.0 - clip_param, 1.0 + clip_param) * normalized_adv

            LCLIP = tf.minimum(surrogate, clipped_surrogate)
            VF = tf.square(self.value_out - self.vtarget)

            c1 = value_coefficient
            c2 = entropy_coefficient
            
            entropy_bonus = self.obf.entropy()

            ############################################
            # https://arxiv.org/pdf/1707.06347.pdf (9) #
            ############################################
            self.Lt = tf.reduce_mean(-LCLIP + c1 * VF + c2 * -entropy_bonus)

            gradients = tf.gradients(self.Lt, train_vars)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr * lr_scale)\
                                .apply_gradients(zip(gradients, train_vars))

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

    def create_actor(self, state, layers, neurons, exp_scale, trainable=True):

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
            mean = tf.layers.dense(x, self.a_dim, 
                                   activation=tf.tanh,
                                   trainable=trainable)

            log_sigma = tf.layers.dense(x, self.a_dim,
                                    kernel_initializer=tf.zeros_initializer(),
                                    activation=tf.nn.tanh,
                                    trainable=trainable)

            obf = Normal(mean, log_sigma, exp_scale)
        else:
            logits = tf.layers.dense(x, self.a_dim,
                                     activation=tf.nn.softmax,
                                     trainable=trainable)

            obf = Softmax(logits, exp_scale)

        return obf

    def predict(self, state):
        return self.sess.run(self.policy, feed_dict={self.state: state})


    def train(self, trajectory):
        states      = np.vstack(trajectory["observations"] + [trajectory["final_state"]])
        values      = self.sess.run((self.value_out), feed_dict={self.state: states})

        trajectory = pd.DataFrame(data={"observations":trajectory["observations"],
                                        "actions": trajectory["actions"],
                                        "rewards": trajectory["rewards"],
                                        "terminals": trajectory["terminals"]})


        rewards     = trajectory["rewards"]
        rewards     = np.append(rewards, [0])
        is_terminal = 1 - trajectory["terminals"]

        trajectory_length = len(is_terminal)

        #################################################
        # GAE https://arxiv.org/pdf/1707.06347.pdf (11) #
        #################################################
        adv = np.zeros(trajectory_length)
        for i in reversed(range(trajectory_length)):
            delta      = rewards[i] + (self.gamma * values[i + 1] - values[i]) * is_terminal[i]
            rewards[i] = adv[i] = delta + self.gamma * self.lam * rewards[i + 1] * is_terminal[i]


        trajectory["adv"]          = adv
        trajectory["value_target"] = values[:-1] + adv

        self.trajectory_storage = pd.concat([self.trajectory_storage, trajectory])
        if self.trajectory_storage.shape[0] > self.storage:
            self.sess.run(self.equal_op)
            total_loss = 0
            training_samples = self.trajectory_storage.shape[0] // min(self.batch, self.trajectory_storage.shape[0])
            for _ in range(training_samples):
                sample = self.trajectory_storage.sample(min(self.batch, self.trajectory_storage.shape[0]))
                obs   = np.vstack(sample["observations"])
                acs   = np.vstack(sample["actions"])
                vtarg = np.asarray(sample["value_target"])
                adv   = np.asarray(sample["adv"])

                loss = 0
                for _ in range(self.optim_epoch):
                    _, epoch_l = self.sess.run((self.optimizer, self.Lt),
                                    feed_dict={self.state: obs,
                                               self.actions: acs,
                                               self.vtarget: vtarg,
                                               self.advantages: adv})



                    loss += abs(epoch_l)

                total_loss += abs(loss / self.optim_epoch)

            self.sess.run(self.decay_u_op)
            self.trajectory_storage = pd.DataFrame()
            return total_loss / training_samples

        return 0


