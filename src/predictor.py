"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf
import tflearn as tfl
import os

class Predictor(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim):

        self.lr = 1e-3
        self.keep_prob = 0.7
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

        self.save_path = "agents/predictor"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():

            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')
            self.rew_ph = tf.placeholder(tf.float32, (None, 1), 'rew')
            self.n_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'n_obs')

            self.l1 = tfl.fully_connected(tf.concat([self.obs_ph, self.act_ph], 1), 300, 'relu', weights_init='xavier')
            self.l1 = tfl.dropout(self.l1, keep_prob=self.keep_prob)

            self.l1_act = tfl.fully_connected(self.l1, 240, 'tanh', weights_init='xavier')
            self.l1_act = tfl.dropout(self.l1_act, keep_prob=self.keep_prob)
            self.l2_act = tfl.fully_connected(self.l1_act, 240, 'tanh', weights_init='xavier')
            self.l2_act = tfl.dropout(self.l2_act, keep_prob=self.keep_prob)
            self.out_state = tfl.fully_connected(self.l2_act, self.obs_dim, 'linear', weights_init='xavier')

            self.l1_rew = tfl.fully_connected(self.l1, 96, 'tanh', weights_init='xavier')
            self.l1_rew = tfl.dropout(self.l1_rew, keep_prob=self.keep_prob)
            self.l2_rew = tfl.fully_connected(self.l1_rew, 64, 'tanh', weights_init='xavier')
            self.l2_rew = tfl.dropout(self.l2_rew, keep_prob=self.keep_prob)
            self.out_rew = tfl.fully_connected(self.l2_rew, 1, 'linear', weights_init='xavier')

            self.state_prediction_loss_vec = tf.square(self.out_state - self.n_obs_ph)
            self.rew_prediction_loss = tf.square(self.out_rew - self.rew_ph)
            self.prediction_loss = tf.reduce_mean(self.state_prediction_loss_vec) + tf.reduce_mean(self.rew_prediction_loss)
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.prediction_loss)

            self.init = tf.global_variables_initializer()


    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def train(self, obs_batch, act_batch, rew_batch, n_obs_batch):
        feed_dict = {self.obs_ph : obs_batch, self.act_ph : act_batch, self.rew_ph : np.expand_dims(rew_batch, 1), self.n_obs_ph : n_obs_batch}
        _, state_loss_vec, rew_loss = self.sess.run([self.optim, self.state_prediction_loss_vec, self.rew_prediction_loss], feed_dict=feed_dict)
        return state_loss_vec, rew_loss

    def predict(self, obs, act):
        """Draw sample from prediction policy """
        feed_dict = {self.obs_ph : obs, self.act_ph : act}

        state, rew = self.sess.run([self.out_state, self.out_rew], feed_dict=feed_dict)
        return state[0], rew[0]

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def save_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, os.path.join(self.save_path, "trained_model"))

    def restore_weights(self):
        with self.g.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))
