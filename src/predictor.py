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

        self.lr = 1e-4
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._build_graph()
        self._init_session()

        self.save_path = "agents/ppo/predictor"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():

            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
            self.n_obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'n_obs')
            self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'act')

            self.l1 = tfl.fully_connected(tf.concat([self.obs_ph, self.act_ph], 1), 180, 'relu', weights_init='xavier')
            self.l2 = tfl.fully_connected(self.l1, 180, 'relu', weights_init='xavier')
            self.out = tfl.fully_connected(self.l2, self.obs_dim, 'linear', weights_init='xavier')

            self.prediction_loss_vec = tf.square(self.out - self.n_obs_ph)
            self.prediction_loss = tf.reduce_mean(self.prediction_loss_vec)
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.prediction_loss)

            self.init = tf.global_variables_initializer()


    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def train(self, obs_batch, act_batch, n_obs_batch):
        feed_dict = {self.obs_ph : obs_batch, self.act_ph : act_batch, self.n_obs_ph : n_obs_batch}
        _, loss_vec = self.sess.run([self.optim, self.prediction_loss_vec], feed_dict=feed_dict)
        return loss_vec

    def predict(self, obs):
        """Draw sample from prediction policy """
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.out, feed_dict=feed_dict)

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
