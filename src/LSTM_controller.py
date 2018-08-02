from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd

import CopulaModel as cm
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from functools import partial
from itertools import product
from rnn_cell_util import (
    GeneratorCellBuilder, GeneratorRNNCellBuilder, mean_field_cell,
    OutputRangeWrapper
)



from tensorflow.python.ops import variable_scope

class RNNCellGenerator(object):
    """Train a model to emit action sequences"""
    def __init__(self, name, m, seq_len, action_set, cell_builder, reuse=False,
        **kwargs):
        self.m  = m
        self.seq_len    = seq_len
        self.action_set = action_set
        self.cell_build = cell_builder
        self.name       = name
        self.action_seq = None
        self.logits     = None
        with tf.variable_scope(name, reuse=reuse):
            self.batch_size = tf.placeholder(tf.int32, [])
            self._build_generator(**kwargs)

    def _build_generator(self, feed_actions=True):
        """Build the RNN action generator
        @feed_actions: Feed one-hot actions taken in previous step, rather than
            logits (from which action was sampled)
        @init_type: How to initialize the sequence generation:
            * train: Train a variable to init with
            * zeros: Send in an all-zeros vector
        """
        # Build the cell
        cell, state = self.cell_build.build_cell_and_init_state(
            self.batch_size, feed_actions
        )
        #initialize one hot vector of action
        feed = tf.zeros((self.batch_size, self.m), dtype=tf.float32)


        # Placeholders to recover policy for updates
        self.rerun         = tf.placeholder_with_default(False, [])
        self.input_actions = tf.placeholder_with_default(
            tf.zeros((1, self.seq_len), dtype=tf.int32), (None, self.seq_len)
        )
        self.coo_actions   = tf.placeholder(tf.int32, (None, 3))
        # Run loopy feed forward
        actions_arr = tf.TensorArray(tf.int32, self.seq_len)
        logits_arr  = tf.TensorArray(tf.float32, self.seq_len)
        for t in range(self.seq_len):
            if t > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # Compute logits for next action using RNN cell
            # state need to be update to deal with the new action set
            logits, state = cell(feed, state)
            #set logits for nodes not in action set to 0
            
            # Samplers to draw actions
            def sample():
                return tf.to_int32(tf.multinomial(logits, 1))
            def rerun_sample():
                return self.input_actions[:, t]
            # If rerunning to apply policy gradients, draw is the input
            draw = tf.reshape(tf.cond(self.rerun, rerun_sample, sample), (-1,))
            # Write to arrays
            logits_arr  = logits_arr.write(t, logits)
            actions_arr = actions_arr.write(t, draw)
            # Update feed- either with the action taken (default), or with
            # the logits output at the previous timestep
            if feed_actions:
                feed = tf.one_hot(draw, self.m)
            else:
                feed = logits
        # Reshape logits to [batch_size, seq_len, n_actions]
        self.logits = tf.transpose(logits_arr.stack(), (1, 0, 2))
        # Reshape action_seq to [batch_size, seq_len]
        self.action_seq = tf.transpose(actions_arr.stack())

    def _get_generated_probabilities(self):
        """Returns a [batch_size, seq_len] Tensor with probabilities for each
           action that was drawn
        """
        input_batch_size = tf.shape(self.input_actions)[0]
        dists            = tf.nn.softmax(self.logits)
        r_dists          = tf.gather_nd(dists, self.coo_actions)
        return tf.reshape(r_dists, (input_batch_size, self.seq_len))

    def _build_discounts_matrix(self, T, gamma):
        """Build lower-triangular matrix of discounts.
        For example for T = 3: D = [[1,       0,     0]
                                   [gamma,   1,     0]
                                   [gamma^2, gamma, 1]]
        Then with R, our N x T incremental rewards matrix, the discounted sum is
            R * D
        """
        power_ltri  = tf.cumsum(
            tf.sequence_mask(tf.range(T)+1, T, dtype=tf.float32), exclusive=True
        )
        gamma_ltri  = tf.pow(gamma, power_ltri)
        gamma_ltri *= tf.sequence_mask(tf.range(T)+1, T, dtype=tf.float32)
        return gamma_ltri

    def get_policy_loss_op(self, incremental_rewards, gamma):
        """Input is a [batch_size, seq_len] Tensor where each entry represents
           the incremental reward for an action on a data point
           incremental reward is log probability of new added edge minus the new added node
        """
        T = tf.shape(incremental_rewards)[1]
        # Form matrix of discounts to apply
        gamma_ltri = self._build_discounts_matrix(T, gamma)
        # Compute future discounted rewards as [batch_size x seq_len] matrix
        future_rewards = tf.matmul(incremental_rewards, gamma_ltri)
        # Compute baseline and advantage
        baseline   = tf.reduce_mean(future_rewards, axis=0)
        advantages = future_rewards - baseline
        # Apply advantage to policy
        policy = self._get_generated_probabilities()
        return tf.reduce_sum(tf.log(policy) * tf.stop_gradient(advantages))

    def get_action_sequence(self, session, batch_size):
        """Sample action sequences"""
        return session.run(self.action_seq, {self.batch_size: batch_size})

    def get_feed(self, actions, **kwargs):
        """Get the feed_dict for the training step.
        @action_seqs: The sequence of actions taken to generate the transformed
            data in this training step.
        Note that we feed `action_seqs` back in and set rerun=True to indicate
        that the exact same sequence of actions should be used in all other
        operations in this step!
        """
        coord   = product(range(actions.shape[0]), range(actions.shape[1]))
        feed    = {
            self.batch_size   : actions.shape[0],
            self.input_actions: actions,
            self.coo_actions:   [[i, j, actions[i, j]] for i, j in coord],
            self.rerun:         True,
        }
        kwargs.update(feed)
        return kwargs





class LSTMGenerator(RNNCellGenerator):
    def __init__(self, name, m, seq_len, action_set, reuse=False, n_stack=1,
        logit_range=4.0, **kwargs):
        # Get LSTM cell builder
        def norm(x):
            return 0.5 * (x + 1.)
        range_wrapper = partial(
            OutputRangeWrapper, output_range=logit_range, norm_op=norm
        )
        cb = GeneratorRNNCellBuilder(
            rnn.BasicLSTMCell, m=m, n_stack=n_stack, wrappers=[range_wrapper]
        )
        # Super constructor
        super(LSTMGenerator, self).__init__(
            name, m, seq_len, action_set, cell_builder=cb, reuse=reuse, **kwargs
        )



def main():
     controller = LSTMGenerator('haha',2,5,[2,3])
     print(controller.m)


if __name__ == '__main__':
    main()
