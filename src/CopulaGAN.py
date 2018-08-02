from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import rvine


ADAM = tf.train.AdamOptimizer
SGD  = tf.train.GradientDescentOptimizer


class CopulaGAN(object):
    def __init__(self, generator, dims, d_lr, g_lr, discriminator = None,
                 mse_term=1.0, mse_layer=None, d_trainer=ADAM, g_trainer=SGD,
                 reuse=False, gamma=0.0):
        self.discriminator = discriminator
        self.generator = generator
        # Note that we inherit the data dimensions from the discriminator
        self.dims = dims
        self.d    = np.prod(self.dims) # Flattened size
        # Build training operations
        self.d_train_op = None
        self.g_train_op = None
        self.batch_size = self.generator.batch_size
        self.reuse      = reuse
        # Build model graph
        self._build(d_lr, g_lr, mse_term, mse_layer, d_trainer, g_trainer,
            gamma)

    def _build(self, d_lr, g_lr, mse_term, mse_layer, d_trainer, g_trainer,
        gamma):
        T = self.generator.seq_len
        # Placeholders for basic input data
        self.data  = tf.placeholder(tf.float32, (None, self.d))
        batch_size = tf.shape(self.data)[0]

        ### GENERATOR LOSS
        ###
        # Define generative loss for training
        # G_loss_all is a batch_size x (T+1) matrix
        if not discriminator:
            G_loss_all = self.generator.get_loss()
        else:
            G_loss_all = self.discriminator.get_loss_op(D_tf_g, mean=False)
        self.G_loss_deltas = G_loss_all[:,1:]-G_loss_all[:,:-1]



        # Get the change in loss between each incremental transformation
        # G_loss_deltas is a [batch_size, T] Tensor
        self.G_loss_deltas = G_loss_all[:, 1:] - G_loss_all[:, :-1]
        #get policy

        # Get the policy loss op
        q = self.generator.get_policy_loss_op(self.G_loss_deltas, gamma)

    def train_step(self, session, data, n_gen_steps, n_sample=1):
         # Update generator
        for _ in range(n_gen_steps):
            # Define training op
            g_loss, summary, _, g_loss_deltas = session.run(
                [self.G_loss, self.dg_summary, self.g_train_op,
                 self.G_loss_deltas], g_feed
            )
        # Return losses
        return d_loss, g_loss, summary, g_loss_deltas, tf_seqs



class PolicyGradient(self):
    def __init__(self, g_i,g_o, dims, g_lr,
                 mse_term=1.0, mse_layer=None, g_trainer=SGD,
                 reuse=False, gamma=0.0):
        self.g_i = g_t
        self.g_o = g_n
        # Note that we inherit the data dimensions from the discriminator
        self.dims = dims
        self.d    = np.prod(self.dims) # Flattened size
        # Build training operations
        self.g_train_op = None
        self.batch_size = self.generator.batch_size
        self.reuse      = reuse
        # Build model graph
        self._build(g_lr, mse_term, mse_layer,g_trainer,
            gamma)
    #assemble the controller and copula model here
    #should return N*T tensor of action logits


    def _build(g_lr, mse_term, mse_layer,g_trainer,
        gamma):
        #build computation graph
        # Get the policy loss op
        T = self.g_i.seq_len

        # Placeholders for basic input data
        self.data  = tf.placeholder(tf.float32, (None, self.d))
        batch_size = tf.shape(self.data)[0]
        #intree include a sequence of nodes to be added an edge
        self.edges = tf.TensorArray()
        self.intree= tf.TensorArray(tf.int32, T+1)
        #outtree include a sequence of nodes to join the tree
        self.outtree = tf.TensorArray(tf.int32,T+1)

        #get logits for each edge addded to the tree
        with tf.name_scope("compute_loss"):
             D_tf_g_array = tf.TensorArray(tf.float32, T+1)
             for i in range(T+1):
                 d_tf_g = tf.log(self.edges[i].get_likehood(self.data))-tf.log(self.data)
                 D_tf_g_array = D_tf_g_array.write(i, d_tf_g)
            # D_tf_g is reshaped to [batch_size, T+1, dim]
             D_tf_g = tf.transpose(D_tf_g_array.stack())
        self.G_loss_deltas = D_tf_g[:,1:]-D_tf_g[:,:-1]
        q = self.g_i.get_policy_loss_op(self.G_loss_deltas, gamma)

         # Define generative operation for selecting in tree nodes
        g_vars = [
            v for v in tf.trainable_variables()
            if v.name.startswith(self.g_i.name)
        ]
        g_update_ops = [
            u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if u.name.startswith(self.g_i.name)
        ]
        with tf.variable_scope(self.g_i.name, reuse=self.reuse):
            g_step = tf.Variable(0, trainable=False)
            # Note: This is necessary for batch_norm to be handled correctly
            with tf.control_dependencies(g_update_ops):
                self.g_train_op = g_trainer(g_lr).minimize(q,
                    global_step=g_step, var_list=g_vars)

        # Define generative operation for selecting out of tree nodes
        g_o_vars = [
            v for v in tf.trainable_variables()
            if v.name.startswith(self.g_o.name)
        ]
        g_o_update_ops = [
            u for u in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if u.name.startswith(self.g_o.name)
        ]
        with tf.variable_scope(self.g_o.name, reuse=self.reuse):
            g_step = tf.Variable(0, trainable=False)
            # Note: This is necessary for batch_norm to be handled correctly
            with tf.control_dependencies(g_o_update_ops):
                self.g_train_op = g_trainer(g_lr).minimize(q,
                    global_step=g_step, var_list=g_o_vars)

        #logging
        self.G_loss = D_tf_g
        self.G_loss_summary = tf.summary.scalar("gen_loss", self.G_loss)
        loss_summary   = tf.summary.scalar("loss", self.G_loss)
        ### Saver
        vars_list = [
            v for v in tf.trainable_variables()
            if v.name.startswith(self.g_i.name)
        ]
        self.saver = tf.train.Saver(var_list=vars_list)


    def train_step(self,session,data):
        #update the policy gradient of the generator
        in_seq = self.g_i.get_action_sequence(session, data.shape[0])
        out_seq = self.g_o.get_action_sequence(session,data.shape[0])
        g_i_feed = self.g_i.get_feed(in_seq)
        g_o_feed = self.g_o.get_feed(out_seq)
        # g_feed.update({self.data: data_rep})
        # Define training op
        g_loss, summary, _, g_loss_deltas = session.run(
            [self.G_loss, self.G_loss_summary, self.g_train_op,
             self.G_loss_deltas], g_i_feed,g_o_feed
        )
        return g_loss,summary,g_loss_deltas,in_seq,out_seq

    def save(self, session, save_path):
        _ = self.saver.save(session, save_path)

    def restore(self, session, save_path):
        self.saver.restore(session, save_path)
