from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf

from gcn.models import *
from gcn.utils import *


def get_reward_simple(selected_list, labels):

    count = 0
    for id in selected_list:
        if labels[id] == 1:
            count += 1

    return count / len(selected_list)


def model_train(dataset):

    # Load data
    adj_norm_1, adj_norm_2, adj_1, adj_2, features, labels = load_data_rgcn(dataset)
    num_nodes = adj_norm_1[2][0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = get_placeholder()

    # Create Model
    model_rgcn_main = RGCN(placeholders, num_features, features_nonzero, "main")
    model_rgcn_target = RGCN(placeholders, num_features, features_nonzero, "target")
    # state_representations = model_rgcn.outputs

    # Create Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    selected_list = []
    # Training
    for epoch in range(FLAGS.epochs):
        feed_dict = construct_feed_dict(adj_norm_1, adj_norm_2, features, FLAGS.dropout, placeholders)
        # outs = sess.run([model_gcn.opt_op, model_gcn.loss, state_representations], feed_dict=feed_dict)


        # RL
        """
        pseudo code 
        
        selected_node_id = RL_model.act(state)
        reward = get_reward(selected_node_id, labels)
        RL_model.update(selected_node_id, state, reward)
        """

        # Update adjacency matrices
        selected_node_id = None
        selected_list.append(selected_node_id)
        reward = get_reward_simple(selected_list, labels)
        adj_norm_1, adj_norm_2, adj_1, adj_2 = update_adj(selected_node_id, adj_1, adj_2)

        # Update RL model


if __name__ == '__main__':

    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')

    dataset = "BlogCatalog"
    model_train(dataset)

