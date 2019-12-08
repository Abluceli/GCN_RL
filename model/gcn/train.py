from gcn.utils import *
from gcn.models import *
import tensorflow as tf
import scipy.io as sio
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_reward(sess, selected_id_list):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_gcn(FLAGS.dataset, selected_id_list)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model = GCN(placeholders, input_dim=features[2][1], logging=True)

    epochs = 100
    for i in range(epochs):
        feed_dict = construct_feed_dict(features, adj, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)




