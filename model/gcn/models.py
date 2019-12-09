from gcn.layers import *
import tensorflow as tf
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCN(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.adj = placeholders['adj']
        self.labels = placeholders['labels']
        self.labels_mask = placeholders['labels_mask']
        self.dropout = placeholders['dropout']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero

        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                        output_dim=FLAGS.hidden2,
                                        adj=self.adj,
                                        act=lambda x: x,
                                        dropout=self.dropout,
                                        logging=self.logging)(self.hidden1)

        self.outputs = tf.nn.softmax(self.hidden2)

        self._loss()
        self._accuracy()

    def _loss(self):
        self.loss = masked_softmax_cross_entropy(self.outputs, self.labels, self.labels_mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.labels, self.labels_mask)


class RGCN(Model):
    def __init__(self, placeholders, num_features, features_nonzero, scope, **kwargs):
        with tf.variable_scope(scope):
            super(RGCN, self).__init__(**kwargs)

            self.inputs = placeholders['features']
            self.adj_1 = placeholders['adj_1']
            self.adj_2 = placeholders['adj_2']
            self.dropout = placeholders['dropout']
            self.input_dim = num_features
            self.features_nonzero = features_nonzero

            self.build()

    def _build(self):
        self.hidden1 = RelationalGraphConvolutionSparse(input_dim=self.input_dim,
                                                        output_dim=FLAGS.hidden1,
                                                        adj_1=self.adj_1,
                                                        adj_2=self.adj_2,
                                                        features_nonzero=self.features_nonzero,
                                                        act=tf.nn.relu,
                                                        dropout=self.dropout,
                                                        logging=self.logging)(self.inputs)

        self.outputs = RelationalGraphConvolution(input_dim=FLAGS.hidden1,
                                                  output_dim=FLAGS.hidden2,
                                                  adj_1=self.adj_1,
                                                  adj_2=self.adj_2,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging)(self.hidden1)



