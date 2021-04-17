import spacy
import numpy as np
import os, shutil, json, sys
import json, argparse
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, units, heads=1,
                 output_dense = True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.units = units
        self.heads = heads
        self.head_units = self.units // self.heads
        self.output_dense = output_dense
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({'units': self.units,
                       'heads': self.heads,
                       'output_dense': self.output_dense,
                       'activation': self.activation,
                       'use_bias': self.use_bias})
        return config

    @classmethod
    def from_config(cls, config):
        if "head_units" in config:
            config.pop("head_units")
        if "acvivation" in config:
            config['activation'] = config.pop('acvivation')
        return cls(**config)

    def build(self, input_shape):
        # (batch, timesteps, features)
        assert len(input_shape) == 3
        self.seq_length = input_shape[1]
        self.input_size = input_shape[2]
        self.query_kernel = self.add_weight(name = 'query_kernel',
                                            shape = (self.heads, input_shape[-1], self.head_units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.key_kernel = self.add_weight(name = 'key_kernel',
                                            shape = (self.heads, input_shape[-1], self.head_units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.value_kernel = self.add_weight(name = 'value_kernel',
                                            shape = (self.heads, input_shape[-1], self.head_units),
                                            initializer = self.kernel_initializer,
                                            regularizer = self.kernel_regularizer,
                                            constraint = self.kernel_constraint,
                                            trainable = True)
        self.query_bias = self.add_weight(name = 'query_bias',
                                            shape = (self.heads, self.head_units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.key_bias = self.add_weight(name = 'key_bias',
                                            shape = (self.heads, self.head_units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        self.value_bias = self.add_weight(name = 'value_bias',
                                            shape = (self.heads, self.head_units),
                                            initializer = self.bias_initializer,
                                            regularizer = self.bias_regularizer,
                                            constraint = self.bias_constraint,
                                            trainable = True)
        if self.output_dense:
            self.head_kernel = self.add_weight(name = 'head_kernel',
                                                shape = (self.units, self.units),
                                                initializer = self.kernel_initializer,
                                                regularizer = self.kernel_regularizer,
                                                constraint = self.kernel_constraint,
                                                trainable = True)
            self.head_bias = self.add_weight(name = 'head_bias',
                                                shape = (self.units,),
                                                initializer = self.bias_initializer,
                                                regularizer = self.bias_regularizer,
                                                constraint = self.bias_constraint,
                                                trainable = True)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis = 1)
        distribute = lambda x: tf.expand_dims(x, axis = 1)
        queries = tf.matmul(inputs, self.query_kernel) + (distribute(self.query_bias) if self.use_bias else 0)
        keys = tf.matmul(inputs, self.key_kernel) + (distribute(self.key_bias) if self.use_bias else 0)
        values = tf.matmul(inputs, self.value_kernel) + (distribute(self.value_bias) if self.use_bias else 0)
        sims = tf.matmul(queries, tf.transpose(keys, (0, 1, 3, 2))) / np.sqrt(self.units)
        attentions = tf.nn.softmax(sims, axis = -1)
        weighted_sims = tf.matmul(attentions, values)
        flattened_sims = tf.reshape(tf.transpose(weighted_sims, (0, 2, 1, 3)), [-1, self.seq_length, self.units])
        if not self.output_dense:
            return flattened_sims
        outputs = tf.matmul(flattened_sims, self.head_kernel) + (self.head_bias if self.use_bias else 0)
        return self.activation(outputs)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)
