import spacy
import numpy as np
import os, shutil, json, sys
import json, argparse
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

from MultiHeadAttention import *

class GAN:
    def __init__(self, input_shapes, embedding_size, question_padding):
        #input shapes is in the order context, answers, noise
        self.input_shapes = input_shapes
        self.embedding_size = embedding_size
        self.question_padding = question_padding

        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.adversarial = self.adversarial_model()

    def generator_model(self):
        c_in = tf.keras.layers.Input(self.input_shapes[0], name = 'c_in')
        a_in = tf.keras.layers.Input(self.input_shapes[1], name = 'a_in')
        r_in = tf.keras.layers.Input(self.input_shapes[2], name = 'r_in')

        c_h = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200))(c_in)
        a_h = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100))(a_in)

        h1 = tf.keras.layers.Concatenate()([c_h, a_h, r_in])
        h2 = tf.keras.layers.RepeatVector(self.question_padding)(h1)
        h3 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200, return_sequences = True))(h2)
        q_out = tf.keras.layers.GRU(self.embedding_size, return_sequences = True)(h3)
        #q_out is shape (1, batch, question_padding, embedding_size)

        return tf.keras.models.Model([c_in, a_in, r_in], q_out)

    def discriminator_model(self):
        q_in = tf.keras.layers.Input((None, self.question_padding, self.embedding_size))

        h1 = tf.keras.layers.Conv1D(128, 3, activation='relu')(q_in)
        h2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(h1)

        h3 = MultiHeadAttention(128, 16, output_dense=False)(h2)
        h4 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences = True))(h3)
        h5 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(h4)

        k_out = tf.keras.layers.Dense(1, activation = 'sigmoid')(h5)

        return tf.keras.models.Model(q_in, k_out)

    def adversarial_model(self):
        c_in = tf.keras.layers.Input(self.input_shapes[0], name = 'c_in')
        a_in = tf.keras.layers.Input(self.input_shapes[1], name = 'a_in')
        r_in = tf.keras.layers.Input(self.input_shapes[2], name = 'r_in')

        q_out = self.generator([c_in, a_in, r_in])

        k_out = self.discriminator(q_out)

        return tf.keras.models.Model([c_in, a_in, r_in], k_out)

    def compile(self, optimizer=tf.keras.optimizers.Adam(), adversarial_loss='binary_crossentropy', question_loss='cosine_similarity'):
        self.optimizer = optimizer
        self.adversarial_loss = adversarial_loss
        self.question_loss = question_loss
        self.discriminator.compile(optimizer=self.optimizer, loss=self.adversarial_loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        self.generator.compile(optimizer=self.optimizer, loss=self.question_loss)
        self.adversarial.compile(optimizer=self.optimizer, loss=self.adversarial_loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
        self.compiled = True

    def fit(self, loader, epochs, generator_epochs = 1, discriminator_epochs = 1, question_epochs = 1, pretrain_epochs = 1):
        if not self.compiled:
            raise AssertionError("Model must be compiled first.")
        for i in range(pretrain_epochs):
            print(f'Pretrain Epoch {i + 1}:')
            dl = 0
            self.generator.trainable = False
            self.discriminator.trainable = True

            for b, j in enumerate(np.random.permutation(len(loader))):

                c, a, q = loader[j]
                r = np.random.normal(0, 1, size = [len(a), self.input_shapes[2][0]])

                oq = self.generator.predict_on_batch([c, a, r])
                fake_labels = np.zeros([len(a)])
                true_labels = np.ones_like(fake_labels)
                qs, labels = np.concatenate([q, oq]), np.concatenate([true_labels, fake_labels])

                for _ in range(discriminator_epochs):
                    dl = self.discriminator.train_on_batch(qs, labels)

            loader.on_epoch_end()
            print(f"Discriminator loss - {dl}")

        for i in range(epochs):
            print(f'Epoch {i + 1}:')
            ql = gl = dl = 0
            for b, j in enumerate(np.random.permutation(len(loader))):

                c, a, q = loader[j]
                r = np.random.normal(0, 1, size = [len(a), self.input_shapes[2][0]])

                self.generator.trainable = True
                self.discriminator.trainable = False

                for _ in range(question_epochs):
                    ql = self.generator.train_on_batch([c, a, r], q)

                for _ in range(generator_epochs):
                    gl = self.adversarial.train_on_batch([c, a, r], np.ones(len(a)))

                if discriminator_epochs:
                    self.generator.trainable = False
                    self.discriminator.trainable = True

                    oq = self.generator.predict_on_batch([c, a, r])
                    fake_labels = np.zeros([len(a)])
                    true_labels = np.ones_like(fake_labels)
                    qs, labels = np.concatenate([q, oq]), np.concatenate([true_labels, fake_labels])

                    for _ in range(discriminator_epochs):
                        dl = self.discriminator.train_on_batch(qs, labels)

                if b % 500 == 0:
                    print(f"Batch {b}/{len(loader)}")
                    print(f"Question loss      - {ql}")
                    print(f"Generator loss     - {gl}")
                    print(f"Discriminator loss - {dl}")

            loader.on_epoch_end()
            if i % 10 == 0:
                self.generator.save('models/generator.h5', save_format = 'h5')
                self.discriminator.save('models/discriminator.h5', save_format = 'h5')
                self.adversarial.save('models/adversarial.h5', save_format = 'h5')

    def load(self, load_directory):
        load_dir = load_directory.rstrip('/')
        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        self.generator = tf.keras.models.load_model(f'{load_dir}/generator.h5', custom_objects=custom_objects)
        self.discriminator = tf.keras.models.load_model(f'{load_dir}/discriminator.h5', custom_objects=custom_objects)
        self.adversarial = self.adversarial_model()
