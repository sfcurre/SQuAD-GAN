import spacy
import numpy as np
import os, shutil, json, sys
import json, argparse
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str)
    parser.add_argument('-l', '--load', type=str)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    return parser.parse_args()

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

class DataLoader:
    def __init__(self, json_file, batch_size = 16, context_padding = 200, answer_padding = 10, question_padding = 40, shuffle = True):
        self.json = json_file
        self.batch_size = batch_size
        self.context_padding = context_padding
        self.answer_padding = answer_padding
        self.question_padding = question_padding
        self.shuffle = shuffle

        with open(self.json, 'r') as fp:
            data = json.load(fp)['data']
            self.data = []
            for title_set in data:
                self.data.extend(title_set['paragraphs'])

        self.ids = sorted(range(len(self.data)), key=lambda i: self.data[i]['context'])
        if shuffle:
            np.random.shuffle(self.ids)

        self.nlp = spacy.load("en_core_web_lg")

    def extract_text_features(self, text, padding):
        if type(text) is str:
            vector = None
            for i, token in enumerate(self.nlp(text)):    
                if vector is None:
                    vector = np.zeros((padding, len(token.vector)))
                vector[i, :] = token.vector
            return vector
        elif isinstance(text, (list, np.ndarray)):
            return np.stack([self.extract_text_features(item, padding) for item in text])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        context = self.data[idx]['context']
        questions, answers = [], []
        for qas in self.data[idx]['qas']:
            questions.append(qas['question'])
            if qas['answers']:
                answers.append('<b>'.join([a['text'] for a in qas['answers']]))
            else:
                answers.append('<b>'.join([a['text'] for a in qas['plausible_answers']]))
        context_vector = self.extract_text_features(context, self.context_padding)
        answer_vectors = self.extract_text_features(answers, self.answer_padding)
        question_vectors = self.extract_text_features(questions, self.question_padding)
        return context_vector[None,...], answer_vectors[None,...], question_vectors[None,...]

    def get_batch_ids(self, index):
        return self.ids[index]

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
        #c_h is shape (1, 500)
        a_h = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100)))(a_in)
        #a_h is shape (1, batch, 100)

        def repeat_vector(args):
            layer_to_repeat = args[0]
            sequence_layer = args[1]
            return tf.keras.layers.RepeatVector(tf.shape(sequence_layer)[1])(layer_to_repeat)

        c_h2 = tf.keras.layers.Lambda(repeat_vector, output_shape=(None, 500)) ([c_h, a_h])

        h1 = tf.keras.layers.Concatenate()([c_h2, a_h, r_in])
        h2 = tf.keras.layers.TimeDistributed(tf.keras.layers.RepeatVector(self.question_padding))(h1)
        h3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(200, return_sequences = True)))(h2)
        q_out = tf.keras.layers.TimeDistributed(tf.keras.layers.GRU(self.embedding_size, return_sequences = True))(h3)
        #q_out is shape (1, batch, question_padding, embedding_size)

        return tf.keras.models.Model([c_in, a_in, r_in], q_out)

    def discriminator_model(self):
        q_in = tf.keras.layers.Input((None, self.question_padding, self.embedding_size))
        
        h1 = tf.keras.layers.Conv1D(128, 3, activation='relu')(q_in)
        h2 = tf.keras.layers.Conv1D(128, 3, activation='relu')(h1)

        h3 = tf.keras.layers.TimeDistributed(MultiHeadAttention(128, 16, output_dense=False))(h2)
        h4 = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences = True)))(h3)
        h5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)))(h4)

        k_out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = 'sigmoid'))(h5)

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
                r = np.random.normal(0, 1, size = [1, a.shape[1], self.input_shapes[2][-1]])

                oq = self.generator.predict_on_batch([c, a, r])
                fake_labels = np.zeros([*oq.shape[:2]])
                true_labels = np.ones_like(fake_labels)
                qs, labels = np.concatenate([q, oq], axis=1), np.concatenate([true_labels, fake_labels], axis=1)

                for _ in range(discriminator_epochs):
                    dl = self.discriminator.train_on_batch(qs, labels)

            loader.on_epoch_end()
            print(f"Discriminator loss - {dl}")

        for i in range(epochs):
            print(f'Epoch {i + 1}:')
            ql = gl = dl = 0
            for b, j in enumerate(np.random.permutation(len(loader))):

                c, a, q = loader[j]
                r = np.random.normal(0, 1, size = [1, a.shape[1], self.input_shapes[2][-1]])
 
                self.generator.trainable = True
                self.discriminator.trainable = False

                for _ in range(question_epochs):
                    ql = self.generator.train_on_batch([c, a, r], q)

                for _ in range(generator_epochs):
                    gl = self.adversarial.train_on_batch([c, a, r], np.ones((1, a.shape[1])))

                if discriminator_epochs:
                    self.generator.trainable = False
                    self.discriminator.trainable = True
                    
                    oq = self.generator.predict_on_batch([c, a, r])
                    fake_labels = np.zeros([*oq.shape[:2]])
                    true_labels = np.ones_like(fake_labels)
                    qs, labels = np.concatenate([q, oq], axis=1), np.concatenate([true_labels, fake_labels], axis=1)

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

def lookup(embedded_vector, nlp):
    sentence = []
    for vec in embedded_vector:
        sentence.append(max(nlp.vocab, key = lambda x: (x.vector * vec).sum()).text)
    return sentence

#=============================================
def main():
    args = parse_args()

    loader = DataLoader(args.json, batch_size=1, context_padding=200, answer_padding=10, question_padding=40)

    #shapes are in the order c a r
    gan = GAN([(200, 300), (None, 10, 300), (None, 100)], embedding_size=300, question_padding=40)

    if args.load:
        gan.load(args.load)

    gan.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001))

    gan.generator.summary()
    gan.discriminator.summary()
    gan.adversarial.summary()
    
    gan.fit(loader, args.epochs, generator_epochs=1, discriminator_epochs=1, question_epochs=1, pretrain_epochs=1)

    c, a, q = loader[0]
    r = np.random.normal(0, 1, size = [1, a.shape[1], 100])
    output = gan.generator.predict_on_batch([c, a, r])
    output = np.squeeze(output)

    np.save('gan_output.npy', output)

    nlp = loader.nlp
    context = lookup(c[0], nlp)
    outs = [lookup(out, nlp) for out in output]

    with open('gan_output.txt', 'w') as fp:
        print(context, file = fp)
        print()
        print(*outs, sep='\n', file = fp)

if __name__ == '__main__':
    main()
