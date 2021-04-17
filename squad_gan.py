import spacy
import numpy as np
import os, shutil, json, sys
import json, argparse
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

from GAN import *
from MultiHeadAttention import *
from DataLoader import *


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str)
    parser.add_argument('-l', '--load', type=str)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    return parser.parse_args()

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
