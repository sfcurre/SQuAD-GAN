import spacy
import numpy as np
import os, shutil, json, sys
import json, argparse
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict

## TODO: RE-WRIT THE DATALOADER TO LOAD THE DATA IN A "BATCH" FORMAT
# MAPPING 1 CONTEXT TO 1 QUETSION/ ANSWER, Rather Than One To Multiple?


class DataLoader:
    def __init__(self, json_file, batch_size = 16, context_padding = 400, answer_padding = 10, question_padding = 40, shuffle = True):
        self.json = json_file
        self.batch_size = batch_size
        self.context_padding = context_padding
        self.answer_padding = answer_padding
        self.question_padding = question_padding
        self.shuffle = shuffle

        with open(self.json, 'r') as fp:
            data = json.load(fp)['data']
            self.data = []
            #print(data)
            for title_set in data:
                #print(title_set['paragraphs'])
                self.data.extend(title_set['paragraphs'])

        self.ids = sorted(range(len(self.data)), key=lambda i: self.data[i]['context'])
        if shuffle:
            np.random.shuffle(self.ids)

        self.nlp = spacy.load("en_core_web_lg")

    def extract_text_features(self, text, padding):
        if type(text) is str:
            vector = None
            #print("TEXT: " + str(self.nlp(text)))
            p = 0
            for i, token in enumerate(self.nlp(text)):
                if vector is None:
                    vector = np.zeros((padding, len(token.vector)))
                if i < padding:
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
        context_vector = self.extract_text_features(context, self.context_padding)

        C,A,Q = [],[],[]
        for qas in self.data[idx]['qas']:
            C.append(context_vector)
            if qas['answers']:
                answer = '<b>'.join([a['text'] for a in qas['answers']])
            else:
                answer = '<b>'.join([a['text'] for a in qas['plausible_answers']])
            A.append(self.extract_text_features(answer, self.answer_padding))

            question = qas['question']
            Q.append(self.extract_text_features(question, self.question_padding))
        C = np.array(C)
        A = np.array(A)
        Q = np.array(Q)
        # print(C.shape)
        # print(A.shape)
        # print(Q.shape)
        return C,A,Q

    def get_batch_ids(self, index):
        return self.ids[index]
