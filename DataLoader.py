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
                p += 1
            for i, token in enumerate(self.nlp(text)):
                if vector is None:
                    #print("padding  = " + str(padding))
                    #print("numwords = " + str(p))
                    vector = np.zeros((padding, len(token.vector)))
                if i < padding:
                    vector[i, :] = token.vector
            return vector
        elif isinstance(text, (list, np.ndarray)):
            #print("inst-padding : " + str(padding))
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
