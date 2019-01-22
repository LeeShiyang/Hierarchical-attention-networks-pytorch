"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle

class MyDataset(Dataset):

    def __init__(self, data_path,label_path, dict_path):
        super(MyDataset, self).__init__()

        texts, labels = [],[]
        max_length_sentences = 0
        max_length_word = 0
        count = 0
        max_count = 0
        with open(data_path) as txt_file:
            reader = txt_file.readlines()
            for line in reader:
                super_con = []
                super_concepts = line.strip().split('. ')
                if(len(super_concepts) > max_length_sentences):
                    max_length_sentences = len(super_concepts)
                    max_count = count
                for super_concept in super_concepts:
                    concept = super_concept.split(' ')
                    if(len(concept) > max_length_word):
                        max_length_word = len(concept)
                    super_con.append(concept)
                texts.append(super_con)
                count += 1

        label_name = []

        with open(label_path) as txt_file:
            reader = txt_file.readlines()
            for line in reader:
                label_txt = line.strip()
                if label_txt not in label_name:
                    label_name.append(label_txt)
                label_id = label_name.index(label_txt)
                labels.append(label_id)

        self.texts = texts
        self.labels = labels
        self.label_name = label_name
        data = open(dict_path,'rb')
        self.dict = pickle.load(data)

        print('max_length_sentences',max_length_sentences)
        print('max_length_word',max_length_word)
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(self.label_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in sentences] for sentences
            in text]

        for sentences in document_encode:
            if len(sentences) < self.max_length_word:
                extended_words = [-1 for _ in range(self.max_length_word - len(sentences))]
                sentences.extend(extended_words)

        if len(document_encode) < self.max_length_sentences:
            extended_sentences = [[-1 for _ in range(self.max_length_word)] for _ in
                                  range(self.max_length_sentences - len(document_encode))]
            document_encode.extend(extended_sentences)

        document_encode = [sentences[:self.max_length_word] for sentences in document_encode][
                          :self.max_length_sentences]

        document_encode = np.stack(arrays=document_encode, axis=0)
        document_encode += 1

        return document_encode.astype(np.int64), label


if __name__ == '__main__':
    test = MyDataset(data_path="/disk/home/klee/data/cs_merged_tokenized_superspan_HANs.txt", label_path='/disk/home/klee/data/cs_merged_label',dict_path="/disk/home/klee/data/cs_merged_tokenized_dictionary.bin")
    doc = test[0]
    print (test.__getitem__(index=1)[0].shape)
