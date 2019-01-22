"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class MyDataset(Dataset):

    def __init__(self, data_path,label_path, dict_path, max_length_sentences=30, max_length_word=35):
        super(MyDataset, self).__init__()

        texts, labels = [],[]
        with open(data_path) as txt_file:
            reader = txt_file.readlines()
            for line in reader:
                super_con = []
                for super_concept in line.strip().split('.'):
                    concept = super_concept.split(' ')
                    if '' in concept:
                        concept.remove('')
                    super_con.append(concept)
                texts.append(super_con)
        label_name = []

        with open(label_path) as txt_file:
            reader = txt_file.readlines()
            for line in reader:
                label_txt = line.strip()
                if label_txt not in label_name:
                    label_name.append(label_txt)
                label_id = label_name.index(label_txt)
                labels.append(label_id)

        import pdb;
        pdb.set_trace()




        self.texts = texts
        self.labels = labels
        self.label_name = label_name
        self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
        self.dict = [word[0] for word in self.dict]
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        document_encode = [
            [self.dict.index(word) if word in self.dict else -1 for word in word_tokenize(text=sentences)] for sentences
            in
            sent_tokenize(text=text)]

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
    test = MyDataset(data_path="/disk/home/klee/data/cs_merged_tokenized_text_HAN.txt", label_path='/disk/home/klee/data/cs_merged_label',dict_path="../data/glove.6B.50d.txt")
    print (test.__getitem__(index=1)[0].shape)
