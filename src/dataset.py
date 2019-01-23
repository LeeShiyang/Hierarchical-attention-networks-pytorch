"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pickle


def get_label(label):
    try:
        return eval(label.strip())[0]
    except Exception as e:
        pass

    return label


class MyDataset(Dataset):
    def __init__(self, data_path, label_path, dict_path, ImportanceFeatureMatsFile, max_vocab, class_idsFile, VvFile, model_save_path):
        self.Vv = pickle.load(open(VvFile, 'rb'))
        super(MyDataset, self).__init__()

        texts, labels = [], []
        max_length_sentences = 0
        max_length_word = 0
        count = 0
        docind_withMaxLength = 0
        text_lines = open(data_path).readlines()
        labels_lines = open(label_path).readlines()

        self.model_gensim = Word2Vec.load(model_save_path)

        class_ids = pickle.load(open(class_idsFile, 'rb'))
        class_id2ind = {id: ind for ind, id in enumerate(class_ids)}

        for line, label in zip(text_lines, labels_lines):
            label = get_label(label.strip())

            if label in class_id2ind:
                label_id = class_id2ind.get(label)
                labels.append(label_id)

                super_con = []
                super_concepts = line.strip().split('. ')
                if(len(super_concepts) > max_length_sentences):
                    max_length_sentences = len(super_concepts)
                    docind_withMaxLength = count
                for super_concept in super_concepts:
                    concept = super_concept.split(' ')
                    if(len(concept) > max_length_word):
                        max_length_word = len(concept)
                    super_con.append(concept)
                texts.append(super_con)
                count += 1

        self.texts = texts
        self.labels = labels
        self.labels_list = class_ids
        self.class_id2ind = class_id2ind
        self.index_dict = self.model_gensim.wv.index2word[:max_vocab]
        self.vocab_dict = {}
        for index, value in enumerate(self.index_dict[:max_vocab]):
            self.vocab_dict[value] = index

        self.ImportanceFeatureMats = pickle.load(open(ImportanceFeatureMatsFile, 'rb'))
        self.max_length_sentences = max_length_sentences
        self.max_length_word = max_length_word
        self.num_classes = len(self.class_id2ind)
        print('max_length_sentences', max_length_sentences)
        print('max_length_word', max_length_word)
        print('num_classes: ', self.num_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]

        ImportanceFeatureMat = self.ImportanceFeatureMats[index]
        try:
            ImportanceFeatureMat_padded = np.zeros((self.max_length_sentences, ImportanceFeatureMat.shape[1]))
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise e
        ImportanceFeatureMat_padded[:ImportanceFeatureMat.shape[0]] = ImportanceFeatureMat

        document_encode = [
            [self.vocab_dict[word] if word in self.vocab_dict else -1 for word in sentences] for sentences
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

        return document_encode.astype(np.int64), ImportanceFeatureMat_padded, label, text


if __name__ == '__main__':
    test = MyDataset(data_path="/disk/home/klee/data/cs_merged_tokenized_superspan_HANs.txt", label_path='/disk/home/klee/data/cs_merged_label', dict_path="/disk/home/klee/data/cs_merged_tokenized_dictionary.bin")
    print(test[0])
    print(test.__getitem__(index=1)[0].shape)
