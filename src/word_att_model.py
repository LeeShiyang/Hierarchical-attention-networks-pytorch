"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
# from utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv
import pickle
from sklearn.preprocessing import normalize

torch.set_default_dtype(torch.float64)

#

class WordAttNet(nn.Module):
    def __init__(self, feature_path, dict_path, max_vocab, use_cuda, dataset):
        super(WordAttNet, self).__init__()

        # data = open(dict_path, 'rb')
        # index_dict = pickle.load(data)[:max_vocab]
        # vocab_dict = {}
        # for index, value in enumerate(index_dict):
        #     vocab_dict[value] = index

        mapping = pickle.load(open(feature_path, 'rb'))

        dict_len = len(dataset.index_dict)
        word_feature_size = len(mapping[dataset.index_dict[2002]])

        feature = np.zeros((dict_len, word_feature_size))
        for key, value in mapping.items():
            if key in dataset.vocab_dict:
                feature[dataset.vocab_dict[key]] = value

        feature = normalize(feature, axis=0, norm='max')
        unknown_word = np.zeros((1, word_feature_size))
        feature = torch.from_numpy(np.concatenate([unknown_word, feature], axis=0).astype(np.float))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=word_feature_size).from_pretrained(feature)
        self.lookup.weight.requires_grad = False

        dict_len += 1

        self.word_weight = nn.Parameter(torch.Tensor(word_feature_size, word_feature_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, word_feature_size))
        self.context_weight = nn.Parameter(torch.Tensor(word_feature_size, 1))
        self.context_bias = nn.Parameter(torch.Tensor(1))
        self.dict_len = dict_len
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.005):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(1, std)

        self.word_bias.data.normal_(mean, std)
        self.context_bias.data.normal_(mean, std)

    def forward(self, input):
        # [word ind, batch]
        input = input.permute(1, 0)
        # [batch, word ind, emb ind]
        f_output = self.lookup(input)
        # output = matrix_mul(f_output, self.word_weight, self.word_bias)
        # [batch, word ind]
        output = matrix_mul(f_output, self.context_weight, self.context_bias).permute(1, 0)
        # [batch, word ind]
        attn_score = F.softmax(output, dim=1)

        # [batch, word ind]
        output = element_wise_mul(f_output, attn_score.permute(1, 0))

        return output, attn_score


if __name__ == "__main__":
    abc = WordAttNet("/disk/home/klee/data/cs_merged_tokenized_phrases2feature_vector.bin", '/disk/home/klee/data/cs_merged_tokenized_dictionary.bin')
