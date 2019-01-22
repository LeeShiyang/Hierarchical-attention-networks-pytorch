"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from src.utils import matrix_mul, element_wise_mul
from utils import matrix_mul, element_wise_mul
import pandas as pd
import numpy as np
import csv
import pickle
torch.set_default_dtype(torch.float64)

class WordAttNet(nn.Module):
    def __init__(self, feature_path,dict_path, word_feature_size=4):
        super(WordAttNet, self).__init__()

        data = open(dict_path,'rb')
        dict = pickle.load(data)


        data = open(feature_path,'rb')
        mapping = pickle.load(data)
        dict_len = len(dict)
        feature = np.zeros((dict_len, word_feature_size))

        for key, value in mapping.items():
            index = dict.index(key)
            feature[index] = value


        import pdb;
        pdb.set_trace()
        dict_len += 1

        unknown_word = np.zeros((1, word_feature_size))
        feature = torch.from_numpy(np.concatenate([unknown_word, feature], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(word_feature_size,word_feature_size))
        self.word_bias = nn.Parameter(torch.Tensor(1,word_feature_size))
        self.context_weight = nn.Parameter(torch.Tensor(word_feature_size, 1))
        self.dict_len = dict_len
        self.embed_size = embed_size
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(feature)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.word_bias.data.normal_(mean, std)

    def forward(self, input):

        f_output = self.lookup(input)
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        attn_score = F.softmax(output,dim=1)
        output = element_wise_mul(f_output,attn_score.permute(1,0))

        return output,attn_score


if __name__ == "__main__":
    abc = WordAttNet("/disk/home/klee/data/cs_merged_tokenized_concept_feature.bin",'/disk/home/klee/data/cs_merged_tokenized_dictionary.bin')
