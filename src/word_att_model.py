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
    def __init__(self, feature_path,dict_path):
        super(WordAttNet, self).__init__()

        data = open(dict_path,'rb')
        index_dict = pickle.load(data)
        vocab_dict = {}
        for index,value in enumerate(index_dict):
            vocab_dict[value] = index

        data = open(feature_path,'rb')
        mapping = pickle.load(data)


        dict_len = len(index_dict)
        word_feature_size = len(mapping[index_dict[0]])


        feature = np.zeros((dict_len, word_feature_size))
        for key, value in mapping.items():
            index = vocab_dict[key]
            feature[index] = value

        dict_len += 1

        unknown_word = np.zeros((1, word_feature_size))
        feature = torch.from_numpy(np.concatenate([unknown_word, feature], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(word_feature_size,word_feature_size))
        self.word_bias = nn.Parameter(torch.Tensor(1,word_feature_size))
        self.context_weight = nn.Parameter(torch.Tensor(word_feature_size, 1))
        self.dict_len = dict_len
        self.embed_size = word_feature_size
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=self.embed_size).from_pretrained(feature)
        self.lookup.weight.requires_grad = False
        self._create_weights(mean=0.0, std=0.05)
        import pdb;
        pdb.set_trace()
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
