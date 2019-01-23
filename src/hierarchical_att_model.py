"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import pickle
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
from src.utils import matrix_mul, element_wise_mul
import torch.nn.functional as F


def create_embedding(mat):
    mat = torch.from_numpy(mat)
    embedding = nn.Embedding(num_embeddings=mat.shape[0], embedding_dim=mat.shape[1]).from_pretrained(mat)
    embedding.weight.requires_grad = False
    return embedding


class HierAttNet(nn.Module):
    def __init__(self, sent_feature_size, feature_path, dict,
                 max_sent_length, max_word_length,
                 model_save_path, Vv_embedding_path, path_semanticsFile, max_vocab, use_cuda, class_idsFile):
        super(HierAttNet, self).__init__()

        self.sent_feature_size = sent_feature_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(feature_path, dict, max_vocab, use_cuda)
        self.sent_att_net = SentAttNet(sent_feature_size)
        # Nv_len = 6
        # node_num = 34
        # node_score = torch.rand(size=(node_num, Nv_len)).cuda()
        # self.node_score = F.normalize(node_score, p=1, dim=1)
        # self.node_index = torch.tensor([1121, 732, 85, 258, 884, 1935], dtype=torch.int32).cuda()
        # self.similarity_mat = torch.DoubleTensor(self.word_att_net.dict_len, Nv_len).uniform_(-1, 1).cuda()
        # self.similarity_mat[0, :] = 0
        # for id, index in enumerate(self.node_index):
        #     self.similarity_mat[index] = 1

        self.model_gensim = Word2Vec.load(model_save_path)
        self.model_gensim.most_similar(self.model_gensim.wv.index2word[0])
        feature = self.model_gensim.wv.vectors_norm[:max_vocab]
        unknown_word = np.zeros((1, self.model_gensim.vector_size))
        feature = np.concatenate([unknown_word, feature], axis=0).astype(np.float)
        self.embedding = create_embedding(feature)
        # feature = torch.from_numpy(feature)
        # self.embedding = nn.Embedding(num_embeddings=feature.shape[0], embedding_dim=feature.shape[1]).from_pretrained(feature)
        # self.embedding.weight.requires_grad = False

        try:
            Vv_embedding = pickle.load(open(Vv_embedding_path, 'rb'))
            self.Vv_embeddingT = torch.from_numpy(Vv_embedding.T.astype(np.float64))
            if use_cuda:
                self.Vv_embeddingT = self.Vv_embeddingT.cuda()
            self.Vv_embeddingT.requires_grad = False
        except Exception as e:
            import ipdb; ipdb.set_trace()
            raise e
        # torch.DoubleTensor(self.word_att_net.dict_len, Nv_len).uniform_(-1, 1)
        # create_embedding(Vv_embedding)
        # Vv_embedding = torch.from_numpy(Vv_embedding)
        # self.Vv_embedding = nn.Embedding(num_embeddings=Vv_embedding.shape[0], embedding_dim=Vv_embedding.shape[1]).from_pretrained(Vv_embedding)
        # self.Vv_embedding.weight.requires_grad = False

        path_semantics = pickle.load(open(path_semanticsFile, 'rb'))
        self.path_semantics = torch.from_numpy(path_semantics.astype(np.float64))
        if use_cuda:
            self.path_semantics = self.path_semantics.cuda()
        self.path_semantics.requires_grad = False
        # create_embedding(path_semantics)
        # path_semantics = torch.from_numpy(path_semantics)
        # self.path_semantics = nn.Embedding(num_embeddings=path_semantics.shape[0], embedding_dim=path_semantics.shape[1]).from_pretrained(path_semantics)
        # self.path_semantics.weight.requires_grad = False

    def forward(self, input, ImportanceFeatureMat, text):

        batch_size = input.size(0)
        word_attn_score = torch.zeros(self.max_sent_length, batch_size, self.max_word_length)
        # if torch.cuda.is_available():
        #     word_attn_score = word_attn_score.cuda()
        output_list = []

        # iterate over: [sent ind, batch, word ind]
        for idx, i in enumerate(input.permute(1, 0, 2)):
            output, word_attn_score[idx] = self.word_att_net(i)
            output_list.append(output)
        # [batch, sent ind, word ind]
        word_attn_score = word_attn_score.permute(1, 0, 2)
        # [sent ind, batch, word ind]
        output = torch.cat(output_list, 0)

        sent_attn_score = self.sent_att_net(ImportanceFeatureMat)
        # broadcast second word dimension
        attn_score = word_attn_score.permute(2, 0, 1) * sent_attn_score
        attn_score = attn_score.permute(1, 2, 0)
        # flatten
        attn_score = attn_score.contiguous().view(batch_size, -1)
        doc_index = input.view(batch_size, -1)

        final_score = self.compute_score(doc_index, attn_score)
        # import pdb;
        # pdb.set_trace()
        return final_score, attn_score

    def compute_score(self, doc_index, attn_score):
        import ipdb; ipdb.set_trace()
        # , dict_len, phi_vs, embedding, Vv_embedding
        batch_size = doc_index.size(0)
        Nd_len = doc_index.size(1)
        # similarity_mats = matrix_mul(self.embedding(doc_index), self.Vv_embeddingT)

        node_num = self.path_semantics.size(0)
        final_score = torch.zeros(batch_size, node_num)
        # .cuda()
        for i in range(batch_size):
            # similarity_mats[i]
            similarity_mat = self.embedding(doc_index[i]).mm(self.Vv_embeddingT)
            for j in range(node_num):
                try:
                    # outprod_score = torch.ger(attn_score[i], self.path_semantics[j]).cuda()
                    # final_score[i, j] = torch.sum(outprod_score * similarity_weighted_bynode)
                    similarity_weighted_bynode = similarity_mat * self.path_semantics[j]
                    similarity_weighted_bynode = similarity_weighted_bynode.sum(1)
                    final_score[i, j] = torch.sum(attn_score[i] * similarity_weighted_bynode)
                except Exception as e:
                    pass
        return final_score
