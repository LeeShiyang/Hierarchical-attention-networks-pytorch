"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet
import torch.nn.functional as F

def compute_score(doc_index,attn_score,dict_len,node_score,node_index,similarity_mat):

    batch_size = doc_index.size(0)
    Nd_len = doc_index.size(1)

    node_num = node_score.size(0)
    Nv_len = node_score.size(1)
    import pdb;
    pdb.set_trace()
    final_score = torch.zeros(batch_size,node_num).cuda()
    for i in range(batch_size):
        for j in range(node_num):
            outprod_score = torch.ger(attn_score[i],node_score[j]).cuda()
            sub_simi_mat = similarity_mat[doc_index[i]]
            final_score[i,j] = torch.sum(outprod_score*sub_simi_mat)


    return final_score

class HierAttNet(nn.Module):
    def __init__(self, word_feature_size, sent_feature_size, feature_path,dict
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()

        self.word_feature_size = word_feature_size
        self.sent_feature_size = sent_feature_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(feature_path, dict,word_feature_size)
        self.sent_att_net = SentAttNet(sent_feature_size, word_feature_size)
        Nv_len = 6
        node_num = 4
        node_score = torch.rand(size = (node_num,Nv_len)).cuda()
        self.node_score = F.normalize(node_score, p=1, dim=1)
        self.node_index = torch.tensor([1121,732,85,258,884,1935],dtype = torch.int32).cuda()
        self.similarity_mat =  torch.DoubleTensor(self.word_att_net.dict_len,Nv_len).uniform_(-1, 1).cuda()
        # import pdb;
        # pdb.set_trace()
        self.similarity_mat[0,:] = 0
        for id,index in enumerate(self.node_index):
            self.similarity_mat[index] = 1




    def forward(self, input):
        batch_size = input.size(0)
        word_attn_score = torch.zeros(self.max_sent_length, batch_size, self.max_word_length)
        if torch.cuda.is_available():
            word_attn_score = word_attn_score.cuda()
        output_list = []
        input = input.permute(1, 0, 2)
        for idx, i in enumerate(input):
            output, word_attn_score[idx] = self.word_att_net(i.permute(1, 0))
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, sent_attn_score = self.sent_att_net(output)
        attn_score = word_attn_score.permute(2,1,0)*sent_attn_score
        doc_index = input.permute(1,0,2).view(batch_size,-1)
        attn_score = attn_score.permute(1,2,0).contiguous().view(batch_size,-1)

        final_score = compute_score(doc_index,attn_score,self.word_att_net.dict_len,self.node_score,self.node_index,self.similarity_mat)

        return final_score,attn_score
