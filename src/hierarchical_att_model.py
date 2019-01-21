"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
from src.sent_att_model import SentAttNet
from src.word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(self, word_feature_size, sent_feature_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_feature_size = word_feature_size
        self.sent_feature_size = sent_feature_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_feature_size)
        self.sent_att_net = SentAttNet(sent_feature_size, word_feature_size, num_classes)
        self.word_attn_score = torch.zeros(max_sent_length, batch_size, self.max_word_length)
        if torch.cuda.is_available():
            self.word_attn_score = self.word_attn_score.cuda()

    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for idx, i in enumerate(input):
            output, self.word_attn_score[idx] = self.word_att_net(i.permute(1, 0))
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_attn_score = self.sent_att_net(output)
        attn_score = self.word_attn_score.permute(2,1,0)*self.sent_attn_score

        return output,attn_score
