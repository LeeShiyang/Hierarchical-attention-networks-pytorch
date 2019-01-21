"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import matrix_mul, element_wise_mul
torch.set_default_dtype(torch.float64)

class SentAttNet(nn.Module):
    def __init__(self, sent_feature_size=50, word_feature_size=50, num_classes=14):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(sent_feature_size,sent_feature_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1,sent_feature_size))
        self.context_weight = nn.Parameter(torch.Tensor(sent_feature_size, 1))
        self.fc = nn.Linear(sent_feature_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)

    def forward(self, input):


        output = matrix_mul(input, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        attn_score = F.softmax(output,dim=1)
        output = element_wise_mul(input, attn_score.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, attn_score


if __name__ == "__main__":
    abc = SentAttNet()
