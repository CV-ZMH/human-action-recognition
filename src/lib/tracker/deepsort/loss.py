# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function. ref: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * distance + (1 + (-1 * label)).float() * \
                        F.relu(self.margin - (distance + self.eps).sqrt()).pow(2))

        loss_contrastive = torch.mean(losses)
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a postive sample and a negative sample
    """

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor, positive)
        distance_negative = F.cosine_similarity(anchor, negative) # .pow(.5)
        losses = (1 - distance_positive)**2 + (0 - distance_negative)**2 # Margin not used in cosine case.
        return losses.mean() if size_average else losses.sum()