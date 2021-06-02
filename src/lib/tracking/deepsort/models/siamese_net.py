import torch
from torch import nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, reid=False, num_classes=None):
        super(SiameseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            )

    def forward_once(self, x):
        output = self.net(x)
        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1, output2, output3

        return output1, output2, output3


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
        distance_positive = F.cosine_similarity(anchor, positive) # each is batch X 512
        distance_negative = F.cosine_similarity(anchor, negative) # .pow(.5)
        losses = (1 - distance_positive)**2 + (0 - distance_negative)**2 # Margin not used in cosine case.
        return losses.mean() if size_average else losses.sum()


def test():
    torch.manual_seed(1234)
    inp1 = torch.randn(1, 3, 256, 128).cuda()
    inp2 = torch.randn(1, 3, 256, 128).cuda()
    inp3 = torch.randn(1, 3, 256, 128).cuda()
    criterion = TripletLoss()
    net = SiameseNet().cuda()
    out1, out2, out3 = net(inp1, inp2, inp3)
    loss = criterion(out1, out2, out3)
    print(loss)

if __name__ == '__main__':
    # net = SiameseNet()
    test()

