import torch
from torch import nn
from torch.nn import functional as F

class SiameseNet(nn.Module):
    def __init__(self, reid=False, **kwargs):
        super(SiameseNet, self).__init__()
        self.reid = reid
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

    def forward(self, x):
        feat = self.net(x)
        size = [int(s) for s in feat.size()[2:]]
        output = F.avg_pool2d(feat, size)
        if self.reid:
            output.squeeze_(2).squeeze_(2)
            output.div_(output.norm(p=2, dim=1, keepdim=True))
        return output

    def forward_twice(self, input1, input2, input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1, output2, output3

        return output1, output2, output3


def test():
    torch.manual_seed(1234)
    inp1 = torch.randn(1, 3, 256, 128).cuda()
    inp2 = torch.randn(1, 3, 256, 128).cuda()
    inp3 = torch.randn(1, 3, 256, 128).cuda()
    net = SiameseNet().cuda()
    out1, out2, out3 = net(inp1, inp2, inp3)

if __name__ == '__main__':
    inp1 = torch.ones(1, 3, 256, 128).cuda()
    net = SiameseNet().cuda()
    out = net.forward_once(inp1)
    out = out.div(out.norm(p='fro', dim=1, keepdim=True))
    out.shape, out.min(), out.max()
    # test()
