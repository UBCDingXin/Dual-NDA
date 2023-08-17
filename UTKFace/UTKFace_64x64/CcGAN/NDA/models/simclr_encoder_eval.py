# net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.resnet import resnet50
from torchvision import models

# stage one ,unsupervised learning
class encoder_eval(nn.Module):
    def __init__(self, feature_dim=128):
        super(encoder_eval, self).__init__()

        resnet50 = models.resnet50()

        self.f = []
        for name, module in resnet50.named_children():
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        # return F.normalize(feature, dim=-1)
        return feature.view(-1, feature.size(1))
        # feature = self.g(feature)
        # return F.normalize(feature, dim=-1)


if __name__ == "__main__":
    net = encoder_eval().cuda()
    x = torch.randn(10,3,224,224).cuda()
    feature = net(x)
    print(feature.size())

