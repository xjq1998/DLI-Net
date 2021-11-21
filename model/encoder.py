import torch
from torch import nn
from torchvision import models
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, feature_type="mid"):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(True)
        self.feature_type=feature_type
        if feature_type == "high":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-2])
            )
            self.fea_wh = 8
            self.fea_channel = 2048
        elif feature_type == "mid":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-3])
            )
            self.fea_wh = 16
            self.fea_channel = 1024
        elif feature_type == "global":
            self.backbone = nn.Sequential(
                OrderedDict(list(resnet50.named_children())[:-2])
            )
            self.fea_wh = 1
            self.fea_channel = 2048

    def forward(self, x):
        fea = self.backbone(x)
        if self.feature_type == "global":
            fea = torch.nn.functional.adaptive_avg_pool2d(fea, 1)
            fea = torch.flatten(fea, 1)
            out = torch.nn.functional.normalize(fea, dim=1)
        elif self.feature_type == "mid":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
        elif self.feature_type == "high":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
        return out


if __name__ == "__main__":
    encoder = Encoder("mid")
    x = torch.rand(4, 3, 256, 256)
    out = encoder(x)
    print(out.shape)
