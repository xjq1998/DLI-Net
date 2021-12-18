import torch
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, feature_type="mid", num_class=125):
        super(Encoder, self).__init__()
        resnet50 = models.resnet50(True)
        self.backbone = resnet50
        self.backbone.fc = nn.Linear(2048, num_class)
        self.feature_type = feature_type

        def hook_fn(module, input, output):
            self.mid_feature = output

        if feature_type == "mid":
            layer = "layer3"
        elif feature_type == "high":
            layer = "layer4"
        elif feature_type == "global":
            layer = "avgpool"

        for name, module in self.backbone.named_children():
            if name == layer:
                module.register_forward_hook(hook_fn)

    def forward(self, x):
        cls_out = self.backbone(x)
        fea = self.mid_feature
        if self.feature_type == "global":
            fea = torch.flatten(fea, 1)
            out = torch.nn.functional.normalize(fea, dim=1)
        elif self.feature_type == "mid":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
        elif self.feature_type == "high":
            fea = torch.nn.functional.normalize(fea, dim=1)
            out = fea.flatten(2).permute(0, 2, 1)
        return out, cls_out


if __name__ == "__main__":
    encoder = Encoder("global")
    x = torch.rand(4, 3, 256, 256)
    out = encoder(x)[0]
    print(out.shape)
