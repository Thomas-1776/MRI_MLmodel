"""U-Net (ResNet18 encoder) - same architecture as unet_mri.py, for inference only."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class Up(nn.Module):
    def __init__(self, i, skip, o):
        super().__init__()
        self.c = block(i + skip, o)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.c(torch.cat([x, skip], 1))


class UNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # pretrained=False: no download needed, the trained .pt overwrites all weights anyway
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.u4 = Up(512, 256, 256); self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64);   self.u1 = Up(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):                      # x: float tensor, range 0-255
        x = (x - self.mean) / self.std
        x0 = self.stem(x)
        x1 = self.l1(self.pool(x0)); x2 = self.l2(x1); x3 = self.l3(x2); x4 = self.l4(x3)
        d = self.u4(x4, x3); d = self.u3(d, x2); d = self.u2(d, x1); d = self.u1(d, x0)
        d = F.interpolate(d, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.head(d)
