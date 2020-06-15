from torchvision.models.vgg import vgg19
import torch
import torch.nn as nn

class VGGFeature(nn.Module):
    def __init__(self, before_act=True, feature_layer=34):
        super(VGGFeature, self).__init__()
        self.vgg = vgg19(pretrained=True)
        self.feature_layer = feature_layer

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        if before_act:
            self.features = nn.Sequential(*list(self.vgg.features.children())[:(feature_layer+1)])
        else:
            self.features = nn.Sequential(*list(self.vgg.features.children())[:(feature_layer)])

        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def __call__(self,x):
        x = (x - self.mean) / self.std
        x_vgg = self.features(x)
        return x_vgg
