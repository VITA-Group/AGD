from torchvision.models.vgg import vgg16
import torch.nn as nn

class VGGFeature(nn.Module):
    def __init__(self):
        super(VGGFeature, self).__init__()
        self.vgg = vgg16(pretrained=True)
        self.loss_network = nn.Sequential(*list(self.vgg.features)[:31]).eval()

        for param in self.loss_network.parameters():
            param.requires_grad = False

    def __call__(self,x):
        x = x.clone()
        x_vgg = self.loss_network(x)
        return x_vgg
