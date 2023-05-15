import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class VGG19(nn.Module):
    def __init__(self, content_layers, style_layers):
        super(VGG19, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.vgg19 = vgg19(pretrained=True).features[:36]  # 取前 36 层作为特征提取网络

    def forward(self, contentpic,stylepic,genpic):
        content_outputs = []
        style_outputs = []
        gen_content_out=[]
        gen_style_out=[]
        for i, layer in enumerate(self.vgg19):
            contentpic = layer(contentpic)
            stylepic = layer(stylepic)
            genpic = layer(genpic)
            if i in self.content_layers:
                content_outputs.append(contentpic)
                gen_content_out.append(genpic)
            if i in self.style_layers:
                style_outputs.append(stylepic)
                gen_style_out.append(genpic)
        return content_outputs, style_outputs,gen_content_out,gen_style_out

class PerceptualLoss(nn.Module):
    def __init__(self, content_layers, style_layers, content_weight, style_weight):
        super(PerceptualLoss, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.vgg19 = VGG19(content_layers, style_layers)

    def forward(self, content_img, style_img, gen_img):
        content_outputs, style_outputs,gen_content_out,gen_style_out = self.vgg19.forward(content_img,style_img,gen_img)
        content_loss = 0
        style_loss = 0
        for content, gen in zip(content_outputs, gen_content_out):
            content_loss += F.mse_loss(gen, content) * self.content_weight
        for style, gen in zip(style_outputs, gen_style_out):
            gram_style = self.gram_matrix(style)
            gram_gen = self.gram_matrix(gen)
            style_loss += F.mse_loss(gram_gen, gram_style.expand_as(gram_gen)) * self.style_weight
        return content_loss + style_loss

    def gram_matrix(self, input):
        batch_size, channel, height, width = input.size()
        features = input.view(batch_size * channel, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channel * height * width)