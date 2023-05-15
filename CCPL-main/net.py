import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from function import calc_mean_std, nor_mean_std, nor_mean, calc_cov
import random
from torchvision.models import vgg19

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),#对称填充，更好地保留图像的边缘信息
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), # decoder_pho starts layer
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)]) 

class SCT(nn.Module):
    def __init__(self, training_mode='art'):
        super(SCT, self).__init__()
        if training_mode == 'art':
            self.cnet = nn.Sequential(nn.Conv2d(512,256,1,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,128,1,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,32,1,1,0))
            self.snet = nn.Sequential(nn.Conv2d(512,256,3,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256,128,3,1,0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128,32,1,1,0))
            self.uncompress = nn.Conv2d(32,512,1,1,0)
        else: # pho
            self.cnet = nn.Sequential(nn.Conv2d(256,128,1,1,0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128,32,1,1,0))
            self.snet = nn.Sequential(nn.Conv2d(256,128,3,1,0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128,32,1,1,0))
            self.uncompress = nn.Conv2d(32,256,1,1,0)                        
    def forward(self, content, style):
        cF_nor = nor_mean_std(content)
        sF_nor, smean = nor_mean(style)
        cF = self.cnet(cF_nor)
        sF = self.snet(sF_nor)
        b, c, w, h = cF.size()
        s_cov = calc_cov(sF)
        gF = torch.bmm(s_cov, cF.flatten(2, 3)).view(b,c,w,h)
        gF = self.uncompress(gF)
        gF = gF + smean.expand(cF_nor.size())
        return gF                                             

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class CCPL(nn.Module):
    def __init__(self, mlp):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d =self.mlp[3*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self, feats_q, feats_k, num_s, start_layer, end_layer, tau=0.07):
        loss_ccp = 0.0
        for i in range(start_layer, end_layer):
            f_q, sample_ids = self.NeighborSample(feats_q[i], i, num_s, [])
            f_k, _ = self.NeighborSample(feats_k[i], i, num_s, sample_ids)   
            loss_ccp += self.PatchNCELoss(f_q, f_k, tau)
        return loss_ccp    

class VGG19(nn.Module):
    def __init__(self, content_layers, style_layers,vgg19_model):
        super(VGG19, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.vgg19 = vgg19_model  # 取前 36 层作为特征提取网络

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
    def __init__(self, content_layers, style_layers, content_weight, style_weight,vgg19_model):
        super(PerceptualLoss, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.vgg19 = VGG19(content_layers, style_layers,vgg19_model)

    def gram_matrix(self, input):
        batch_size, channel, height, width = input.size()
        features = input.view(batch_size * channel, height * width)
        G = torch.mm(features, features.t())
        return G.div(batch_size * channel * height * width)

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

class Net(nn.Module):
    def __init__(self, encoder, decoder, training_mode='art',content_layer=None,style_layer=None,perceptual_c_w=1,perceptual_s_w=10,vgg19_model=None):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.SCT = SCT(training_mode)
        self.mlp = mlp if training_mode == 'art' else mlp[:9]
        self.CCPL = CCPL(self.mlp)
        self.mse_loss = nn.MSELoss()
        self.end_layer = 4 if training_mode == 'art' else 3
        self.mode = training_mode
        self.perceptual=PerceptualLoss(content_layer,style_layer,perceptual_c_w,perceptual_s_w,vgg19_model)

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(self.end_layer):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(self.end_layer):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def feature_compress(self, feat):
        feat = feat.flatten(2,3)
        feat = self.mlp(feat)
        feat = feat.flatten(1,2)
        feat = Normalize(2)(feat)
        return feat      

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, tau, num_s, num_layer):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        gF = self.SCT(content_feats[-1], style_feats[-1])
        gimage = self.decoder(gF)
        g_t_feats = self.encode_with_intermediate(gimage)

        end_layer = self.end_layer
        loss_perception=self.perceptual.forward(content,style,gimage)
        loss_c = self.calc_content_loss(g_t_feats[-1], content_feats[-1]) 
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0]) 
        for i in range(1, end_layer):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i]) 

        start_layer = end_layer - num_layer
        loss_ccp = self.CCPL(g_t_feats, content_feats, num_s, start_layer, end_layer)

        return loss_c, loss_s, loss_ccp,loss_perception
