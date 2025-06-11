import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .vit import vit
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion
from .dynamic import *


class AVClassifier(nn.Module):
    def __init__(self, args, n_classes):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        self.fusion = fusion
        self.dset = args.dataset

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        # ADD Dynamic
        elif fusion == 'dynGate':
            self.fusion_module = DynamicGateFusion(output_dim=n_classes)
        elif fusion == 'uni_A':
            self.fusion_module = None
        elif fusion == 'uni_V':
            self.fusion_module = None
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.a_head = nn.Linear(512, n_classes)
        self.v_head = nn.Linear(512, n_classes)


    def forward(self, audio, visual):
        B = audio.shape[0]
        
        if self.fusion == 'uni_A':
            a = self.audio_net(audio)

            a = feature_reshape_single(a, B)
            out_a = self.a_head(a)
            # out_v = self.v_head(v)
            # out = (out_a.detach().clone() + out_v.detach().clone()) / 2

            output = {
                'a': a,
                'v': a,
                'out': out_a,
                'out_a': out_a,
                'out_v': out_a
            }
        elif self.fusion == 'uni_V':
            v = self.visual_net(visual)

            v = feature_reshape_single(v, B)
            out_v = self.v_head(v)
            # out_a = self.a_head(a)
            # out = (out_a.detach().clone() + out_v.detach().clone()) / 2

            output = {
                'a': v,
                'v': v,
                'out': out_v,
                'out_a': out_v,
                'out_v': out_v
            }
        else:
            a = self.audio_net(audio)
            v = self.visual_net(visual)

            a, v = feature_reshape(a, v, B)
            out_a = self.a_head(a.detach().clone())
            out_v = self.v_head(v.detach().clone())
            a, v, out = self.fusion_module(a, v)
            
            output = {
                'a': a,
                'v': v,
                'out': out,
                'out_a': out_a,
                'out_v': out_v
            }
        
        return output


class AVClassifier_DYN(nn.Module):
    def __init__(self, args, n_classes):
        super(AVClassifier_DYN, self).__init__()
        print('now is AVClassifier_DYN')

        fusion = args.fusion_method
        self.fusion = fusion
        self.dset = args.dataset

        embed_size = 512
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')
        self.a_head = nn.Linear(512, n_classes)
        self.v_head = nn.Linear(512, n_classes)
    
        # ADD Dynamic
        if fusion == 'dynGate':
            self.fusion_module = DynamicGateFusion(output_dim=n_classes)
        elif fusion == 'AttnFusion':
            self.fusion_module = AttnFusion(output_dim=n_classes, embed_dim=embed_size, norm=True)
        elif fusion == 'Transformer':
            self.fusion_module = TransformerFusion(output_dim=n_classes, embed_dim=embed_size, layer = args.layer, num_heads=8)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))


    def forward(self, audio, visual, noisy=False):
        B = audio.shape[0]
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        device = a.device
        if noisy:
            v = torch.randn(v.shape).to(device)
            print('replace v')

        fa, fv = feature_reshape(a, v, B)

        if self.fusion in ['AttnFusion', 'Transformer']:
            out_a = self.a_head(fa.detach().clone())
            out_v = self.v_head(fv.detach().clone())

            token_a, token_v = token_reshape(a, v, B)
                
            _, _, out = self.fusion_module(token_a, token_v)

            output = {
                'a': a,
                'v': v,
                'fa': fa,
                'fv': fv,
                'out': out,
                'out_a': out_a,
                'out_v': out_v,
                'token_a': token_a,
                'token_v': token_v,
            }

        return output

    

class AVClassifier_Frozen_Encoder(nn.Module):
    def __init__(self, args, n_classes, detach_A = True, detach_V = True):
        super(AVClassifier_Frozen_Encoder, self).__init__()

        print("AVClassifier_Frozen_Encoder")

        fusion = args.fusion_method
        self.fusion = fusion

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        # ADD Dynamic
        elif fusion == 'dynGate':
            self.fusion_module = DynamicGateFusion(output_dim=n_classes)
        elif fusion == 'AttnFusion':
            self.fusion_module = AttnFusion(output_dim=n_classes)
        elif fusion == 'uni':
            self.fusion_module = None
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
        
        self.fusion = fusion

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.a_head = nn.Linear(512, n_classes)
        self.v_head = nn.Linear(512, n_classes)
    
    def freeze_encoder(self):
        for param in self.audio_net.parameters():
            param.require_grad = False
        for param in self.visual_net.parameters():
            param.require_grad = False

        print('Freeze Encoder')

    def forward(self, audio, visual):
        B = audio.shape[0]
        with torch.no_grad():
            a = self.audio_net(audio).detach()
            v = self.visual_net(visual).detach()

        if self.fusion == 'AttnFusion':
            # print('Running AttnFusion')
            fa, fv = feature_reshape(a, v, B)
            out_a = self.a_head(fa.detach().clone())
            out_v = self.v_head(fv.detach().clone())

            token_a, token_v = token_reshape(a, v, B)

            with torch.no_grad():
                _, _, uni_a = self.fusion_module(token_a, torch.zeros_like(token_v).detach())
                _, _, uni_v = self.fusion_module(torch.zeros_like(token_a).detach(), token_v)
                _, _, attn_probe = self.fusion_module(torch.zeros_like(token_a).detach(), torch.zeros_like(token_a).detach())
            _, _, out = self.fusion_module(token_a, token_v)
            # out = torch.zeros([1]).to(device)
            uni_a = uni_a[0]
            uni_v = uni_v[0]

            output = {
                'a': a,
                'v': v,
                'out': out,
                'out_a': out_a,
                'out_v': out_v,
                'uni_a': uni_a,
                'uni_v': uni_v,
                'token_a': token_a,
                'token_v': token_v,
                'attn_probe': attn_probe[1]
            }
        elif self.fusion == 'uni':
            a, v = feature_reshape(a, v, B)
            out_a = self.a_head(a)
            out_v = self.v_head(v)
            out = (out_a.detach().clone() + out_v.detach().clone()) / 2
        else:
            a, v = feature_reshape(a, v, B)
            a, v, out = self.fusion_module(a, v)
            if self.detach_A:
                out_a = self.a_head(a.detach().clone())
            else:
                out_a = self.a_head(a)

            if self.detach_V:
                out_v = self.v_head(v.detach().clone())
            else:
                out_v = self.v_head(v)
        
            output = {
                'a': a,
                'v': v,
                'out': out,
                'out_a': out_a,
                'out_v': out_v
            }
        
        return output


class Unimodal_Classifier(nn.Module):
    def __init__(self, args, n_classes, modality=None):
        super(Unimodal_Classifier, self).__init__()

        fusion = args.fusion_method
        self.fusion = fusion

        if modality is None:
            raise NotImplementedError('Unknown Which Modality')
        
        self.modality = modality

        self.net = resnet18(modality=modality)
        self.head = nn.Linear(512, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.net(x)
        
        # reshape
        (_, C, H, W) = x.size()
        x = x.view(B, -1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = F.adaptive_avg_pool3d(x, 1)
        x = torch.flatten(x, 1)

        out = self.head(x)
        return out



def feature_reshape_single(x, B):
    _, C, H, W = x.size()
    x = x.view(B, -1, C, H, W)
    x = x.permute(0, 2, 1, 3, 4)

    x = F.adaptive_avg_pool3d(x, 1)

    x = torch.flatten(x, 1)
    return x


def feature_reshape(a, v, B):
    (_, C, H, W) = v.size()
    v = v.view(B, -1, C, H, W)
    v = v.permute(0, 2, 1, 3, 4)

    _, C, H, W = a.size()
    a = a.view(B, -1, C, H, W)
    a = a.permute(0, 2, 1, 3, 4)

    a = F.adaptive_avg_pool3d(a, 1)
    v = F.adaptive_avg_pool3d(v, 1)

    a = torch.flatten(a, 1)
    v = torch.flatten(v, 1)
    return a, v 


def token_reshape(a, v, B):
    (_, C, H, W) = v.size()
    # v = v.view(B, -1, C, H, W)
    # v = v.contiguous()
    v = v.view(B, -1, C)

    _, C, H, W = a.size()
    # a = a.view(B, -1, C, H, W)
    # a = a.contiguous()
    a = a.view(B, -1, C)
    
    return a, v
