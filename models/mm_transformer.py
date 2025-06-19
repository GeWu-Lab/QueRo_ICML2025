import torch
import torch.nn as nn
from .mbt_backbone import mm_transformer_encoder
from .fusion_modules import *
from .dynamic import *


class MMTransformerBase(nn.Module):
    def __init__(self, args, n_classes):
        super(MMTransformerBase, self).__init__()
        
        edim = args.edim

        self.encoder = mm_transformer_encoder(args)
        self.fusion = args.fusion_method

        if args.fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=edim * 2, output_dim=n_classes)
        elif args.fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=edim, output_dim=n_classes)

        self.a_head = nn.Linear(edim, n_classes)
        self.v_head = nn.Linear(edim, n_classes)
    
    def forward(self, a, v, affine=False):
        token_a, token_v = self.encoder(a, v)

        if self.fusion == 'uni_A':
            fa = token_a[:, 0, :]
            out_a = self.a_head(fa)

            output = {
                'token_a': token_a,
                'token_v': token_a,
                'out': out_a,
                'out_a': out_a,
                'out_v': out_a
            }
        elif self.fusion == 'uni_V':
            fv = token_v[:, 0, :]
            out_v = self.v_head(fv)

            output = {
                'token_a': token_a,
                'token_v': token_a,
                'out': out_v,
                'out_a': out_v,
                'out_v': out_v
            }
        else:
            fa = token_a[:, 0, :]
            fv = token_v[:, 0, :]
            out_a = self.a_head(fa.detach().clone())
            out_v = self.v_head(fv.detach().clone())

            _, _, out = self.fusion_module(fa, fv)

            output = {
                'token_a': token_a,
                'token_v': token_v,
                'fa': fa,
                'fv': fv,
                'out': out,
                'out_a': out_a,
                'out_v': out_v,
            }
        return output
    

class MMTransformerAttn(nn.Module):
    def __init__(self, args, n_classes):
        super(MMTransformerAttn, self).__init__()
        edim = args.edim
        
        self.encoder = mm_transformer_encoder(args)

        self.fusion_module = AttnFusion(output_dim=n_classes, embed_dim=edim)

        self.a_head = nn.Linear(edim, n_classes)
        self.v_head = nn.Linear(edim, n_classes)

    def forward(self, a, v):
        token_a, token_v = self.encoder(a, v)
        fa = token_a[:, 0, :]
        fv = token_v[:, 0, :]
        out_a = self.a_head(fa.detach().clone())
        out_v = self.v_head(fv.detach().clone())

        _, _, out = self.fusion_module(token_a, token_v)

        output = {
            'token_a': token_a,
            'token_v': token_v,
            'fa': fa,
            'fv': fv,
            'out': out,
            'out_a': out_a,
            'out_v': out_v,
        }
        return output
    
