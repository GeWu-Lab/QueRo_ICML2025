import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from functools import partial
# from .transformer import *


class AttnFusion(nn.Module):
    def __init__(self, embed_dim=512, output_dim=100, num_heads=1):
        super(AttnFusion, self).__init__()
        print('Using Fusion Head: AttnFusion')
        self.embed_dim = embed_dim
        # self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)
        self.FC = nn.Linear(embed_dim, output_dim)

        # self.register_buffer('affine', None)
        self.register_parameter('affine', nn.Parameter(torch.eye(self.embed_dim), requires_grad=False))
        self.r_time = 0
        # self.register_buffer('affine', nn.Parameter(torch.zeros(self.embed_dim, self.embed_dim)))

        self._cls_token()

    def _cls_token(self): # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)

    def get_Q_cls(self):
        with torch.no_grad():
            x = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            y = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            cls_tokens = self.cls_token.expand(32, -1, -1) # cls token

            seq = torch.cat([cls_tokens, x ,y], dim=1)
            _, _, infos = self.attn(seq, seq, seq)
            qcls = infos['q'][:, 0, :]
            return qcls[0, :]

    def set_affine(self, affine, target_cls):
        with torch.no_grad():
            self.affine.data = torch.matmul(self.affine.data, affine.detach())
            self.r_time += 1
        print('Now check affine matrix')
        with torch.no_grad():
            x = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            y = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            cls_tokens = self.cls_token.expand(32, -1, -1) # cls token

            seq = torch.cat([cls_tokens, x ,y], dim=1)

            out_seq, attn_matrix, infos = self.attn(seq, seq, seq, affine=self.affine)

            affined_qcls = infos['q'][:, 0, :]
            print("THE ERROR (BIG means wrong inside dynamic.py): ")
            print((affined_qcls[0] - target_cls.to(affined_qcls.device)).norm(p=2))

    def forward(self, x, y):
        bs = x.shape[0]
        assert x.shape[0] == y.shape[0]
        
        cls_tokens = self.cls_token.expand(bs, -1, -1)

        seq = torch.cat([cls_tokens, x ,y], dim=1)
        # seq = self.norm(seq)
        
        if self.r_time > 0:
            print('Using Affine')
            out_seq, attn_matrix, infos = self.attn(seq, seq, seq, affine=self.affine)
        else:
            out_seq, attn_matrix, infos = self.attn(seq, seq, seq)

        cls_embed = out_seq[:,0]
        out = self.FC(cls_embed)

        return x, y, [out, attn_matrix, infos]
        # return x, y, [out, attn_matrix]
    

class TransformerFusion(nn.Module):
    def __init__(self, embed_dim=512, output_dim=100, num_heads=1, layer=2,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0.2, attn_drop_rate=0.2):
        super(TransformerFusion, self).__init__()
        print(f'NOW IMPLEMENTS A MM_Transformer: with {layer} layers')
        self.embed_dim = embed_dim
        self.train_layer = 1

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.blocks = nn.ModuleList([])
        for i in range(layer):
            if i == 0:
                # norm in mbt backbone
                self.blocks.append(
                    MyBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer, use_norm1=False)
                )
            else:
                self.blocks.append(
                    MyBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer)
                )
        
        self.FC = nn.Linear(embed_dim, output_dim)
        self._cls_token()
        self.affine = False
    
    def _cls_token(self): # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
    
    def set_affine(self, affine, target_cls):
        self.affine = True
        self.blocks[self.train_layer - 1].set_affine(affine, target_cls)

    def next_stage(self):
        print("TransformerFusion NEXT STAGE")
        self.train_layer = self.train_layer + 1
        self.affine = False

    def forward(self, x, y):
        bs = x.shape[0]
        assert x.shape[0] == y.shape[0]
        cls_tokens = self.cls_token.expand(bs, -1, -1) # cls token

        # print(cls_tokens.shape, x.shape, y.shape)
        seq = torch.cat([cls_tokens, x ,y], dim=1)

        layer = 0
        for block in self.blocks:
            if layer >= self.train_layer:
                break
            print("train layer", layer + 1)
            seq, attn, info = block(seq)
            layer += 1
        
        cls_embed = seq[:, 0, :]
        out = self.FC(cls_embed)
        
        return x, y, [out, attn, info]     


from .mbt_backbone import Mlp
class MyBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, drop=0., attn_drop=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_norm1=True):
        super().__init__()
        self.norm1 = norm_layer(dim) if use_norm1 else nn.Identity()
        self.attn = MultiHeadAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.register_buffer('affine', None)
    
    def set_affine(self, affine, target_cls):
        if self.affine is None:
            print("Affine Matrix is set")
            self.affine = affine.detach()
        else:
            raise ValueError("set affine matrix twice!")

    def forward(self, x):
        norm_seq = self.norm1(x)
        # print(self.norm1)
        # print(torch.mean(norm_seq, dim = 2))
        out, attn_matrix, info = self.attn(norm_seq, norm_seq, norm_seq)
        x = x + out # residual connect

        norm_seq = self.norm2(x)
        x = x + self.mlp(norm_seq)
        # norm_seq = self.norm1(x)
        # out, attn_matrix, info = self.attn(norm_seq, norm_seq, norm_seq)
        # seq = x + out
        # norm_seq = self.norm2(seq)
        # x = x + self.mlp(x)

        return x, attn_matrix, info


'''
From https://github.com/yaohungt/Multimodal-Transformer/blob/master/modules/multihead_attention.py
'''

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # [B, heads, L, edim]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # For monitor
        # print("qk^T", attn[0, 0, 0, :].shape, attn[0, 0, 0, :])

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # DONE 
        output = torch.matmul(attn, v)

        return output, attn

'''
    https://github.com/jadore801120/attention-is-all-you-need-pytorch
'''
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, embed_dim, num_heads, bias=True, d_k=None, d_v=None, dropout=0.0):
        super().__init__()

        d_model = embed_dim
        n_head = num_heads

        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, affine=None):
        # print('MHA forward')
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        qn = self.w_qs(q)
        kn = self.w_ks(k)
        vn = self.w_vs(v)

        if affine is not None:
            qcls = qn[:, 0, :]
            qcls_affined = qcls @ affine.T.to(qcls.device)
            qn[:, 0 ,:] = qcls_affined

        q = qn.view(sz_b, len_q, n_head, d_k)
        k = kn.view(sz_b, len_k, n_head, d_k)
        v = vn.view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)
        # print(attn.shape)
        # attn = torch.mean(attn, dim=1)
        if self.n_head == 1:
            attn = attn.squeeze(2)

        # print("attn inside: ", attn[0, 0, :])
        # print("", torch.sum(attn[0, 0, :177]))
        # print("", torch.sum(attn[0, 0, 177:]))

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.dropout(self.fc(out))
        # out += residual

        # out = self.layer_norm(out)
        # print(torch.mean(attn, dim=0)[0])
        infos = {
            'q': qn,
            'k': kn,
            'v': vn
        }
        # print('info:', infos)

        return out, attn, infos


# class TransformerFusion(nn.Module):
#     def __init__(self, embed_dim=512, output_dim=100, num_heads=1, layer=2):
#         super(TransformerFusion, self).__init__()
#         print(f'NOW IMPLEMENTS A MM_Transformer: with {layer} layers')
#         # self.norm = nn.LayerNorm(embed_dim)
#         self.embed_dim = embed_dim

#         self.layers = nn.ModuleList([])
#         for _ in range(layer):
#             self.layers.append(nn.ModuleList([
#                 MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads),
#                 nn.LayerNorm(embed_dim),
#                 nn.Linear(embed_dim, embed_dim),
#                 nn.LayerNorm(embed_dim),
#             ]))
        
#         self.FC = nn.Linear(embed_dim, output_dim)
    
#         self.train_layer = 1

#         self._cls_token()

#     def _cls_token(self): # init
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         trunc_normal_(self.cls_token, std=0.02)

#     def forward(self, x, y, mask=None):
#         bs = x.shape[0]
#         assert x.shape[0] == y.shape[0]
#         cls_tokens = self.cls_token.expand(bs, -1, -1) # cls token

#         # print(cls_tokens.shape, x.shape, y.shape)
#         seq = torch.cat([cls_tokens, x ,y], dim=1)

#         attns = []
#         infos = []
#         tokens = []
#         layer = 1
#         for mha, norm1, fc, norm2 in self.layers:
#             if layer > self.train_layer:
#                 break
#             print(f'train layer {layer}')
#             tokens.append(seq)
#             out_seq, attn, info = mha(seq, seq, seq)
#             out_seq = out_seq + seq # residual
#             attns.append(attn.squeeze())
#             infos.append(info)

#             out_seq = norm1(out_seq)
#             out_seq = fc(out_seq) + out_seq
#             seq = norm2(out_seq)
#             layer += 1

#         cls_embed = out_seq[:,0]
#         out = self.FC(cls_embed)

#         return x, y, [out, attns, infos, tokens]

# class DynamicGateFusion(nn.Module):
#     def __init__(self, input_dim = 512, output_dim = 512):
#         super(DynamicGateFusion, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
 
#         self.weight = nn.Sequential(nn.Linear(input_dim * 2, input_dim // 16), nn.ReLU(), nn.Linear(input_dim // 16, input_dim * 2))
#         self.FC = nn.Linear(self.input_dim * 2, output_dim)
#         self.tanh = nn.Tanh()
       
#     def forward(self, x, y):
#         z = torch.cat((x, y), dim=1)
#         weight = self.weight(z)
#         alpha, beta = torch.chunk(weight, chunks=2, dim=1) # ==> [16, input_dim]

#         alpha = 1 + self.tanh(alpha)
#         beta  = 1 + self.tanh(beta)

#         t_x = torch.mul(alpha, x)
#         t_y = torch.mul(beta, y)

#         with torch.no_grad():
#             similar = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
#             sim_x = torch.mean(similar(alpha, x))
#             sim_y = torch.mean(similar(beta, y))

#         output = self.FC(torch.cat((t_x, t_y), dim=1))
#         helper = [output, sim_x, sim_y, alpha, beta]

#         return x, y, helper
