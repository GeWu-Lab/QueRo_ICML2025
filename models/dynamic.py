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
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2)
        self.FC = nn.Linear(embed_dim, output_dim)

        self.register_parameter('rotate', nn.Parameter(torch.eye(self.embed_dim), requires_grad=False))
        self.r_time = 0

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

    def set_rotate(self, rotate, target_cls):
        with torch.no_grad():
            print("before", self.rotate.data)
            print("before", rotate)
            self.rotate.data = torch.matmul(rotate.detach(), self.rotate.data)
            print("after, ", self.rotate.data)
            self.r_time += 1
        print('Now check rotary matrix') 
        with torch.no_grad():
            x = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            y = torch.randn([32, 50, self.cls_token.shape[-1]]).to(self.cls_token.device)
            cls_tokens = self.cls_token.expand(32, -1, -1) # cls token

            seq = torch.cat([cls_tokens, x ,y], dim=1)

            out_seq, attn_matrix, infos = self.attn(seq, seq, seq, affine=self.rotate)

            affined_qcls = infos['q'][:, 0, :]
            print("THE ERROR (BIG means wrong inside dynamic.py): ")
            print((affined_qcls[0] - target_cls.to(affined_qcls.device)).norm(p=2))

    def forward(self, x, y):
        bs = x.shape[0]
        assert x.shape[0] == y.shape[0]
        
        cls_tokens = self.cls_token.expand(bs, -1, -1)

        seq = torch.cat([cls_tokens, x ,y], dim=1)
        
        if self.r_time > 0:
            print(f'Rotation Time {self.r_time}')
            out_seq, attn_matrix, infos = self.attn(seq, seq, seq, affine=self.rotate)
        else:
            out_seq, attn_matrix, infos = self.attn(seq, seq, seq)

        cls_embed = out_seq[:,0]
        out = self.FC(cls_embed)

        return x, y, [out, attn_matrix, infos]
    

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
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        qn = self.w_qs(q)
        kn = self.w_ks(k)
        vn = self.w_vs(v)

        if affine is not None:
            '''
                y = R @ x
                but here, qcls -> [batch, edim] = stack(x^T)
                out: [B, edim] <--> y^T = x^T @ R^T
            '''
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
        if self.n_head == 1:
            attn = attn.squeeze(2)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        out = self.dropout(self.fc(out))

        infos = {
            'q': qn,
            'k': kn,
            'v': vn
        }

        return out, attn, infos

