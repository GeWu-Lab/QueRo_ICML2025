import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.helpers import named_apply
from timm.models.layers import DropPath, Mlp, to_2tuple
from timm.layers.weight_init import lecun_normal_, trunc_normal_
import numpy as np
# from models.heads import *

def mm_transformer_encoder(args):
    dset = args.dataset
    edim = args.edim
    depth = args.depth
    nframes = args.use_video_frames
    o_nframes = args.use_optical_frames
    multi_depth = args.multi_depth if args.backbone == 'mbt' else 0
    if dset in ['CREMAD', 'KineticSound']:
        if dset == 'CREMAD':
            input_fdim=188  # w
            input_tdim=257  # h
        elif dset == 'KineticSound':
            input_fdim=128  # w
            input_tdim=1024 # h
        
        if args.fusion_method == 'uni_A':
            model = MBTBackbone_Unimodal(modality=args.fusion_method, embed_dim=edim, depth=depth+multi_depth, multi_depth=multi_depth, nframes=nframes, o_nframes=o_nframes, 
                                input_fdim=input_fdim, input_tdim=input_tdim, dset=dset)
        elif args.fusion_method == 'uni_V':
            model = MBTBackbone_Unimodal(modality=args.fusion_method, embed_dim=edim, depth=depth+multi_depth, multi_depth=multi_depth, nframes=nframes, o_nframes=o_nframes, 
                                input_fdim=input_fdim, input_tdim=input_tdim, dset=dset)
        else:
            model = MBTBackbone(embed_dim=edim, depth=depth+multi_depth, multi_depth=multi_depth, nframes=nframes, o_nframes=o_nframes, 
                                input_fdim=input_fdim, input_tdim=input_tdim, dset=dset)
    
    return model


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


class LinearEmbed(nn.Module):
    def __init__(self, input_dim=30, output_dim=30):
        super().__init__()
        self.fc = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = x.squeeze()
        x = self.fc(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mod1_length = 513, mod2_length=393):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(N)
        # 一个linear算qkv
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # 用公式和softmax算这个权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    
class Attention_multi_doublevalue(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mod1_length = 513, mod2_length=393):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_cross = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mod1_length = mod1_length
        self.mod2_length = mod2_length
        self.anchor_sum = 1.0 / self.mod1_length + 1.0 / self.mod2_length
        print(self.mod1_length)
        self.p_list = []

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_cross = self.v_cross(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_a = attn[:,:,:self.mod1_length] @ torch.cat((v[:,:,:self.mod1_length], v_cross[:,:,self.mod1_length:]), dim=-2)
        x_v = attn[:,:,self.mod1_length:] @ torch.cat((v_cross[:,:,:self.mod1_length], v[:,:,self.mod1_length:]), dim=-2)
        x = torch.cat((x_a, x_v), dim=-2).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Attention_multi(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mod1_length = 513, mod2_length=393):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mod1_length = mod1_length
        self.mod2_length = mod2_length
        self.anchor_sum = 1.0 / self.mod1_length + 1.0 / self.mod2_length
        print(self.mod1_length)
        self.p_list = []

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # 用公式和softmax算这个权重
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn = 'vanilla', mod1_length = 513, mod2_length = 393, is_multi=False, drop_x = 0.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_type = 'not loss'
        if attn == 'vanilla':
            if is_multi:
                self.attn = Attention_multi(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
                print('multi attention')
            else:
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
        elif attn == 'value':
            if is_multi:
                self.attn = Attention_multi_doublevalue(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
                print('multi attention')
            else:
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
            
        elif attn == 'mask':
            self.attn = Attention_mask(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
            print('masked attention')
        elif attn == 'loss':
            self.attn = Attention_loss(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length, mod2_length=mod2_length)
            print('loss attention') 
            self.attn_type = 'loss'
        else:
            self.attn = Attention_softmax(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mod1_length=mod1_length)
            print('softmax attention')
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.drop_x_1 = DropPath(drop_x) if (drop_x > 0. and is_multi) else nn.Identity()
        self.drop_x_2 = DropPath(drop_x) if (drop_x > 0. and is_multi) else nn.Identity()

    def forward(self, x):
        if self.attn_type == 'loss':
            attn, loss = self.attn(self.norm1(x))
            x = self.drop_x_1(x) + self.drop_path(attn)
            x = self.drop_x_2(x) + self.drop_path(self.mlp(self.norm2(x)))
            return x, loss
        else:
            x = self.drop_x_1(x) + self.drop_path(self.attn(self.norm1(x)))
            x = self.drop_x_2(x) + self.drop_path(self.mlp(self.norm2(x)))
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class AudioPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VideoPatchEmbed(nn.Module):
    """ 
        2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, frames = 3, norm_layer=None, flatten=True):
        super().__init__()

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size, patch_size), stride=(1, patch_size, patch_size))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = (img_size // patch_size) ** 2 * frames
        # print(self.num_patches)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MBTBackbone(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=8, multi_depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, distilled=False, num_bottle_token=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', nframes=2, o_nframes=3, fstride=16, tstride=16, input_fdim=128, input_tdim=1024, dset='CREMAD'):

        super().__init__()

        bot_num_heads = 1
        print(input_fdim, input_tdim)
        self.dset = dset
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.multi_depth = multi_depth
        ''' move in dynamic'''
        self.num_bottle_token = num_bottle_token

        # Audio (Optical / Text)
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if dset == 'UCF':
            self.patch_embed_audio = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim, frames=o_nframes)
            num_patches_audio = self.patch_embed_audio.num_patches
        else:
            self.patch_embed_audio = AudioPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches_audio  = f_dim * t_dim
            self.patch_embed_audio.num_patches = num_patches_audio
            new_proj = torch.nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=(fstride, tstride))
            self.patch_embed_audio.proj = new_proj
        self.pos_embed_audio = nn.Parameter(torch.zeros(1, num_patches_audio + 1, embed_dim))
        
        # Video (Text)
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed_video = VideoPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, frames=nframes)
        num_patches_video = self.patch_embed_video.num_patches
        print(num_patches_video)
        self.pos_embed_video = nn.Parameter(torch.zeros(1, num_patches_video + self.num_tokens, embed_dim))

        # trunc_normal_(self.pos_embed, std=.02)

        self.bot_token = nn.Parameter(torch.zeros(1, num_bottle_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.audio_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth)])

        self.video_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth)])

        print(f'depth of attention bottleneck is {multi_depth}')
        print(f'MBT now is (uni){depth-multi_depth} + (multi){multi_depth}')

        self.bot_audio_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=bot_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth, depth)])

        self.bot_video_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=bot_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth, depth)])

        self.norm_audio = norm_layer(embed_dim)
        self.norm_video = norm_layer(embed_dim)

        print("LayerNorm is ???", self.norm_audio)
        print("LayerNorm is ???", self.norm_video)

        self.init_weights(weight_init)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        # head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed_audio, std=.02)
        trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.normal_(self.bot_token, std=.02)
        if self.cls_token_audio is not None:
            nn.init.normal_(self.cls_token_audio, std=1e-6)
            nn.init.normal_(self.cls_token_video, std=1e-6)

    def forward(self, a, v):
        if self.dset in ['UCF']:
            a = a.squeeze(1)
        else:
            a = a.transpose(2, 3)
        
        a = self.patch_embed_audio(a)
        v = self.patch_embed_video(v)
        # print(a.shape, v.shape)
        # print(self.pos_embed_video.shape)
        cls_token_audio = self.cls_token_audio.expand(a.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        cls_token_video = self.cls_token_video.expand(v.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        a = torch.cat((cls_token_audio, a), dim=1)
        v = torch.cat((cls_token_video, v), dim=1)

        a = self.pos_drop(a + self.pos_embed_audio)
        v = self.pos_drop(v + self.pos_embed_video)

        a = self.audio_blocks(a)
        v = self.video_blocks(v)

        bot_token = self.bot_token.expand(a.shape[0], -1, -1)

        for i in range(self.multi_depth):
            a = torch.cat((bot_token, a), dim=1)
            v = torch.cat((bot_token, v), dim=1)

            a = self.bot_audio_blocks[i](a)
            v = self.bot_video_blocks[i](v)

            bot_token = (a[:, :self.num_bottle_token] + v[:, :self.num_bottle_token]) / 2
            a = a[:, self.num_bottle_token:]
            v = v[:, self.num_bottle_token:]

        # return self.pre_logits((multi[:, 0] + multi[:, a.size()[1]]) / 2.0)
        a = self.norm_audio(a)
        v = self.norm_video(v)
        # return self.pre_logits(a[:, 0] ), self.pre_logits(v[:, 0] ), self.pre_logits((a[:, 0] + v[:, 0]) / 2.0)
        return a, v


class MBTBackbone_Unimodal(nn.Module):
    def __init__(self, modality, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=8, multi_depth=2,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, distilled=False, num_bottle_token=4,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', nframes=2, o_nframes=3, fstride=16, tstride=16, input_fdim=128, input_tdim=1024, dset='CREMAD'):
        super().__init__()

        self.modality = modality # A or V
        print(input_fdim, input_tdim)
        self.dset = dset
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.multi_depth = multi_depth
        ''' move in dynamic'''
        self.num_bottle_token = num_bottle_token

        # Audio (Optical / Text)
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if dset == 'UCF':
            self.patch_embed_audio = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=2, embed_dim=embed_dim, frames=o_nframes)
            num_patches_audio = self.patch_embed_audio.num_patches
        else:
            self.patch_embed_audio = AudioPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches_audio  = f_dim * t_dim
            self.patch_embed_audio.num_patches = num_patches_audio
            new_proj = torch.nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=(fstride, tstride))
            self.patch_embed_audio.proj = new_proj
        self.pos_embed_audio = nn.Parameter(torch.zeros(1, num_patches_audio + 1, embed_dim))
        
        # Video (Text)
        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed_video = VideoPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, frames=nframes)
        num_patches_video = self.patch_embed_video.num_patches
        print(num_patches_video)
        self.pos_embed_video = nn.Parameter(torch.zeros(1, num_patches_video + self.num_tokens, embed_dim))

        self.bot_token = nn.Parameter(torch.zeros(1, num_bottle_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.audio_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth)])

        self.video_blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-multi_depth)])

        print(f'depth of attention bottleneck is {multi_depth}')
        print(f'Unimodal MBT now is (uni){depth-multi_depth} + (multi){multi_depth}')

        self.norm_audio = norm_layer(embed_dim)
        self.norm_video = norm_layer(embed_dim)

        print("LayerNorm is ???", self.norm_audio)
        print("LayerNorm is ???", self.norm_video)

        self.init_weights(weight_init)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        # head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed_audio, std=.02)
        trunc_normal_(self.pos_embed_video, std=.02)
        nn.init.normal_(self.bot_token, std=.02)
        if self.cls_token_audio is not None:
            nn.init.normal_(self.cls_token_audio, std=1e-6)
            nn.init.normal_(self.cls_token_video, std=1e-6)
        # named_apply(get_init_weights_vit(mode, head_bias), self)

    def forward(self, a, v):
        if self.modality == "uni_A":
            print("train on uni A")
            if self.dset in ['UCF']:
                a = a.squeeze(1)
            else:
                a = a.transpose(2, 3)

            a = self.patch_embed_audio(a)
            cls_token_audio = self.cls_token_audio.expand(a.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            a = torch.cat((cls_token_audio, a), dim=1)
            a = self.pos_drop(a + self.pos_embed_audio)
            a = self.audio_blocks(a)
            a = self.norm_audio(a)
        elif self.modality == "uni_V":
            print("train on uni V")
            v = self.patch_embed_video(v)
            cls_token_video = self.cls_token_video.expand(v.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            v = torch.cat((cls_token_video, v), dim=1)
            v = self.pos_drop(v + self.pos_embed_video)
            v = self.video_blocks(v)
            v = self.norm_video(v)
        else:
            raise ValueError("Unexpected modality in MBT_UNIMODAL")
        
        return a, v

