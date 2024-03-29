# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:29:40 2021

@author: 5106
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:27:27 2021

@author: 5106
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange

# MLP module
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=False , drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# MLP-based Permutator module
class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, segment_dim=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim3, dim3, bias=qkv_bias)

        self.reweight = Mlp(dim1, dim1 // 2, dim1 *3)

        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H*S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# Complete Permutator block
class PermutatorBlock(nn.Module):

    def __init__(self, dim1, dim2, dim3, segment_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=False, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn = WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.attn = mlp_fn(dim1, dim2, dim3, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim1)
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) / self.skip_lam
        x = x + self.mlp(self.norm2(x)) / self.skip_lam
        
        return x

# Convolutional module
class conv_block1(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, strides,pads, dilas):
        super(conv_block1, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=strides, padding=pads, dilation=dilas,bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
            )

    def forward(self, x):

        x = self.conv(x)
        return x

# Upsampling module
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
# Residual module
class _Res_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Block, self).__init__()

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.res_conb = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.PReLU()
        self.instan = nn.InstanceNorm2d(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.instan(self.res_conv(x)))
        y = self.res_conb(y)
        y *= al
        y = torch.add(y, x)
        return y

# Channel estimation network
class channel_est(nn.Module):
    def __init__(self):
        super(channel_est, self).__init__()

        n1 = 48
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        in_ch=2
        out_ch=2
                
        self.Conv11 = conv_block1(in_ch, filters[0],1,2,2)
        self.Conv22 = conv_block1(filters[0], filters[1],2,1,1)
        self.Conv33 = conv_block1(filters[1], filters[2],2,1,1)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up33 = conv_block1(filters[2], filters[1],1,1,1)
        

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up22 = conv_block1(filters[1], filters[0],1,1,1)
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        
        seg_dim1 = 24
        seg1 = filters[1]//seg_dim1
        
        seg_dim33 = 24
        seg33 = filters[2]//seg_dim33
        
        seg_dim2 = 8
        seg2 = filters[0]//seg_dim2

        self.mlp_mixer11 = PermutatorBlock(dim1 = filters[0], dim2 = 512*seg2, dim3 = 32*seg2, segment_dim=seg_dim2)
        self.mlp_mixer22 = PermutatorBlock(dim1 = filters[1], dim2 = 256*seg1, dim3 = 16*seg1, segment_dim=seg_dim1)
        self.mlp_mixer33 = PermutatorBlock(dim1 = filters[2], dim2 = 128*seg33, dim3 = 8*seg33, segment_dim=seg_dim33)

    def forward(self, x):
        
        e1 = self.Conv11(x) 
        e1 = rearrange(e1, 'b c h w -> b h w c')
        e1 = self.mlp_mixer11(e1)
        e1 = rearrange(e1, 'b h w c-> b c h w')

        
        e2 = self.Conv22(e1)
        e2 = rearrange(e2, 'b c h w -> b h w c')
        e2 = self.mlp_mixer22(e2)
        e2 = rearrange(e2, 'b h w c-> b c h w')
        
        
        e3 = self.Conv33(e2)
        e3 = rearrange(e3, 'b c h w -> b h w c')
        e3 = self.mlp_mixer33(e3)
        e3 = rearrange(e3, 'b h w c-> b c h w')
        

        d3 = self.Up3(e3) 
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up33(d3)
        d3 = rearrange(d3, 'b c h w -> b h w c')
        d3 = self.mlp_mixer22(d3)
        d3 = rearrange(d3, 'b h w c-> b c h w')
        

        d2 = self.Up2(d3) 
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up22(d2)
        d2 = rearrange(d2, 'b c h w -> b h w c')
        d2 = self.mlp_mixer11(d2)
        d2 = rearrange(d2, 'b h w c-> b c h w')
        
        out = self.Conv(d2)
            
        return out

# NMSE function
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real  + 1j * (x_imag )
    x_hat_C = x_hat_real  + 1j * (x_hat_imag )
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

# Data argumentation operations to avoid the network overfitting 
def _cutmix(im2, prob=1.0, alpha=1.0):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return None

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = int(h*cut_ratio), int(w*cut_ratio)

    fcy = np.random.randint(0, h-ch+1)
    fcx = np.random.randint(0, w-cw+1)
    tcy, tcx = fcy, fcx
    rindex = torch.randperm(im2.size(0)).to(im2.device)

    return {
        "rindex": rindex, "ch": ch, "cw": cw,
        "tcy": tcy, "tcx": tcx, "fcy": fcy, "fcx": fcx,
    }

def cutmixup(
    im1, im2,    
    mixup_prob=1.0, mixup_alpha=1.0,
    cutmix_prob=1.0, cutmix_alpha=1.0
):
    c = _cutmix(im2, cutmix_prob, cutmix_alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    v = np.random.beta(mixup_alpha, mixup_alpha)
    if mixup_alpha <= 0 or np.random.rand(1) >= mixup_prob:
        im2_aug = im2[rindex, :]
        im1_aug = im1[rindex, :]

    else:
        im2_aug = v * im2 + (1-v) * im2[rindex, :]
        im1_aug = v * im1 + (1-v) * im1[rindex, :]

    # apply mixup to inside or outside
    if np.random.random() > 0.5:
        im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2_aug[..., fcy:fcy+ch, fcx:fcx+cw]
        im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1_aug[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
    else:
        im2_aug[..., tcy:tcy+ch, tcx:tcx+cw] = im2[..., fcy:fcy+ch, fcx:fcx+cw]
        im1_aug[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[..., hfcy:hfcy+hch, hfcx:hfcx+hcw]
        im2, im1 = im2_aug, im1_aug

    return im1, im2

def rgb(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2

    perm = np.random.permutation(2)

    im1 = im1[:,perm,:,:]
    im2 = im2[:,perm,:,:]

    return im1, im2

def rgb1(im1, im2, prob=1.0):
    if np.random.rand(1) >= prob:
        return im1, im2
    
    se = np.zeros(2)
    se[0]=1
    se[1]=-1
    
    r = np.random.randint(2)
    phase = se[r]
    im1[:,0,:,:] = phase*im1[:,0,:,:]
    im2[:,0,:,:] = phase*im2[:,0,:,:]
    r = np.random.randint(2)
    phase = se[r]
    im1[:,1,:,:] = phase*im1[:,1,:,:]
    im2[:,1,:,:] = phase*im2[:,1,:,:]

    return im1, im2

def cutmix(im1, im2, prob=1.0, alpha=1.0):
    c = _cutmix(im2, prob, alpha)
    if c is None:
        return im1, im2

    scale = im1.size(2) // im2.size(2)
    rindex, ch, cw = c["rindex"], c["ch"], c["cw"]
    tcy, tcx, fcy, fcx = c["tcy"], c["tcx"], c["fcy"], c["fcx"]

    hch, hcw = ch*scale, cw*scale
    hfcy, hfcx, htcy, htcx = fcy*scale, fcx*scale, tcy*scale, tcx*scale

    im2[..., tcy:tcy+ch, tcx:tcx+cw] = im2[rindex, :, fcy:fcy+ch, fcx:fcx+cw]
    im1[..., htcy:htcy+hch, htcx:htcx+hcw] = im1[rindex, :, hfcy:hfcy+hch, hfcx:hfcx+hcw]

    return im1, im2

def mixup(im1, im2, prob=1.0, alpha=1.2):
    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(im1.size(0)).to(im2.device)

    im1 = v * im1 + (1-v) * im1[r_index, :]
    im2 = v * im2 + (1-v) * im2[r_index, :]
    
    return im1, im2