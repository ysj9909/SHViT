import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools
import numpy as np

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple

from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm

class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x
        

class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = GroupNorm(pdim)

        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init = 0))
        

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim = 1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim = 1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim = -1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim = 1))

        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, dim, qk_dim, pdim, type):
        super().__init__()
        if type == "s":    # for later stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
            self.ffn = Residual(FFN(dim, int(dim * 2)))
        elif type == "i":   # for early stages
            self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups = dim, bn_weight_init = 0))
            self.mixer = torch.nn.Identity()
            self.ffn = Residual(FFN(dim, int(dim * 2)))
    
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))


class Partial_ViT_Exp(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 frozen_stages = 0,
                 in_chans=3,
                 embed_dim=[128, 256, 384],
                 partial_dim = [32, 64, 96],
                 qk_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 types = ["s", "s", "s"],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 pretrained = None,
                 distillation=False,):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1))

        resolution = img_size // patch_size
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build SHViT blocks
        for i, (ed, kd, pd, dpth, do, t) in enumerate(
                zip(embed_dim, qk_dim, partial_dim, depth, down_ops, types)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(BasicBlock(ed, kd, pd, t))
            if do[0] == 'subsample':
                # Build SHViT downsample block
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i])),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2))),))
                blk.append(PatchMerging(*embed_dim[i:i + 2]))
                
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1])),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2))),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        if pretrained is not None:
            self.init_weights(pretrained = pretrained) 

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')
            
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {filename}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            model_state_dict = self.state_dict()
            # bicubic interpolate attention_biases if not match

            rpe_idx_keys = [
                k for k in state_dict.keys() if "attention_bias_idxs" in k]
            for k in rpe_idx_keys:
                print("deleting key: ", k)
                del state_dict[k]

            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "attention_biases" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model_state_dict[k]
                nH1, L1 = relative_position_bias_table_pretrained.size()
                nH2, L2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {k} due to different number of heads")
                else:
                    if L1 != L2:
                        print("resizing key {} from {} * {} to {} * {}".format(k, L1, L1, L2, L2))
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                            nH2, L2)     

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Partial_ViT_Exp, self).train(mode)
        self._freeze_stages()
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)
        return tuple(outs)


shvit_s4 = {
        'img_size': 256,
        'patch_size': 16,
        'embed_dim': [224, 336, 448],
        'depth': [4, 7, 6],
        'partial_dim': [48, 72, 96],
        'types' : ["i", "s", "s"]
    }


@BACKBONES.register_module()
def shvit_s4(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=shvit_s4):
    model = Partial_ViT_Exp(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    return model