'''
Build the SHViT model family
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .shvit import SHViT
from timm.models.registry import register_model


SHViT_s1 = {
        'embed_dim': [128, 224, 320],
        'depth': [2, 4, 5],
        'partial_dim': [32, 48, 68],  
        'types' : ["i", "s", "s"]
    }

@register_model
def shvit_s1(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s1):
    model = SHViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s2 = {
        'embed_dim': [128, 308, 448],
        'depth': [2, 4, 5],
        'partial_dim': [32, 66, 96],
        'types' : ["i", "s", "s"]
    }

@register_model
def shvit_s2(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s2):
    model = SHViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s3 = {
        'embed_dim': [192, 352, 448],
        'depth': [3, 5, 5],
        'partial_dim': [48, 75, 96],  
        'types' : ["i", "s", "s"]
    }

@register_model
def shvit_s3(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s3):
    model = SHViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model


SHViT_s4 = {
        'embed_dim': [224, 336, 448],
        'depth': [4, 7, 6],
        'partial_dim': [48, 72, 96],
        'types' : ["i", "s", "s"]
    }

@register_model
def shvit_s4(num_classes=1000, pretrained=False, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=SHViT_s4):
    model = SHViT(num_classes=num_classes, distillation=distillation, **model_cfg)
    if pretrained:
        pretrained = _checkpoint_url_format.format(pretrained)
        checkpoint = torch.hub.load_state_dict_from_url(
            pretrained, map_location='cpu')
        d = checkpoint['model']
        D = model.state_dict()
        for k in d.keys():
            if D[k].shape != d[k].shape:
                d[k] = d[k][:, :, None, None]
        model.load_state_dict(d)
    if fuse:
        replace_batchnorm(model)
    return model



def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)