# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from .position_encoding import build_position_encoding
from .hubconf import dinov2_vitb14, dinov2_vits14
from .utils.misc import NestedTensor


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
        # xs = self[0](tensor_list)

        # return xs

class DINO_backbone(nn.Module):
    def __init__(self, args):
        super().__init__()  
        self.vit_arc = args.vit_arc
        if(self.vit_arc=='vit_b'):
            self.backbone = dinov2_vitb14(pretrained=True)
            self.num_channels = 768
        else:
            self.backbone = dinov2_vits14(pretrained=True)
            self.num_channels = 384            
        for _, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        self.backbone._modules["blocks"][-1*args.enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)

    def hook_fn_forward_qkv(self, module, input, output) -> Callable:
        self.qkv_feats['qkv_feats'] = output    
    
    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
        xs = self.backbone.get_intermediate_layers(xs)[0]
        feats = self.qkv_feats['qkv_feats']
        if(self.vit_arc=='vit_b'):
            nh = 12
        else:
            nh = 6
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        xs = q[:,1:,:]
        xs = {'layer_top':xs}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            x = torch.reshape(x, (x.shape[0],int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1])),self.num_channels)).permute(0,3,1,2)
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        
        # Braiav-CLS 
        # out = q[:,:1,:].squeeze()

        return out 
        
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = DINO_backbone(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model