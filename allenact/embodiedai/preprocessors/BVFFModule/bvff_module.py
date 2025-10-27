import torch 
import torch.nn as nn
import numpy as np 

from .Cross_Attention_Fusion.cross_attention_fusion import cross_attn_fusion, FiLM_fusion


class bvff_module(nn.Module):
    def __init__(self, fusion_method):
        super().__init__()
        if(fusion_method == 'cross_attn'):
            self.bvff_module = cross_attn_fusion()

    def forward(self, rois_embed, visual_embed):
        if(rois_embed.shape[0]==1 and visual_embed.shape[0]==1): # rois_embed:[1, B, 768/384] visual_embed:[1, B, 1568]
            rois_embed = rois_embed.squeeze(dim=0)
            visual_embed = visual_embed.squeeze(dim=0)
            rois_embed = rois_embed.unsqueeze(1)
            visual_embed = visual_embed.unsqueeze(1)   
            joint_embeds = self.bvff_module(rois_embed, visual_embed) # [B, 1, dim]
            joint_embeds = joint_embeds.squeeze(dim=1)
            joint_embeds = joint_embeds.unsqueeze(dim=0) # [1, B, 1568+xxx]
        else:
            rollout_steps, agent_nums = rois_embed.shape[0], rois_embed.shape[1]
            rois_embed = rois_embed.reshape(-1, rois_embed.shape[-1])
            visual_embed = visual_embed.reshape(-1, visual_embed.shape[-1])
            rois_embed = rois_embed.unsqueeze(1)
            visual_embed = visual_embed.unsqueeze(1)      
            joint_embeds = self.bvff_module(rois_embed, visual_embed) # [B, 1, dim]       
            joint_embeds = joint_embeds.reshape(rollout_steps, agent_nums, -1)
        return joint_embeds 

