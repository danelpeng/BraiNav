import torch
import torch.nn as nn
import numpy as np 

from .models.detr import build_models
from .models.utils.get_args import get_brain_args
from .models.utils.misc import fmri_pooling

# ddn_encoder
class ddn_encoder(nn.Module):
    def __init__(self):
        super(ddn_encoder, self).__init__()
        args = get_brain_args()
        self.freeze = True
        #rois idx
        rois_idx = np.load('BraiNav/allenact/embodiedai/preprocessors/DDNEncoder/checkpoints/rois_idx.npz')
        self.lh_RSC_idx = rois_idx['left_RSC_idx']
        self.rh_RSC_idx = rois_idx['right_RSC_idx']
        self.lh_early_idx = rois_idx['left_early_idx']
        self.rh_early_idx = rois_idx['right_early_idx']
        self.lh_midventral_idx = rois_idx['left_midventral_idx']
        self.rh_midventral_idx = rois_idx['right_midventral_idx']
        self.lh_midlateral_idx = rois_idx['left_midlateral_idx']
        self.rh_midlateral_idx = rois_idx['right_midlateral_idx']
        self.lh_midparietal_idx = rois_idx['left_midparietal_idx']
        self.rh_midparietal_idx = rois_idx['right_midparietal_idx']
        self.lh_ventral_idx = rois_idx['left_ventral_idx']
        self.rh_ventral_idx = rois_idx['right_ventral_idx']
        self.lh_lateral_idx = rois_idx['left_lateral_idx']
        self.rh_lateral_idx = rois_idx['right_lateral_idx']
        self.lh_parietal_idx = rois_idx['left_parietal_idx']
        self.rh_parietal_idx = rois_idx['right_parietal_idx']

        # DETR backbone
        self.backbone = build_models(args)
        if(args.vit_arc=='vit_b'):
            pretrained_model_state = torch.load("BraiNav/allenact/embodiedai/preprocessors/DDNEncoder/checkpoints/detr_dino_vitb_1_streams_inc_16_subj01_run1.pth", map_location='cuda:0') 
        else:
            pretrained_model_state = torch.load("BraiNav/allenact/embodiedai/preprocessors/DDNEncoder/checkpoints/detr_dino_vitb_1_streams_inc_16_subj01_run1.pth", map_location='cuda:0') 
        if 'model' in pretrained_model_state: pretrained_model_state = pretrained_model_state['model']
        missing, unexpected = self.backbone.load_state_dict(pretrained_model_state, strict=False)
        print(f'[initialize_weight] missing_keys={missing}')
        print(f'[initialize_weight] unexpected_keys={unexpected}')
        if(self.freeze):
            for p in self.backbone.parameters():
                p.requires_grad = False        

        # DETR output tokens embedding heads
        self.detr_token_embed_0 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_1 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_2 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_3 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_4 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_5 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_6 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_7 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_8 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_9 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_10 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_11 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_12 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_13 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_14 = nn.Sequential(nn.Linear(768, 48))
        self.detr_token_embed_15 = nn.Sequential(nn.Linear(768, 48))  
        
        self.brain_projector = nn.Sequential(nn.Linear(672, 768))
        self.low_brain_projector = nn.Sequential(nn.Linear(96, 768))
        self.middle_brain_projector = nn.Sequential(nn.Linear(288, 768))
        self.high_brain_projector = nn.Sequential(nn.Linear(288, 768))

    def forward(self, x):
        fmri_pred = self.backbone(x)
        # lh_f_pred = fmri_pred["lh_f_pred"] # [batch, 19004, 8]
        # rh_f_pred = fmri_pred["rh_f_pred"] # [batch, 20544, 8]
        output_tokens = fmri_pred["output_tokens"]
        # fmri_embed = self.fmri_embedding_heads(lh_f_pred, rh_f_pred)
        fmri_embed = self.detr_token_embed_embedding_heads(output_tokens)

        return fmri_embed

    def fmri_embedding_heads(self, lh_f_pred, rh_f_pred):
        lh_RSC_fmri = lh_f_pred[:,:,0][:,self.lh_RSC_idx]
        rh_RSC_fmri = rh_f_pred[:,:,0][:,self.rh_RSC_idx]
        lh_early_fmri = lh_f_pred[:,:,1][:,self.lh_early_idx]
        rh_early_fmri = rh_f_pred[:,:,1][:,self.rh_early_idx]
        lh_midventral_fmri = lh_f_pred[:,:,2][:,self.lh_midventral_idx]
        rh_midventral_fmri = rh_f_pred[:,:,2][:,self.rh_midventral_idx]
        lh_midlateral_fmri = lh_f_pred[:,:,3][:,self.lh_midlateral_idx]
        rh_midlateral_fmri = rh_f_pred[:,:,3][:,self.rh_midlateral_idx]
        lh_midparietal_fmri = lh_f_pred[:,:,4][:,self.lh_midparietal_idx]
        rh_midparietal_fmri = rh_f_pred[:,:,4][:,self.rh_midparietal_idx]
        lh_ventral_fmri = lh_f_pred[:,:,5][:,self.lh_ventral_idx]
        rh_ventral_fmri = rh_f_pred[:,:,5][:,self.rh_ventral_idx]
        lh_lateral_fmri = lh_f_pred[:,:,6][:,self.lh_lateral_idx]
        rh_lateral_fmri = rh_f_pred[:,:,6][:,self.rh_lateral_idx]
        lh_parietal_fmri = lh_f_pred[:,:,7][:,self.lh_parietal_idx]
        rh_parietal_fmri = rh_f_pred[:,:,7][:,self.rh_parietal_idx]

        lh_RSC_embed = self.lh_RSC_embed(lh_RSC_fmri)
        lh_early_embed = self.lh_early_embed(lh_early_fmri)
        lh_midventral_embed = self.lh_midventral_embed(lh_midventral_fmri)
        lh_midlateral_embed = self.lh_midlateral_embed(lh_midlateral_fmri)
        lh_midparietal_embed = self.lh_midparietal_embed(lh_midparietal_fmri)
        lh_ventral_embed = self.lh_ventral_embed(lh_ventral_fmri)
        lh_lateral_embed = self.lh_lateral_embed(lh_lateral_fmri)
        lh_parietal_embed = self.lh_parietal_embed(lh_parietal_fmri)

        rh_RSC_embed = self.rh_RSC_embed(rh_RSC_fmri)
        rh_early_embed = self.rh_early_embed(rh_early_fmri)
        rh_midventral_embed = self.rh_midventral_embed(rh_midventral_fmri)
        rh_midlateral_embed = self.rh_midlateral_embed(rh_midlateral_fmri)
        rh_midparietal_embed = self.rh_midparietal_embed(rh_midparietal_fmri)
        rh_ventral_embed = self.rh_ventral_embed(rh_ventral_fmri)
        rh_lateral_embed = self.rh_lateral_embed(rh_lateral_fmri)
        rh_parietal_embed = self.rh_parietal_embed(rh_parietal_fmri)

        all_rois_fmri = torch.cat([lh_RSC_embed, rh_RSC_embed, lh_early_embed, rh_early_embed, lh_midventral_embed, rh_midventral_embed, lh_midlateral_embed, rh_midlateral_embed,
                                        lh_midparietal_embed, rh_midparietal_embed, lh_ventral_embed, rh_ventral_embed, lh_lateral_embed, rh_lateral_embed, lh_parietal_embed, rh_parietal_embed], dim=-1)
        
        # all_rois_fmri = self.brain_projector(all_rois_fmri)
        return all_rois_fmri

    def fmri_roi_pooling_embedding_heads(self, lh_f_pred, rh_f_pred):
        """
        ref roi pooling fMRI Dimensionality Reduction from <<Multi-View Multi-Label Fine-Grained Emotion Decoding From Human Brain Activity>>
        """
        lh_early_fmri = lh_f_pred[:,:,1][:,self.lh_early_idx]
        rh_early_fmri = rh_f_pred[:,:,1][:,self.rh_early_idx]
        lh_midventral_fmri = lh_f_pred[:,:,2][:,self.lh_midventral_idx]
        rh_midventral_fmri = rh_f_pred[:,:,2][:,self.rh_midventral_idx]
        lh_midlateral_fmri = lh_f_pred[:,:,3][:,self.lh_midlateral_idx]
        rh_midlateral_fmri = rh_f_pred[:,:,3][:,self.rh_midlateral_idx]
        lh_midparietal_fmri = lh_f_pred[:,:,4][:,self.lh_midparietal_idx]
        rh_midparietal_fmri = rh_f_pred[:,:,4][:,self.rh_midparietal_idx]
        lh_ventral_fmri = lh_f_pred[:,:,5][:,self.lh_ventral_idx]
        rh_ventral_fmri = rh_f_pred[:,:,5][:,self.rh_ventral_idx]
        lh_lateral_fmri = lh_f_pred[:,:,6][:,self.lh_lateral_idx]
        rh_lateral_fmri = rh_f_pred[:,:,6][:,self.rh_lateral_idx]
        lh_parietal_fmri = lh_f_pred[:,:,7][:,self.lh_parietal_idx]
        rh_parietal_fmri = rh_f_pred[:,:,7][:,self.rh_parietal_idx]       

        lh_early_pooling_fmri = fmri_pooling(lh_early_fmri, self.lh_early_grid_pooling_idx)
        rh_early_pooling_fmri = fmri_pooling(rh_early_fmri, self.rh_early_grid_pooling_idx)
        lh_midventral_pooling_fmri = fmri_pooling(lh_midventral_fmri, self.lh_midventral_grid_pooling_idx)
        rh_midventral_pooling_fmri = fmri_pooling(rh_midventral_fmri, self.rh_midventral_grid_pooling_idx)
        lh_midlateral_pooling_fmri = fmri_pooling(lh_midlateral_fmri, self.lh_midlateral_grid_pooling_idx)
        rh_midlateral_pooling_fmri = fmri_pooling(rh_midlateral_fmri, self.rh_midlateral_grid_pooling_idx)
        lh_midparietal_pooling_fmri = fmri_pooling(lh_midparietal_fmri, self.lh_midparietal_grid_pooling_idx)
        rh_midparietal_pooling_fmri = fmri_pooling(rh_midparietal_fmri, self.rh_midparietal_grid_pooling_idx)
        lh_ventral_pooling_fmri = fmri_pooling(lh_ventral_fmri, self.lh_ventral_grid_pooling_idx)
        rh_ventral_pooling_fmri = fmri_pooling(rh_ventral_fmri, self.rh_ventral_grid_pooling_idx)
        lh_lateral_pooling_fmri = fmri_pooling(lh_lateral_fmri, self.lh_lateral_grid_pooling_idx)
        rh_lateral_pooling_fmri = fmri_pooling(rh_lateral_fmri, self.rh_lateral_grid_pooling_idx)
        lh_parietal_pooling_fmri = fmri_pooling(lh_parietal_fmri, self.lh_parietal_grid_pooling_idx)
        rh_parietal_pooling_fmri = fmri_pooling(rh_parietal_fmri, self.rh_parietal_grid_pooling_idx)

        all_rois_fmri = torch.cat([lh_early_pooling_fmri, rh_early_pooling_fmri, lh_midventral_pooling_fmri, rh_midventral_pooling_fmri, lh_midlateral_pooling_fmri, rh_midlateral_pooling_fmri, lh_midparietal_pooling_fmri, rh_midparietal_pooling_fmri,
                            lh_ventral_pooling_fmri, rh_ventral_pooling_fmri, lh_lateral_pooling_fmri, rh_lateral_pooling_fmri, lh_parietal_pooling_fmri, rh_parietal_pooling_fmri], dim=-1)

        return all_rois_fmri
    
    def detr_token_embed_embedding_heads(self, output_tokens):

        # embed_0 = self.detr_token_embed_0(output_tokens[:,0,:])
        embed_1 = self.detr_token_embed_1(output_tokens[:,1,:])
        embed_2 = self.detr_token_embed_2(output_tokens[:,2,:])
        embed_3 = self.detr_token_embed_3(output_tokens[:,3,:])
        embed_4 = self.detr_token_embed_4(output_tokens[:,4,:])
        embed_5 = self.detr_token_embed_5(output_tokens[:,5,:])
        embed_6 = self.detr_token_embed_6(output_tokens[:,6,:])
        embed_7 = self.detr_token_embed_7(output_tokens[:,7,:])
        # embed_8 = self.detr_token_embed_8(output_tokens[:,8,:])
        embed_9 = self.detr_token_embed_9(output_tokens[:,9,:])
        embed_10 = self.detr_token_embed_10(output_tokens[:,10,:])
        embed_11 = self.detr_token_embed_11(output_tokens[:,11,:])
        embed_12 = self.detr_token_embed_12(output_tokens[:,12,:])
        embed_13 = self.detr_token_embed_13(output_tokens[:,13,:])
        embed_14 = self.detr_token_embed_14(output_tokens[:,14,:])
        embed_15 = self.detr_token_embed_15(output_tokens[:,15,:])
        
        # all ROI
        embed = [embed_1, embed_9, embed_2, embed_10, embed_3, embed_11, embed_4,
                 embed_12, embed_5, embed_13, embed_6, embed_14, embed_7, embed_15]
        embed = torch.cat(embed, dim=-1)
        embed = self.brain_projector(embed)
        
        # low ROI
        # embed = [embed_1, embed_9]
        # embed = torch.cat(embed, dim=-1)
        # embed = self.low_brain_projector(embed)
        
        # middle ROI
        # embed = [embed_2, embed_10, embed_3, embed_11, embed_4, embed_12]
        # embed = torch.cat(embed, dim=-1)
        # embed = self.middle_brain_projector(embed)

        # high ROI
        # embed = [embed_5, embed_13, embed_6, embed_14, embed_7, embed_15]
        # embed = torch.cat(embed, dim=-1)
        # embed = self.high_brain_projector(embed)
        
        return embed        
    
