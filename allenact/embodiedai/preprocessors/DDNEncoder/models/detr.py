import torch
import torch.nn as nn

from .backbone import build_backbone
from .transformer import build_transformer
from .utils.misc import NestedTensor, nested_tensor_from_tensor_list


class DETR(nn.Module):
    def __init__(self, args, backbone, transformer, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.query_pos = torch.zeros_like(self.query_embed.weight)
        feature_dim = self.hidden_dim     
        self.backbone = backbone
        self.decoder_arch = args.decoder_arch
        self.enc_layers = args.enc_layers
        self.dec_layers = args.dec_layers    
        self.readout_res = args.readout_res
        self.masks = False
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        if self.readout_res == 'hemis':
            self.lh_embed = nn.Sequential(nn.Linear(feature_dim, args.lh_vs))
            self.rh_embed = nn.Sequential(nn.Linear(feature_dim, args.rh_vs))
        elif self.readout_res != 'hemis':
            lh_vs = args.lh_vs
            rh_vs = args.rh_vs
            # #stream 0
            # self.lh_embed_0 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_0 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 1
            # self.lh_embed_1 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_1 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 2
            # self.lh_embed_2 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_2 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 3
            # self.lh_embed_3 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_3 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            #  #stream 4
            # self.lh_embed_4 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_4 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 5
            # self.lh_embed_5 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_5 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 6
            # self.lh_embed_6 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_6 = nn.Sequential(nn.Linear(feature_dim, rh_vs))
            # #stream 7 
            # self.lh_embed_7 = nn.Sequential(nn.Linear(feature_dim, lh_vs))
            # self.rh_embed_7 = nn.Sequential(nn.Linear(feature_dim, rh_vs))   

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        with torch.no_grad():
            features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        pos_embed = pos[-1]
        input_proj_src = src
        src_all = input_proj_src
        hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.masks, src_all) 
        output_tokens = hs[-1]

        if self.readout_res == 'hemis':
            lh_f_pred = self.lh_embed(output_tokens[:,0,:])
            rh_f_pred = self.rh_embed(output_tokens[:,1,:])
        else: #if self.readout_res == 'streams':
            pass 
            # lh_f_pred_0 = self.lh_embed_0(output_tokens[:,0,:])
            # lh_f_pred_1 = self.lh_embed_1(output_tokens[:,1,:])
            # lh_f_pred_2 = self.lh_embed_2(output_tokens[:,2,:])
            # lh_f_pred_3 = self.lh_embed_3(output_tokens[:,3,:])
            # lh_f_pred_4 = self.lh_embed_4(output_tokens[:,4,:])
            # lh_f_pred_5 = self.lh_embed_5(output_tokens[:,5,:])
            # lh_f_pred_6 = self.lh_embed_6(output_tokens[:,6,:])
            # lh_f_pred_7 = self.lh_embed_7(output_tokens[:,7,:])

            # rh_f_pred_0 = self.rh_embed_0(output_tokens[:,8,:])
            # rh_f_pred_1 = self.rh_embed_1(output_tokens[:,9,:])
            # rh_f_pred_2 = self.rh_embed_2(output_tokens[:,10,:])
            # rh_f_pred_3 = self.rh_embed_3(output_tokens[:,11,:])
            # rh_f_pred_4 = self.rh_embed_4(output_tokens[:,12,:])
            # rh_f_pred_5 = self.rh_embed_5(output_tokens[:,13,:])
            # rh_f_pred_6 = self.rh_embed_6(output_tokens[:,14,:])
            # rh_f_pred_7 = self.rh_embed_7(output_tokens[:,15,:])

            # lh_f_pred = torch.stack((lh_f_pred_0, lh_f_pred_1, lh_f_pred_2,lh_f_pred_3,lh_f_pred_4,lh_f_pred_5,lh_f_pred_6,lh_f_pred_7), dim=2)
            # rh_f_pred = torch.stack((rh_f_pred_0, rh_f_pred_1, rh_f_pred_2,rh_f_pred_3,rh_f_pred_4,rh_f_pred_5,rh_f_pred_6,rh_f_pred_7), dim=2)

        # out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}
        out = {'output_tokens': output_tokens}
        
        # BraiNav-CLS
        
        # if isinstance(samples, (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(samples)

        # with torch.no_grad():
        #     features = self.backbone(samples)
        
        # out = {'output_tokens': features}
            
        return out

def build_models(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = DETR(
        args,
        backbone,
        transformer,
        num_queries=args.num_queries,
    )

    return model 