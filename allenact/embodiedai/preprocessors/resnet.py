from typing import List, Callable, Optional, Any, cast, Dict

import gym
import numpy as np
import torch
from torch import nn as nn
from torchvision import models
import os 

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super
import allenact.embodiedai.preprocessors.base_vit as vit 
from allenact.embodiedai.preprocessors.DDNEncoder.ddn_encoder import ddn_encoder

_MODEL_FUNC = {
    "vitt": vit.vit_t16,
    "vits": vit.vit_s16,
    "vitb": vit.vit_b16,
    "vitl": vit.vit_l16,
}


from sklearn.decomposition import PCA

# save image
all_brain_feats = []
all_lfmri_res = []
all_rfmri_res = []
all_brain_feats_debug = []
all_lfmri_res_debug = []
all_rfmri_res_debug = []
all_obs_imgs = []
all_obs_clean_imgs = []


class ResNetEmbedder(nn.Module):
    def __init__(self, 
        resnet,
        freeze = True, 
        emb_dim = 512, # visual embed. dim     
        pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()
        self.brain_encoder = ddn_encoder()

    def forward(self, x):
        with torch.no_grad():
            x_conv1_output = self.model.conv1(x) # x's shape is [batch_size, c=3, h=224, w=224]
            x_bn1_output = self.model.bn1(x_conv1_output)
            x_relu_output = self.model.relu(x_bn1_output)
            x_maxpool_output = self.model.maxpool(x_relu_output)

            x_layer1_output = self.model.layer1(x_maxpool_output)
            x_layer2_output = self.model.layer2(x_layer1_output)
            x_layer3_output = self.model.layer3(x_layer2_output)
            x_layer4_output = self.model.layer4(x_layer3_output) # torch.float32 tensor. shape is [batch_size, 512, 7, 7]

            # DDNEncoder
            brain_feat = self.brain_encoder(x)

            if not self.pool: # pool==False
                x_joint_brain_feat_tensor = {
                    "resnet_feat": x_layer4_output, # [B, 512, 7, 7]
                    "brain_feat": brain_feat
                }
                # print("x_layer4_output's shape is ",*x_layer4_output.shape)
                # print("brain_feat's shape is ",*brain_feat.shape)
                return  x_joint_brain_feat_tensor 
            else:
                x_avgpool_output = self.model.avgpool(x_layer4_output)
                x_flatten_output = torch.flatten(x_avgpool_output, 1)
                return x_flatten_output
    

class ResNetPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model."""

    def __init__(
        self,
        input_uuids: List[str],
        output_uuid: str,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        output_dims: int,
        pool: bool,
        torchvision_resnet_model: Callable[..., models.ResNet] = models.resnet18,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any
    ):
        def f(x, k):
            assert k in x, "{} must be set in ResNetPreprocessor".format(k)
            return x[k]

        def optf(x, k, default):
            return x[k] if k in x else default

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool
        self.make_model = torchvision_resnet_model

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )

        self._resnet: Optional[ResNetEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)

        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def resnet(self) -> ResNetEmbedder:
        if self._resnet is None:
            self._resnet = ResNetEmbedder(
                self.make_model(pretrained=True).to(self.device), pool=self.pool
            )
        return self._resnet

    def to(self, device: torch.device) -> "ResNetPreprocessor":
        self._resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x.to(self.device))
