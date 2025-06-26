import torch.nn as nn
import torch

from pi_seg.utils.serialization import serialize
from pi_seg.model.base import ISModel
from pi_seg.model.modifiers import LRMult
from pi_seg.model.ops import DistMaps
from pi_seg.model.modeling.probabilistic_unet.segmentator import ProbabilisticUnet


class DeepPI(ISModel):
    @serialize
    def __init__(self, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)
        
        self.feature_extractor = ProbabilisticUnet(input_channels=64+1, num_classes=1, num_filters=[64, 128, 256, 512, 1024], latent_dim=6, no_convs_fcomb=2, beta=10.0)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        self.dist_maps = DistMaps(norm_radius=20, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def backbone_forward(self, image, mask=None, sample_cnt=1, coord_features=None, training=True, un_weight=False, un_scale=(2, 1)):        
        if training:
            self.feature_extractor.forward(patch=image, segm=mask, coord_features=coord_features, training=training)
            ps = [self.feature_extractor.sample(testing=False) for _ in range(sample_cnt)]
            
            outputs = self.feature_extractor.elbo(mask, ps, un_weight=un_weight, un_scale=un_scale)
            outputs['samples'] = ps

        else:
            self.feature_extractor.forward(patch=image, segm=mask, coord_features=coord_features, training=training)
            ps = [self.feature_extractor.sample(testing=True) for _ in range(sample_cnt)]
            
            outputs = {'samples': ps}

        return outputs

    def forward(self, image, points, sample_cnt=1, mask=None, training=True, un_weight=False, un_scale=(2, 1)):        
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:,1:,:,:]
        coord_features = self.maps_transform(coord_features)
        
        outputs = self.backbone_forward(image=image, mask=mask, sample_cnt=sample_cnt, coord_features=coord_features, 
                                        training=training, un_weight=un_weight, un_scale=un_scale)
        
        outputs['click_map'] = click_map

        return outputs
