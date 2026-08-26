import torch
import torch.nn as nn
from channelshuffle import channel_shuffle

class SpatialComplementaryBlock(nn.Module):
    def __init__(self, num_levels, in_channels, level):
        super(SpatialComplementaryBlock, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.level = level
        self.sc_convs = nn.ModuleList()
        for i in range(num_levels):
            self.sc_convs.append(
                nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=1),nn.Sigmoid())
            )
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, features):
        masked_features = [] 
        selective_feature = None

        for i in range(self.num_levels):
            mask = self.sc_convs[i](features[i])
            if i == self.level:
                selective_feature = mask * features[i]
            else:
                masked_feature = mask * features[i]
                masked_features.append(masked_feature)

        masked_features = torch.stack(masked_features)
        complementary_feature = torch.sum(masked_features, dim=0)
        gamma_map = self.gate_predictor(features[self.level])
        enhanced_feature = features[self.level] + selective_feature + gamma_map * complementary_feature
        return enhanced_feature, gamma_map


class SpatialComplementaryIntegrationModule(nn.Module):
    def __init__(self, num_levels, in_channels):
        super(SpatialComplementaryIntegrationModule, self).__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.sc_blocks = nn.ModuleList([
            SpatialComplementaryBlock(num_levels=num_levels,in_channels=in_channels,level=i)
            for i in range(num_levels)
        ])
        self.channel_interaction_fusion = nn.Sequential(
            nn.Conv2d(in_channels=num_levels * in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0,groups=num_levels),
            nn.GELU()
        )

    def forward(self, features):
        enhanced_features = []
        gamma_maps = []
        for scb in self.sc_blocks:
            enhanced_feature, gamma_map = scb(features)
            enhanced_features.append(enhanced_feature)
            gamma_maps.append(gamma_map)
        enhanced_features = torch.cat(enhanced_features,dim=1)
        enhanced_features = channel_shuffle(enhanced_features,groups=self.num_levels)
        fused_feature = self.channel_interaction_fusion(enhanced_features)
        return fused_feature, gamma_maps