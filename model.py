import torch
import torch.nn as nn
from pget import PromptGuidedEndmemberTransformer
from scim import SpatialComplementaryIntegrationModule
from utils import LayerNormProxy


class PGTNet(nn.Module):
    def __init__(self, P, L, size, patch, dim, init_bundles, bundle_sizes):
        super(PGTNet, self).__init__()
        self.P, self.L, self.size, self.dim, self.patch = P, L, size, dim, patch
        self.bundles = nn.ParameterList([nn.Parameter(b, requires_grad=False) for b in init_bundles])
        self.bundle_sizes = bundle_sizes
        self.patch_proj = nn.Sequential(
            nn.Conv2d(self.L, self.dim, 1, 1, 0),
            LayerNormProxy(self.dim)
        )

        self.intra_weight_layers = nn.ModuleList()
        for bundle_size in self.bundle_sizes:
            self.intra_weight_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=self.dim, out_channels=bundle_size, kernel_size=1, stride=1, padding=0),
                    nn.Softmax(dim=1)
                    )
            )
        
        self.encoder_abd = nn.Sequential(
            nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(in_channels=self.dim, out_channels=self.P, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.pget = PromptGuidedEndmemberTransformer(
            image_size=self.size, 
            patch_size=self.patch, 
            dim=self.dim, 
            depth=2, 
            heads=4, 
            mlp_dim=2*self.dim, 
            num_endmembers=self.P, 
            dropout=0.0, 
            emb_dropout=0.0
        )
        self.scim = SpatialComplementaryIntegrationModule(num_levels=self.P,in_channels=self.dim)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x, bundle_mean):
        x_emb = self.patch_proj(x)
        prompt_embeddings = self.patch_proj(bundle_mean.unsqueeze(0).unsqueeze(3))
        prompt_embeddings = prompt_embeddings.squeeze(3).squeeze(0)
        edm_features = self.pget(x_emb,prompt_embeddings)
        fused_feature, gamma_maps = self.scim(edm_features)

        intra_weights = []
        for i, encoder_intra in enumerate(self.intra_weight_layers):
            intra_weight = encoder_intra(edm_features[i])
            intra_weight = intra_weight.flatten(start_dim=-2).squeeze(0)
            intra_weights.append(intra_weight)

        weighted_bundles = []
        for i in range(self.P):
            weighted_bundle = self.bundles[i] @ intra_weights[i]
            weighted_bundle = weighted_bundle.unsqueeze(0)
            weighted_bundles.append(weighted_bundle)

        weighted_endmember = torch.cat(weighted_bundles, dim=0)
        abu_est = self.encoder_abd(fused_feature)
        abu_est = abu_est.flatten(start_dim=-2).squeeze(0).transpose(0, 1).unsqueeze(2)
        endmember = weighted_endmember.permute(2, 1, 0)
        re_result = (endmember @ abu_est).squeeze(2)
        output_weights = []
        
        for i, w in enumerate(intra_weights):
            k = w.shape[0]
            w_map = w.view(k, self.size, self.size) 
            output_weights.append(w_map)
            
        abu_est = abu_est.squeeze(2)
        re_result = re_result.unsqueeze(0)

        return abu_est, re_result, endmember, output_weights, gamma_maps

