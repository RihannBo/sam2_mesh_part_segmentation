import torch
import torch.nn as nn
import torch.nn.functional as f


class MultiScaleResidualFusion(nn.Module):
    """
    Apply ResidualFusionBlock independently at each FPN level.

    normal_feats: list of tensors [G_0, G_1, ..., G_{L-1}]
    point_feats:  list of tensors [P_0, P_1, ..., P_{L-1}]
    Each tensor has shape (B, C_l, H_l, W_l).
    """
    def __init__(self, channels_per_level):
        super().__init__()
        
        assert len(channels_per_level) > 0
        
        self.blocks = nn.ModuleList(
            [ResidualFusionBlock(c) for c in channels_per_level]
        )
    
    def forward(self, normal_feats, point_feats):
        
        assert len(normal_feats) == len(point_feats) == len(self.blocks)
        
        fused_list = []
        for G, P, block in zip(normal_feats, point_feats, self.blocks):
            fused_list.append(block(G, P))

        return fused_list

class ResidualFusionBlock(nn.Module):
    """
    Residual fusion of normal-map feature G and point-map feature P
    as described in GeoSAM2.

    Inputs:
        G: (B, C, H, W)  - normal-map feature
        P: (B, C, H, W)  - point-map feature

    Output:
        G_hat: (B, C, H, W)
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.conv = nn.Conv2d(
            in_channels=2*channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        
        self._zero_init()
        
    def _zero_init(self):
        
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, G: torch.Tensor, P:torch.Tensor) -> torch.Tensor:
        """
        G, P: (B, C, H, W)
        """
        assert G.shape == P.shape, \
            f"ResidualFusionBlock expects G and P with same shape, got {G.shape} vs {P.shape}"
        
        # concatenate along channel dimension
        X = torch.cat([G, P], dim=1)    # (B, 2C, H, W)
        # apply the convolution
        Y = self.conv(X)                # (B, C, H, W)
        # residual addition
        G_hat = G + Y
        
        return G_hat


