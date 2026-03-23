import torch
from torch.utils.checkpoint import checkpoint

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe

from seg3d.utils.view_sampling import compute_angular_distance


class SAM2DualOrigMem(SAM2Base):
    """
    Extension of SAM2Base that supports multi-view / multi-camera memory
    using angular view proximity between frames.
    """

    def forward_image(self, inputs: dict):
        """Get the image features on the input batch (multi-modal: normal + point)."""
        # NOTE: image_encoder should be GeoSAM2MultimodalEncoder that accepts (normal, point)
        backbone_out = self.image_encoder(inputs["normal"], inputs["point"])
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

