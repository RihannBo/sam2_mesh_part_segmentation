import math
import torch
import torch.nn as nn

from seg3d.models.backbones.residual_fusion import MultiScaleResidualFusion


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _freeze_params(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


# ---------------------------------------------------------------------
# Two-bank LoRA QKV (hook-based, SAM2-safe)
# ---------------------------------------------------------------------

class _TwoBankLoRAQKV(nn.Module):
    """
    Two-bank LoRA for SAM2 attention (Q,V only).
    Banks:
      - normal
      - point
      - dual  (first half normal, second half point)
    """

    def __init__(self, qkv: nn.Linear, rank: int):
        super().__init__()
        assert isinstance(qkv, nn.Linear)
        self.qkv = qkv
        self.dim = qkv.in_features

        # --- normal bank ---
        self.A_q_n = nn.Linear(self.dim, rank, bias=False)
        self.B_q_n = nn.Linear(rank, self.dim, bias=False)
        self.A_v_n = nn.Linear(self.dim, rank, bias=False)
        self.B_v_n = nn.Linear(rank, self.dim, bias=False)

        # --- point bank ---
        self.A_q_p = nn.Linear(self.dim, rank, bias=False)
        self.B_q_p = nn.Linear(rank, self.dim, bias=False)
        self.A_v_p = nn.Linear(self.dim, rank, bias=False)
        self.B_v_p = nn.Linear(rank, self.dim, bias=False)

        for A in (self.A_q_n, self.A_v_n, self.A_q_p, self.A_v_p):
            nn.init.kaiming_uniform_(A.weight, a=math.sqrt(5))
        for B in (self.B_q_n, self.B_v_n, self.B_q_p, self.B_v_p):
            nn.init.zeros_(B.weight)

        self._bank = "normal"

    @torch.no_grad()
    def set_bank(self, bank: str) -> None:
        if bank not in ("normal", "point", "dual"):
            raise ValueError(f"Invalid bank: {bank}")
        self._bank = bank

    def delta(self, x: torch.Tensor):
        if self._bank == "normal":
            return (
                self.B_q_n(self.A_q_n(x)),
                self.B_v_n(self.A_v_n(x)),
            )

        if self._bank == "point":
            return (
                self.B_q_p(self.A_q_p(x)),
                self.B_v_p(self.A_v_p(x)),
            )

        # --- dual ---
        B = x.shape[0] // 2
        dq = torch.cat(
            [
                self.B_q_n(self.A_q_n(x[:B])),
                self.B_q_p(self.A_q_p(x[B:])),
            ],
            dim=0,
        )
        dv = torch.cat(
            [
                self.B_v_n(self.A_v_n(x[:B])),
                self.B_v_p(self.A_v_p(x[B:])),
            ],
            dim=0,
        )
        return dq, dv


def _attach_lora_to_qkv(block: nn.Module, lora: _TwoBankLoRAQKV) -> None:
    def hook(_module, inputs, output):
        x = inputs[0]  # (B, N, C)
        dq, dv = lora.delta(x)

        q = output[..., : lora.dim]
        k = output[..., lora.dim : -lora.dim]
        v = output[..., -lora.dim :]

        return torch.cat([q + dq, k, v + dv], dim=-1)

    block.attn.qkv.register_forward_hook(hook)


# ---------------------------------------------------------------------
# GeoSAM2 Multimodal Encoder (CORRECT)
# ---------------------------------------------------------------------

class GeoSAM2MultimodalEncoder(nn.Module):
    """
    GeoSAM2-style multimodal SAM2 encoder.

    ✔ single shared SAM2 trunk (run ONCE)
    ✔ two-bank LoRA (normal / point)
    ✔ residual fusion per FPN level
    ✔ identity-safe at init
    ✔ SAM2Base compatible
    """

    def __init__(self, sam_img_encoder: nn.Module, rank: int = 4, use_gradient_checkpointing: bool = False):
        super().__init__()

        self.sam_img_encoder = sam_img_encoder
        _freeze_params(self.sam_img_encoder)
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ---- attach LoRA ----
        self.loras = nn.ModuleList()
        for blk in self.sam_img_encoder.trunk.blocks:
            lora = _TwoBankLoRAQKV(blk.attn.qkv, rank)
            _attach_lora_to_qkv(blk, lora)
            self.loras.append(lora)

        # ---- residual fusion ----
        scalp = int(getattr(self.sam_img_encoder, "scalp", 0))
        num_levels = len(self.sam_img_encoder.trunk.channel_list) - scalp
        d_model = self.sam_img_encoder.neck.d_model

        self.fusion = MultiScaleResidualFusion([d_model] * num_levels)
        for p in self.fusion.parameters():
            p.requires_grad = True

    # ---- SAM2 compatibility ----
    @property
    def trunk(self):
        return self.sam_img_encoder.trunk

    @property
    def neck(self):
        return self.sam_img_encoder.neck

    @torch.no_grad()
    def _set_bank(self, bank: str):
        for l in self.loras:
            l.set_bank(bank)

    # -----------------------------------------------------------------
    # Forward (CRITICAL: single trunk pass)
    # -----------------------------------------------------------------

    def forward(self, normal_img: torch.Tensor, point_img: torch.Tensor):
        B = normal_img.shape[0]

        # 1) concatenate inputs
        x = torch.cat([normal_img, point_img], dim=0)  # (2B, C, H, W)

        # 2) dual-bank LoRA
        self._set_bank("dual")

        # 3) single trunk + neck pass (with gradient checkpointing if enabled)
        if self.training and self.use_gradient_checkpointing:
            # Checkpoint the trunk (memory-intensive backbone) to save memory
            # Wrap in a function for proper checkpointing
            def _run_trunk(x):
                return self.sam_img_encoder.trunk(x)
            trunk_out = torch.utils.checkpoint.checkpoint(_run_trunk, x, use_reentrant=False)
            feats, pos = self.sam_img_encoder.neck(trunk_out)
        else:
            trunk_out = self.sam_img_encoder.trunk(x)
            feats, pos = self.sam_img_encoder.neck(trunk_out)

        # 4) scalp handling
        scalp = int(getattr(self.sam_img_encoder, "scalp", 0))
        if scalp > 0:
            feats = feats[:-scalp]
            pos = pos[:-scalp]

        # 5) split modalities
        feats_n = [f[:B] for f in feats]
        feats_p = [f[B:] for f in feats]
        # Split positional encodings too (they're the same for normal and point, so just take first B)
        pos_split = [p[:B] for p in pos]

        # 6) residual fusion
        fused_feats = self.fusion(feats_n, feats_p)

        return {
            "backbone_fpn": fused_feats,
            "vision_features": fused_feats[-1],
            "vision_pos_enc": pos_split,
        }