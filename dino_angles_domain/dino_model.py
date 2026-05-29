#!/usr/bin/env python3
# dino_model.py
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


@dataclass
class DinoBackboneBundle:
    model_name: str
    backbone: nn.Module
    feat_dim: int
    config: dict


class DinoHead(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        freeze_backbone: bool = True,
        head: str = "mlp",        # "linear" or "mlp"
        mlp_hidden_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)

        if head == "linear":
            self.head = nn.Linear(feat_dim, num_classes)
        elif head == "mlp":
            hidden_dim = int(feat_dim * mlp_hidden_ratio)
            self.head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head}")

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) tensor, already normalized.
        Returns CLS embedding (B, D).
        """
        out = self.backbone(x)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state          # (B, L, D)
            cls = h[:, 0]                      # CLS token
        elif isinstance(out, (tuple, list)):
            h = out[0]
            cls = h[:, 0]
        else:
            raise RuntimeError(
                "Unexpected backbone output structure for DINOv2 model. "
                "Expected `last_hidden_state` or tuple/list with hidden states."
            )
        return cls

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        logits = self.head(z)
        return logits


def build_dino_backbone(model_name: str) -> DinoBackboneBundle:
    """
    Build backbone using HuggingFace AutoModel + AutoConfig.

    NOTE: We intentionally do NOT use AutoImageProcessor here, because your
    local checkpoint folder is not a modern HF vision checkpoint with
    `preprocessor_config.json`. We rely instead on our own torchvision
    transforms (ColorJitter, RandomAffine, Normalize with IMAGENET stats).
    """
    config = AutoConfig.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)

    feat_dim = getattr(config, "hidden_size", None)
    if feat_dim is None:
        feat_dim = getattr(config, "embed_dim", None)
    if feat_dim is None:
        raise RuntimeError(
            f"Could not infer feature dimension from config for model '{model_name}'. "
            "Expected `hidden_size` or `embed_dim` in config."
        )

    return DinoBackboneBundle(
        model_name=model_name,
        backbone=backbone,
        feat_dim=feat_dim,
        config=config.to_dict(),
    )


# =============================================================================
# Gradient Reversal Layer (for Domain Adversarial Training)
# =============================================================================

class _GRLFunction(torch.autograd.Function):
    """Gradient Reversal Layer: f(x)=x in forward, scales gradient by -lambda in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = float(lambda_val)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_val * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_val: float = 1.0) -> torch.Tensor:
    return _GRLFunction.apply(x, lambda_val)


# =============================================================================
# Angle-conditioned MLP head  (new architecture)
# =============================================================================

class DinoAngleMlpHead(nn.Module):
    """
    DINO backbone + dedicated AngleMLP + shared hidden layer
    + posture head + optional domain adversarial branch with GRL.

    Architecture:
        x  →  backbone  →  CLS  (B, feat_dim)
        angles (B, 4)   →  AngleMLP  →  angle_feat  (B, angle_mlp_dim)

        [CLS ; angle_feat]  →  SharedHidden  →  h  (B, shared_hidden_dim)
                                    ↓
                           posture_head  →  logits  (B, num_classes)
                                    ↓  (if use_domain_adv)
                           GRL(lambda) → domain_head → domain_logits (B, num_domains)

    When use_angles=False, the angle branch is skipped and h = SharedHidden(CLS).

    forward() always returns a tuple:
        (posture_logits, domain_logits_or_None)
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        num_classes: int,
        freeze_backbone: bool = True,
        # angle MLP
        use_angles: bool = True,
        angle_mlp_dim: int = 32,
        # shared hidden layer
        shared_hidden_dim: int = 256,
        dropout: float = 0.0,
        # posture head
        head: str = "mlp",           # "linear" | "mlp"
        mlp_hidden_ratio: float = 1.0,
        # domain adversarial
        use_domain_adv: bool = False,
        num_domains: int = 5,
        grl_lambda: float = 0.1,
        domain_head_type: str = "mlp",  # "linear" | "mlp"
        domain_hidden_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.use_angles = bool(use_angles)
        self.use_domain_adv = bool(use_domain_adv)
        self.grl_lambda = float(grl_lambda)

        # ── AngleMLP ──────────────────────────────────────────────────
        if use_angles:
            self.angle_mlp = nn.Sequential(
                nn.Linear(4, angle_mlp_dim),
                nn.GELU(),
                nn.Linear(angle_mlp_dim, angle_mlp_dim),
            )
            in_dim = feat_dim + angle_mlp_dim
        else:
            self.angle_mlp = None
            in_dim = feat_dim

        # ── Shared hidden layer ────────────────────────────────────────
        self.shared_hidden = nn.Sequential(
            nn.Linear(in_dim, shared_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        # ── Posture head ───────────────────────────────────────────────
        if head == "linear":
            self.posture_head = nn.Linear(shared_hidden_dim, num_classes)
        elif head == "mlp":
            hd = int(shared_hidden_dim * mlp_hidden_ratio)
            self.posture_head = nn.Sequential(
                nn.Linear(shared_hidden_dim, hd),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hd, num_classes),
            )
        else:
            raise ValueError(f"Unknown head type: {head!r}")

        # ── Domain adversarial head ────────────────────────────────────
        if use_domain_adv:
            if domain_head_type == "linear":
                self.domain_head = nn.Linear(shared_hidden_dim, num_domains)
            else:  # mlp
                self.domain_head = nn.Sequential(
                    nn.Linear(shared_hidden_dim, domain_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p=dropout),
                    nn.Linear(domain_hidden_dim, num_domains),
                )
        else:
            self.domain_head = None

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CLS embedding (B, feat_dim)."""
        out = self.backbone(x)
        if hasattr(out, "last_hidden_state"):
            cls = out.last_hidden_state[:, 0]
        elif isinstance(out, (tuple, list)):
            cls = out[0][:, 0]
        else:
            raise RuntimeError("Unexpected backbone output structure.")
        return cls

    def forward(
        self,
        x: torch.Tensor,
        angles: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x      : (B, C, H, W)
        angles : (B, 4) float32  or  None (when use_angles=False or invalid)

        Returns:
            posture_logits : (B, num_classes)
            domain_logits  : (B, num_domains)  or  None
        """
        cls = self.forward_features(x)           # (B, feat_dim)

        if self.use_angles and self.angle_mlp is not None and angles is not None:
            angle_feat = self.angle_mlp(angles.float())
            fused = torch.cat([cls, angle_feat], dim=1)  # (B, feat_dim + angle_mlp_dim)
        else:
            fused = cls

        h = self.shared_hidden(fused)            # (B, shared_hidden_dim)
        posture_logits = self.posture_head(h)    # (B, num_classes)

        domain_logits = None
        if self.use_domain_adv and self.domain_head is not None:
            h_rev = grad_reverse(h, self.grl_lambda)
            domain_logits = self.domain_head(h_rev)  # (B, num_domains)

        return posture_logits, domain_logits


# Keep old names as aliases for backwards compatibility
DinoV2AngleMlpHead = DinoAngleMlpHead
DinoV2AngleHead = DinoAngleMlpHead


def build_angle_head(
    bundle: DinoBackboneBundle,
    num_classes: int,
    num_domains: int = 5,
    use_angles: bool = True,
    angle_mlp_dim: int = 32,
    shared_hidden_dim: int = 256,
    head: str = "linear",
    mlp_hidden_ratio: float = 2.0,
    dropout: float = 0.0,
    freeze_backbone: bool = True,
    use_domain_adv: bool = False,
    grl_lambda: float = 0.1,
    domain_head_type: str = "linear",
    domain_hidden_dim: int = 128,
) -> DinoAngleMlpHead:
    """Factory: build a DinoAngleMlpHead from a backbone bundle."""
    return DinoAngleMlpHead(
        backbone=bundle.backbone,
        feat_dim=bundle.feat_dim,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        use_angles=use_angles,
        angle_mlp_dim=angle_mlp_dim,
        shared_hidden_dim=shared_hidden_dim,
        head=head,
        mlp_hidden_ratio=mlp_hidden_ratio,
        dropout=dropout,
        use_domain_adv=use_domain_adv,
        num_domains=num_domains,
        grl_lambda=grl_lambda,
        domain_head_type=domain_head_type,
        domain_hidden_dim=domain_hidden_dim,
    )

