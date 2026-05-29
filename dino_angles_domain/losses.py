# losses.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", weight=None):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is not None:
            if torch.is_tensor(alpha):
                alpha = alpha.detach().clone().float()
            else:
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, target):
        target = target.long().to(logits.device)

        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()

        idx = target.view(-1, 1)
        logp_t = logp.gather(1, idx).squeeze(1)
        p_t = p.gather(1, idx).squeeze(1)

        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = -focal_factor * logp_t

        sample_weight = None
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)

            if alpha.ndim == 0:
                sample_weight = alpha.expand_as(target).float()
            else:
                sample_weight = alpha.gather(0, target)

            loss = loss * sample_weight

        if self.reduction == "mean":
            if sample_weight is not None and self.alpha.ndim > 0:
                return loss.sum() / (sample_weight.sum() + 1e-12)
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# from __future__ import annotations
# import numpy as np
# import torch
# import torch.nn as nn


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=1.0, weight=None, reduction="mean"):
#         super().__init__()
#         self.gamma = float(gamma)
#         self.alpha = float(alpha)
#         self.register_buffer("weight", weight if weight is not None else None)
#         self.reduction = reduction

#     def forward(self, logits, target):
#         logp = torch.log_softmax(logits, dim=1)
#         p = torch.softmax(logits, dim=1)

#         idx = target.view(-1, 1)
#         logp_t = logp.gather(1, idx).squeeze(1)
#         p_t = p.gather(1, idx).squeeze(1)

#         focal = (1.0 - p_t).clamp(min=0) ** self.gamma
#         loss = -self.alpha * focal * logp_t

#         if self.weight is not None:
#             w = self.weight.gather(0, target)
#             loss = loss * w

#         if self.reduction == "mean":
#             return loss.mean()
#         if self.reduction == "sum":
#             return loss.sum()
#         return loss


def build_class_weight_tensor(freq: np.ndarray, scheme: str, device) -> torch.Tensor | None:
    """
    scheme:
      - none
      - inv_freq:     w = N / n_c
      - inv_freq_norm w = (N / n_c) normalized to mean=1
    """
    scheme = str(scheme)
    if scheme == "none":
        return None
    denom = np.clip(freq, 1.0, None)
    N = float(freq.sum())
    w = (N / denom)
    if scheme == "inv_freq_norm":
        w = w / (w.mean() + 1e-9)
    return torch.tensor(w, dtype=torch.float32, device=device)


# def build_criterion(
#     loss_name: str,
#     label_smooth: float,
#     class_weight_t: torch.Tensor | None,
#     focal_alpha: float,
#     focal_gamma: float,
# ):
#     loss_name = str(loss_name)
#     if loss_name == "ce":
#         return nn.CrossEntropyLoss(weight=class_weight_t, label_smoothing=float(label_smooth))
#     if loss_name == "focal":
#         return FocalLoss(gamma=focal_gamma, alpha=focal_alpha, weight=class_weight_t)
#     raise ValueError(f"Unknown loss: {loss_name}")

def build_criterion(
    loss_name: str,
    label_smooth: float,
    class_weight_t: torch.Tensor | None,
    focal_alpha: float,
    focal_gamma: float,
):
    loss_name = str(loss_name)
    if loss_name == "ce":
        return nn.CrossEntropyLoss(
            weight=class_weight_t,
            label_smoothing=float(label_smooth)
        )
    if loss_name == "focal":
        alpha = class_weight_t if class_weight_t is not None else focal_alpha
        return FocalLoss(
            gamma=focal_gamma,
            alpha=alpha,
            reduction="mean",
        )
    raise ValueError(f"Unknown loss: {loss_name}")