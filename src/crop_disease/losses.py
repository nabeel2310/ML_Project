"""Loss helpers for the multimodal training pipeline."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss; anchors without positive peers contribute zero."""
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Expected embeddings [batch, dim] and labels [batch].")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.sum() * 0.0

    features = F.normalize(embeddings.float(), dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(logits.dtype)
    self_mask = torch.eye(batch_size, device=embeddings.device, dtype=logits.dtype)
    positive_mask = positive_mask * (1.0 - self_mask)
    denominator_mask = 1.0 - self_mask
    exp_logits = torch.exp(logits) * denominator_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1)
    mean_positive_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -mean_positive_log_prob[valid].mean()


def combined_training_loss(
    outputs: dict[str, torch.Tensor],
    disease_labels: torch.Tensor,
    crop_labels: torch.Tensor,
    *,
    label_smoothing: float = 0.05,
    crop_loss_weight: float = 0.2,
    contrastive_loss_weight: float = 0.1,
    contrastive_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute disease, crop auxiliary, and supervised contrastive losses."""
    disease = F.cross_entropy(outputs["logits"], disease_labels, label_smoothing=label_smoothing)
    crop = F.cross_entropy(outputs["crop_logits"], crop_labels)
    contrastive = supervised_contrastive_loss(
        outputs["embedding"], disease_labels, temperature=contrastive_temperature
    )
    total = disease + crop_loss_weight * crop + contrastive_loss_weight * contrastive
    return total, {"disease": disease.detach(), "crop": crop.detach(), "contrastive": contrastive.detach()}
