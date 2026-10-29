"""Research architecture for optional image + symptom-text crop disease classification.

This defines the model only. It does not ship pretrained weights, and must not be
used for diagnosis until trained and independently evaluated.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class SubCenterCosineClassifier(nn.Module):
    """Cosine classifier with multiple learned prototypes per class."""

    def __init__(self, embedding_dim: int, num_classes: int, centers_per_class: int = 3, scale: float = 30.0):
        super().__init__()
        if num_classes < 2 or centers_per_class < 1:
            raise ValueError("num_classes must be >= 2 and centers_per_class >= 1")
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes * centers_per_class, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(self.weight, dim=-1)
        cosine = F.linear(embeddings, prototypes)
        cosine = cosine.view(embeddings.shape[0], self.num_classes, self.centers_per_class)
        return cosine.max(dim=-1).values * self.scale


class GatedMultimodalUnit(nn.Module):
    """Learn a sample-wise gate between image and text representations."""

    def __init__(self, dimension: int):
        super().__init__()
        self.gate = nn.Linear(dimension * 2, dimension)
        self.image_transform = nn.Linear(dimension, dimension)
        self.text_transform = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, image: torch.Tensor, text: torch.Tensor, text_present: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([image, text], dim=-1)))
        fused = gate * torch.tanh(self.image_transform(image)) + (1.0 - gate) * torch.tanh(self.text_transform(text))
        # If no text was provided, make the output strictly vision-only.
        fused = torch.where(text_present[:, None], fused, torch.tanh(self.image_transform(image)))
        return self.norm(fused)


class MultimodalCropDiseaseModel(nn.Module):
    """ConvNeXt + Swin image ensemble with optional DeBERTa symptom encoder.

    Args:
        num_classes: number of approved disease/healthy labels (expected 31).
        num_crops: number of supported crop IDs (expected 6).
        pretrained_vision: request timm pretrained weights (download/license/cache availability varies).
        text_model_name: Hugging Face model ID. DeBERTa-v3-large is the default when text is enabled.
        enable_text: set False for a vision-only baseline and inference fallback.
        image_size: nominal input image side; 384 matches the target spec.
    """

    def __init__(
        self,
        num_classes: int = 31,
        num_crops: int = 6,
        pretrained_vision: bool = True,
        enable_text: bool = True,
        text_model_name: str = "microsoft/deberta-v3-large",
        image_size: int = 384,
        embedding_dim: int = 1024,
        centers_per_class: int = 3,
    ):
        super().__init__()
        if num_classes != 31:
            raise ValueError("The approved project specification expects 31 classes; verify the class manifest before changing this.")
        self.enable_text = enable_text
        self.image_size = image_size

        # These are large backbones; full training usually requires a suitable GPU.
        self.convnext = timm.create_model(
            "convnext_large", pretrained=pretrained_vision, num_classes=0, global_pool="avg"
        )
        self.swin = timm.create_model(
            "swin_large_patch4_window12_384", pretrained=pretrained_vision, num_classes=0, global_pool="avg"
        )
        self.convnext_projection = nn.Sequential(
            nn.Linear(self.convnext.num_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
        )
        self.swin_projection = nn.Sequential(
            nn.Linear(self.swin.num_features, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
        )
        self.visual_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU(), nn.Dropout(0.1)
        )

        self.text_encoder: Optional[nn.Module] = None
        self.text_projection: Optional[nn.Module] = None
        self.cross_attention: Optional[nn.Module] = None
        if enable_text:
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            hidden_size = int(self.text_encoder.config.hidden_size)
            self.text_projection = nn.Sequential(
                nn.Linear(hidden_size, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
            )
            self.cross_attention = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.gmu = GatedMultimodalUnit(embedding_dim)
        self.disease_classifier = SubCenterCosineClassifier(
            embedding_dim, num_classes=num_classes, centers_per_class=centers_per_class
        )
        self.crop_classifier = nn.Linear(embedding_dim, num_crops)
        self.embedding_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        conv_features = self.convnext_projection(self.convnext(images))
        swin_features = self.swin_projection(self.swin(images))
        image_embedding = self.visual_fusion(torch.cat([conv_features, swin_features], dim=-1))

        batch_size = images.shape[0]
        text_present = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
        text_embedding = torch.zeros_like(image_embedding)

        if self.enable_text and input_ids is not None and attention_mask is not None:
            text_present = attention_mask.bool().any(dim=1)
            encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            projected_tokens = self.text_projection(encoded)
            # MultiheadAttention returns NaNs for rows whose key mask is entirely true.
            safe_mask = attention_mask.bool().clone()
            empty_rows = ~text_present
            if safe_mask.shape[1] > 0 and empty_rows.any():
                safe_mask[empty_rows, 0] = True
                projected_tokens = projected_tokens.clone()
                projected_tokens[empty_rows] = 0
            attended, _ = self.cross_attention(
                query=image_embedding.unsqueeze(1),
                key=projected_tokens,
                value=projected_tokens,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            text_embedding = attended.squeeze(1)
            text_embedding = torch.where(text_present[:, None], text_embedding, torch.zeros_like(text_embedding))

        fused = self.embedding_norm(self.gmu(image_embedding, text_embedding, text_present))
        return {
            "logits": self.disease_classifier(fused),
            "crop_logits": self.crop_classifier(fused),
            "embedding": F.normalize(fused, dim=-1),
            "vision_embedding": F.normalize(image_embedding, dim=-1),
            "text_present": text_present,
        }
