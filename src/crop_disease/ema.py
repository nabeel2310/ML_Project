"""Exponential moving average (EMA) helper for stabilizing evaluation."""
from __future__ import annotations

import copy
import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            source = model_state[name].detach().to(device=ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source)
