"""Flat parameter-vector helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ParamLayout:
    shapes: list[torch.Size]
    sizes: list[int]


def get_param_layout(model: nn.Module) -> ParamLayout:
    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    return ParamLayout(shapes=shapes, sizes=sizes)


def flatten_params(model: nn.Module) -> torch.Tensor:
    parts = [p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]
    return torch.cat(parts)


def set_params_from_flat(model: nn.Module, theta: torch.Tensor, layout: ParamLayout) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if theta.ndim != 1:
        raise ValueError("theta must be 1D")

    expected = sum(layout.sizes)
    if theta.numel() != expected:
        raise ValueError(f"theta length mismatch: got {theta.numel()}, expected {expected}")

    idx = 0
    with torch.no_grad():
        for p, shape, size in zip(params, layout.shapes, layout.sizes):
            chunk = theta[idx: idx + size].view(shape)
            p.copy_(chunk)
            idx += size


def task_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = model(x)
    return torch.mean((pred - y) ** 2)
