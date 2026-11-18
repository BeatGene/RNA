"""Differentiable no-regret/protection penalties; empty selections give finite zero."""
import torch


def safe_mean(value):
    return value.sum() / max(value.numel(), 1)


def no_regret(refined_error, source_error, mask):
    return safe_mean(torch.relu(refined_error[mask] - source_error[mask]))


def protect(prediction, source, atom_mask):
    weights = atom_mask.to(prediction.dtype)
    return ((prediction - source).square().sum(-1) * weights).sum() / weights.sum().clamp_min(1)
