"""Graph means with integer counts and at least FP32 accumulation.

Keep low-precision results in FP32: casting back before graph broadcasting
would also put the backward accumulation back in BF16/FP16.
"""

import torch


def reduction_dtype(src):
    if not src.is_floating_point():
        raise TypeError("graph means require floating-point values")
    return torch.float32 if src.dtype in (torch.float16, torch.bfloat16) else src.dtype


def grouped_mean(src, index, dim=0, dim_size=None):
    """Mean over indexed groups, preserving FP64 and promoting FP16/BF16."""
    if not isinstance(dim, int):
        raise TypeError("dim must be an integer; pass graph membership as batch=")
    if not -src.ndim <= dim < src.ndim:
        raise IndexError("reduction dimension is out of range")
    dim = dim % src.ndim
    index = index.to(device=src.device, dtype=torch.long)
    if index.ndim != 1 or index.numel() != src.size(dim):
        raise ValueError("index must contain one group per reduced element")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    with torch.autocast(device_type=src.device.type, enabled=False):
        values = src.to(reduction_dtype(src)).movedim(dim, 0)
        total = values.new_zeros((dim_size, *values.shape[1:]))
        total.index_add_(0, index, values)
        counts = torch.bincount(index, minlength=dim_size).clamp_min(1)
        denominator = counts.to(values.dtype).reshape(dim_size, *([1] * (values.ndim - 1)))
        return (total / denominator).movedim(0, dim)


def center_by_graph(src, batch=None, dim=0):
    """Remove each graph's mean using an FP32-or-higher projection."""
    with torch.autocast(device_type=src.device.type, enabled=False):
        values = src.to(reduction_dtype(src))
        if batch is None:
            return values - values.mean(dim=dim, keepdim=True)
        batch = batch.to(device=src.device, dtype=torch.long)
        means = grouped_mean(values, batch, dim=dim)
        return values - means.index_select(dim, batch)
