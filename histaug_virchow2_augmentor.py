"""
Minimal standalone wrapper for the pretrained HistAug Virchow2 augmentation model.

Loads the model from HuggingFace and applies feature-space augmentation to
pre-extracted Virchow2 patch embeddings. No image loading required.

Requirements: torch, transformers (pip install transformers)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


def load_histaug_virchow2(device: str | torch.device = "cpu") -> nn.Module:
    """
    Download (or load from cache) the pretrained HistAug Virchow2 model.

    Parameters
    ----------
    device : str or torch.device

    Returns
    -------
    nn.Module
        The HistaugModel in eval mode. Call sample_aug_params() + forward() to use.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "sofieneb/histaug-virchow2", trust_remote_code=True
    )
    return model.to(device).eval()


class HistAugAugmentor:
    """
    Callable wrapper around a loaded HistAug model for use as a drop-in
    training-time augmentor in the MIL training loop.

    Parameters
    ----------
    model : nn.Module
        Loaded HistAug model from load_histaug_virchow2().
    mode : "wsi_wise" or "instance_wise"
        Augmentation mode. Use "wsi_wise" for MIL bag classification.
    aug_prob : float
        Probability of applying augmentation on each call. Default 0.5.
    batch_size : int
        Number of patches to process at once (memory cap for large slides).
    transforms_parameters : dict, optional
        Override specific transform parameters. Keys are transform names
        (brightness, contrast, saturation, hue, powerlaw, hed, rotation,
        crop, h_flip, v_flip, gaussian_blur, erosion, dilation); values are
        [lower, upper] for continuous transforms or a float probability for
        binary/discrete ones. Unspecified transforms keep their model defaults.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: Literal["wsi_wise", "instance_wise"] = "wsi_wise",
        aug_prob: float = 0.5,
        batch_size: int = 4096,
        transforms_parameters: dict | None = None,
    ):
        self.model = model
        self.mode = mode
        self.aug_prob = aug_prob
        self.batch_size = batch_size
        if transforms_parameters:
            self.model.histaug.transforms_parameters.update(transforms_parameters)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Augment a slide's patch features with probability aug_prob.

        Parameters
        ----------
        features : torch.Tensor, shape (N, 2560)
            Patch embeddings for one slide on any device.

        Returns
        -------
        torch.Tensor, shape (N, 2560)
            Augmented (or unchanged) patch embeddings on the same device.
        """
        if torch.rand(1).item() >= self.aug_prob:
            return features
        return augment_slide_features(self.model, features, mode=self.mode, batch_size=self.batch_size)


@torch.inference_mode()
def augment_slide_features(
    model: nn.Module,
    features: torch.Tensor,
    mode: Literal["wsi_wise", "instance_wise"] = "wsi_wise",
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Apply HistAug augmentation to a slide's patch embeddings.

    Parameters
    ----------
    model : nn.Module
        Loaded HistAug model (from load_histaug_virchow2).
    features : torch.Tensor, shape (N, 2560)
        Pre-extracted Virchow2 patch embeddings for one slide.
    mode : "wsi_wise" or "instance_wise"
        "wsi_wise"      — one shared random transformation for the whole slide.
                          Use this for MIL: all patches shift together, preserving
                          the slide-level distribution.
        "instance_wise" — independent random transformation per patch.
                          More diverse augmentation but breaks spatial consistency.
    batch_size : int
        Process this many patches at a time to cap GPU memory use.
        Only relevant for very large slides; the aug_params are sampled once
        for the full slide before batching.

    Returns
    -------
    torch.Tensor, shape (N, 2560)
        Augmented patch embeddings on the same device as features.
    """
    device = next(model.parameters()).device
    n = features.shape[0]

    # Sample aug_params once for the full slide so wsi_wise is consistent across batches.
    aug_params = model.sample_aug_params(n, device=device, mode=mode)

    if n <= batch_size:
        return model(features.to(device), aug_params)

    # Chunk features and the corresponding aug_param slices for large slides.
    out_chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_params = {k: (v[start:end], p[start:end]) for k, (v, p) in aug_params.items()}
        out_chunks.append(model(features[start:end].to(device), chunk_params))
    return torch.cat(out_chunks, dim=0)
