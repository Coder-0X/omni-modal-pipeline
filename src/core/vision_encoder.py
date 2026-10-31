"""
Vision Encoder Module for Omni-Modal Pipeline.
Performs visual token extraction, patch embedding projection, and spatial self-attention pooling.
"""

import math
from typing import List, Optional
from .types import VisionInput


class VisionEncoder:
    """
    Vision Encoder modeling a Vision Transformer (ViT) patch embedding
    and spatial multi-head self-attention pooling mechanism.
    """

    def __init__(self, embed_dim: int = 512, patch_size: int = 16, num_heads: int = 8):
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def extract_patch_embeddings(self, vision_input: VisionInput) -> List[List[float]]:
        """
        Converts 2D visual inputs into a sequential grid of patch embeddings.
        N = (H / P) * (W / P)
        """
        num_patches_w = max(1, vision_input.width // self.patch_size)
        num_patches_h = max(1, vision_input.height // self.patch_size)
        num_patches = num_patches_w * num_patches_h

        # Deterministic feature generation or using existing features
        if vision_input.features and len(vision_input.features) == self.embed_dim:
            base_vec = vision_input.features
        else:
            seed = sum(vision_input.image_bytes[:32]) if vision_input.image_bytes else 42
            base_vec = [math.sin(seed + i * 0.1) for i in range(self.embed_dim)]

        patches = []
        for p in range(num_patches):
            patch_token = [
                val * math.cos(p * 0.05 + idx * 0.01)
                for idx, val in enumerate(base_vec)
            ]
            patches.append(patch_token)

        return patches

    def pool_spatial_features(self, patch_tokens: List[List[float]]) -> List[float]:
        """
        Applies normalized attention pooling over visual patch tokens:
        z_v = sum_i alpha_i * v_i
        """
        if not patch_tokens:
            return [0.0] * self.embed_dim

        # Compute dot-product energy against pseudo CLS token query
        query_token = [1.0 / math.sqrt(self.embed_dim)] * self.embed_dim
        energies = []
        for patch in patch_tokens:
            dot = sum(q * p for q, p in zip(query_token, patch))
            energies.append(dot)

        # Softmax normalization
        max_e = max(energies)
        exp_energies = [math.exp(e - max_e) for e in energies]
        sum_exp = sum(exp_energies) or 1.0
        alphas = [e / sum_exp for e in exp_energies]

        # Weighted combination
        pooled = [0.0] * self.embed_dim
        for alpha, patch in zip(alphas, patch_tokens):
            for i in range(self.embed_dim):
                pooled[i] += alpha * patch[i]

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
        return [x / norm for x in pooled]

    def encode(self, vision_input: VisionInput) -> List[float]:
        """End-to-end forward pass: extracts patches and pools into global embedding."""
        patches = self.extract_patch_embeddings(vision_input)
        return self.pool_spatial_features(patches)
