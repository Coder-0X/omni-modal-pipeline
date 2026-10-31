"""
Cross-Modal Attention Fusion Module for Omni-Modal Pipeline.
Implements scaled dot-product cross-modal multi-head attention and gated residual fusion:
H_fused = gamma_t * H_text + gamma_v * CrossAttn(H_text, H_vision) + gamma_a * CrossAttn(H_text, H_audio)
"""

import math
from typing import Dict, List, Optional, Tuple
from .types import ModalityType


class CrossModalAttentionFusion:
    """
    Scaled Dot-Product Cross-Attention Fusion layer.
    Allows textual queries to cross-attend into visual and auditory key-value representations.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, temperature: float = 1.0):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.scale = 1.0 / math.sqrt(embed_dim // num_heads)

    def _cross_attend(self, query: List[float], keys: List[List[float]], values: List[List[float]]) -> List[float]:
        """
        Computes Scaled Dot-Product Cross-Attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) * V
        """
        if not keys or not values:
            return query

        scores = []
        for key in keys:
            dot = sum(q * k for q, k in zip(query, key))
            scores.append(dot * self.scale / self.temperature)

        # Softmax normalization
        max_s = max(scores)
        exp_s = [math.exp(s - max_s) for s in scores]
        sum_exp = sum(exp_s) or 1.0
        attn_weights = [s / sum_exp for s in exp_s]

        # Weighted combination of values
        context_vec = [0.0] * self.embed_dim
        for weight, val in zip(attn_weights, values):
            for i in range(self.embed_dim):
                context_vec[i] += weight * val[i]

        return context_vec

    def compute_modality_gating(
        self,
        text_emb: List[float],
        vision_emb: Optional[List[float]],
        audio_emb: Optional[List[float]]
    ) -> Dict[str, float]:
        """
        Computes dynamic softmax gating coefficients across active modalities:
        gamma_m = exp(score_m) / sum_j exp(score_j)
        """
        logits = {"text": 1.0}

        if vision_emb:
            # Measure cross-modal cosine alignment
            dot_v = sum(t * v for t, v in zip(text_emb, vision_emb))
            logits["vision"] = max(0.2, 1.0 + dot_v)
        else:
            logits["vision"] = 0.0

        if audio_emb:
            dot_a = sum(t * a for t, a in zip(text_emb, audio_emb))
            logits["audio"] = max(0.2, 0.8 + dot_a)
        else:
            logits["audio"] = 0.0

        # Compute softmax for active modalities
        active_keys = [k for k, v in logits.items() if v > 0.0]
        max_logit = max(logits[k] for k in active_keys)
        exp_vals = {k: math.exp(logits[k] - max_logit) for k in active_keys}
        total_exp = sum(exp_vals.values()) or 1.0

        gating = {k: 0.0 for k in ["text", "vision", "audio"]}
        for k in active_keys:
            gating[k] = exp_vals[k] / total_exp

        return gating

    def fuse(
        self,
        text_emb: List[float],
        vision_emb: Optional[List[float]] = None,
        audio_emb: Optional[List[float]] = None
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Fuses modalities into unified multimodal context representation.
        Returns: (fused_embedding, modality_gating_weights)
        """
        gating = self.compute_modality_gating(text_emb, vision_emb, audio_emb)

        fused = [gating["text"] * t for t in text_emb]

        if vision_emb and gating["vision"] > 0:
            # Cross-attend into vision keys/values
            v_context = self._cross_attend(text_emb, [vision_emb], [vision_emb])
            for i in range(self.embed_dim):
                fused[i] += gating["vision"] * v_context[i]

        if audio_emb and gating["audio"] > 0:
            # Cross-attend into audio keys/values
            a_context = self._cross_attend(text_emb, [audio_emb], [audio_emb])
            for i in range(self.embed_dim):
                fused[i] += gating["audio"] * a_context[i]

        # Residual LayerNorm / L2 Normalization
        norm = math.sqrt(sum(x * x for x in fused)) or 1.0
        normalized_fused = [x / norm for x in fused]

        return normalized_fused, gating
