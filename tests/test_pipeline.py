"""
Unit and Integration Test Suite for Omni-Modal Pipeline.
Validates encoder outputs, cross-modal attention math, orchestrator execution, and eval metrics.
"""

import asyncio
import pytest
from src.core.types import (
    AudioInput,
    MultimodalContext,
    TextInput,
    VisionInput,
)
from src.core.vision_encoder import VisionEncoder
from src.core.audio_processor import AudioProcessor
from src.core.fusion import CrossModalAttentionFusion
from src.core.orchestrator import OmniModalOrchestrator
from src.evals.metrics import MultimodalEvaluator


def test_vision_encoder():
    """Validates patch extraction and attention pooling normalization."""
    encoder = VisionEncoder(embed_dim=256, patch_size=16)
    v_in = VisionInput(width=64, height=64)
    patches = encoder.extract_patch_embeddings(v_in)
    assert len(patches) == 16  # (64/16) * (64/16)
    assert len(patches[0]) == 256

    pooled = encoder.encode(v_in)
    assert len(pooled) == 256
    # Verify unit norm
    norm = sum(x * x for x in pooled) ** 0.5
    assert pytest.approx(norm, 0.01) == 1.0


def test_audio_processor():
    """Validates acoustic frame extraction and VAD energy calculation."""
    proc = AudioProcessor(embed_dim=256, n_mels=40)
    a_in = AudioInput(duration_seconds=1.0)
    frames = proc.extract_temporal_features(a_in)
    assert len(frames) > 0
    assert len(frames[0]) == 40

    encoded = proc.encode(a_in)
    assert len(encoded) == 256
    norm = sum(x * x for x in encoded) ** 0.5
    assert pytest.approx(norm, 0.01) == 1.0

    # Test VAD
    is_voiced, db = proc.compute_energy_vad([0.1, 0.2, -0.15, 0.3])
    assert is_voiced is True
    assert db > -50.0


def test_cross_modal_fusion():
    """Validates dynamic modality gating and residual cross-attention fusion."""
    fusion = CrossModalAttentionFusion(embed_dim=128)
    text_emb = [0.1] * 128
    vision_emb = [0.2] * 128
    audio_emb = [0.05] * 128

    fused, weights = fusion.fuse(text_emb, vision_emb, audio_emb)
    assert len(fused) == 128
    assert "text" in weights and "vision" in weights and "audio" in weights
    assert pytest.approx(sum(weights.values()), 0.01) == 1.0


@pytest.mark.asyncio
async def test_orchestrator_execution():
    """Validates asynchronous pipeline execution and token streaming."""
    orchestrator = OmniModalOrchestrator(embed_dim=128)
    ctx = MultimodalContext(
        session_id="test_sess_01",
        text_input=TextInput(prompt="Detect anomalous events in multimodal feed."),
        vision_input=VisionInput(width=64, height=64),
        audio_input=AudioInput(duration_seconds=1.0)
    )

    output = await orchestrator.execute_pipeline(ctx)
    assert output.session_id == "test_sess_01"
    assert len(output.fused_embedding) == 128
    assert output.latency_ms > 0
    assert output.confidence_score > 0.8

    # Test token streaming
    tokens = []
    async for stream_token in orchestrator.stream_inference(ctx):
        tokens.append(stream_token.token)
    assert len(tokens) > 5


def test_evaluation_metrics():
    """Validates Recall@K and BLEU evaluation metrics."""
    # Cosine similarity
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    sim = MultimodalEvaluator.cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, 0.001) == 1.0

    # Recall@K
    queries = [[1.0, 0.0], [0.0, 1.0]]
    targets = [[1.0, 0.0], [0.0, 1.0]]
    gt = [0, 1]
    recalls = MultimodalEvaluator.compute_recall_at_k(queries, targets, gt, k_values=(1, 2))
    assert recalls["R@1"] == 1.0

    # BLEU
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumps over the lazy dog"
    bleu = MultimodalEvaluator.compute_bleu_score(ref, hyp)
    assert pytest.approx(bleu, 0.01) == 1.0
