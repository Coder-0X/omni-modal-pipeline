"""
Zero-dependency test runner using Python standard library unittest.
"""

import asyncio
import unittest
import math
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.types import AudioInput, MultimodalContext, TextInput, VisionInput
from src.core.vision_encoder import VisionEncoder
from src.core.audio_processor import AudioProcessor
from src.core.fusion import CrossModalAttentionFusion
from src.core.orchestrator import OmniModalOrchestrator
from src.evals.metrics import MultimodalEvaluator


class TestOmniModalPipeline(unittest.TestCase):

    def test_vision_encoder(self):
        encoder = VisionEncoder(embed_dim=256, patch_size=16)
        v_in = VisionInput(width=64, height=64)
        patches = encoder.extract_patch_embeddings(v_in)
        self.assertEqual(len(patches), 16)
        self.assertEqual(len(patches[0]), 256)

        pooled = encoder.encode(v_in)
        self.assertEqual(len(pooled), 256)
        norm = math.sqrt(sum(x * x for x in pooled))
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_audio_processor(self):
        proc = AudioProcessor(embed_dim=256, n_mels=40)
        a_in = AudioInput(duration_seconds=1.0)
        frames = proc.extract_temporal_features(a_in)
        self.assertGreater(len(frames), 0)
        self.assertEqual(len(frames[0]), 40)

        encoded = proc.encode(a_in)
        self.assertEqual(len(encoded), 256)
        norm = math.sqrt(sum(x * x for x in encoded))
        self.assertAlmostEqual(norm, 1.0, places=2)

        is_voiced, db = proc.compute_energy_vad([0.1, 0.2, -0.15, 0.3])
        self.assertTrue(is_voiced)
        self.assertGreater(db, -50.0)

    def test_cross_modal_fusion(self):
        fusion = CrossModalAttentionFusion(embed_dim=128)
        text_emb = [0.1] * 128
        vision_emb = [0.2] * 128
        audio_emb = [0.05] * 128

        fused, weights = fusion.fuse(text_emb, vision_emb, audio_emb)
        self.assertEqual(len(fused), 128)
        self.assertTrue("text" in weights and "vision" in weights and "audio" in weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=2)

    def test_orchestrator_execution(self):
        orchestrator = OmniModalOrchestrator(embed_dim=128)
        ctx = MultimodalContext(
            session_id="test_sess_01",
            text_input=TextInput(prompt="Detect anomalous events in multimodal feed."),
            vision_input=VisionInput(width=64, height=64),
            audio_input=AudioInput(duration_seconds=1.0)
        )

        output = asyncio.run(orchestrator.execute_pipeline(ctx))
        self.assertEqual(output.session_id, "test_sess_01")
        self.assertEqual(len(output.fused_embedding), 128)
        self.assertGreater(output.latency_ms, 0)
        self.assertGreater(output.confidence_score, 0.8)

    def test_evaluation_metrics(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        sim = MultimodalEvaluator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 1.0, places=3)

        queries = [[1.0, 0.0], [0.0, 1.0]]
        targets = [[1.0, 0.0], [0.0, 1.0]]
        gt = [0, 1]
        recalls = MultimodalEvaluator.compute_recall_at_k(queries, targets, gt, k_values=(1, 2))
        self.assertEqual(recalls["R@1"], 1.0)

        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quick brown fox jumps over the lazy dog"
        bleu = MultimodalEvaluator.compute_bleu_score(ref, hyp)
        self.assertAlmostEqual(bleu, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
