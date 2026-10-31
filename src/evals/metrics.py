"""
Evaluation and Metrics Suite for Omni-Modal Pipeline.
Implements Cross-Modal Retrieval (Recall@K), Semantic Alignment Scoring, and BLEU-N evaluation.
"""

import math
from typing import Dict, List, Tuple


class MultimodalEvaluator:
    """
    Evaluator computing empirical performance metrics across multimodal inference tasks.
    """

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Computes cosine similarity between two normalized vectors."""
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1)) or 1.0
        norm2 = math.sqrt(sum(b * b for b in v2)) or 1.0
        return max(-1.0, min(1.0, dot / (norm1 * norm2)))

    @classmethod
    def compute_recall_at_k(
        cls,
        query_embeddings: List[List[float]],
        target_embeddings: List[List[float]],
        ground_truth_indices: List[int],
        k_values: Tuple[int, ...] = (1, 5, 10)
    ) -> Dict[str, float]:
        """
        Computes Recall@K metrics for Cross-Modal Retrieval (e.g. Text-to-Vision or Audio-to-Text).
        """
        total_queries = len(query_embeddings)
        if total_queries == 0 or not target_embeddings:
            return {f"R@{k}": 0.0 for k in k_values}

        hits = {k: 0 for k in k_values}

        for q_idx, (q_vec, target_idx) in enumerate(zip(query_embeddings, ground_truth_indices)):
            # Compute similarities against all target representations
            scores = [
                (cls.cosine_similarity(q_vec, t_vec), idx)
                for idx, t_vec in enumerate(target_embeddings)
            ]
            # Rank in descending order of similarity
            scores.sort(key=lambda x: x[0], reverse=True)
            top_indices = [idx for _, idx in scores]

            for k in k_values:
                if target_idx in top_indices[:k]:
                    hits[k] += 1

        return {f"R@{k}": round(hits[k] / total_queries, 4) for k in k_values}

    @staticmethod
    def compute_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
        """
        Computes modified n-gram precision BLEU score with brevity penalty.
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not hyp_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            if len(hyp_tokens) < n or len(ref_tokens) < n:
                precisions.append(1e-9)
                continue

            ref_ngrams = {}
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

            hyp_ngrams = {}
            for i in range(len(hyp_tokens) - n + 1):
                ngram = tuple(hyp_tokens[i:i+n])
                hyp_ngrams[ngram] = hyp_ngrams.get(ngram, 0) + 1

            overlap = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in hyp_ngrams.items())
            total_hyp_ngrams = len(hyp_tokens) - n + 1
            precisions.append((overlap + 1e-9) / (total_hyp_ngrams + 1e-9))

        # Geometric mean
        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        geom_mean = math.exp(log_avg)

        # Brevity penalty
        bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - (len(ref_tokens) / len(hyp_tokens)))
        return round(bp * geom_mean, 4)
