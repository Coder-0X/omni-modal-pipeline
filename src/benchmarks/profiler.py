"""
Performance Profiling and Benchmarking Suite for Omni-Modal Pipeline.
Evaluates end-to-end latency percentiles (P50/P90/P95/P99), throughput, and memory scaling.
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List
from src.core.orchestrator import OmniModalOrchestrator
from src.core.types import MultimodalContext, TextInput, VisionInput, AudioInput


class PipelineProfiler:
    """
    Automated benchmark harness evaluating latency distribution and throughput under load.
    """

    def __init__(self, orchestrator: OmniModalOrchestrator):
        self.orchestrator = orchestrator

    async def benchmark_latency(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Executes sequential iterations to compute empirical latency percentiles.
        """
        latencies_ms: List[float] = []

        for i in range(num_samples):
            ctx = MultimodalContext(
                session_id=f"bench_{i}",
                text_input=TextInput(prompt=f"Benchmark multimodal query sample {i}"),
                vision_input=VisionInput(width=224, height=224),
                audio_input=AudioInput(duration_seconds=2.0)
            )

            start = time.perf_counter()
            _ = await self.orchestrator.execute_pipeline(ctx)
            elapsed = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed)

        latencies_ms.sort()
        p50 = latencies_ms[int(num_samples * 0.50)]
        p90 = latencies_ms[int(num_samples * 0.90)]
        p95 = latencies_ms[int(num_samples * 0.95)]
        p99 = latencies_ms[int(num_samples * 0.99)] if num_samples >= 100 else latencies_ms[-1]

        return {
            "total_samples": num_samples,
            "mean_latency_ms": round(statistics.mean(latencies_ms), 2),
            "stdev_ms": round(statistics.stdev(latencies_ms) if num_samples > 1 else 0.0, 2),
            "p50_ms": round(p50, 2),
            "p90_ms": round(p90, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "min_ms": round(min(latencies_ms), 2),
            "max_ms": round(max(latencies_ms), 2),
            "throughput_qps": round(1000.0 / statistics.mean(latencies_ms), 2)
        }


if __name__ == "__main__":
    async def main():
        orchestrator = OmniModalOrchestrator()
        profiler = PipelineProfiler(orchestrator)
        print("[*] Running Omni-Modal Pipeline Latency Benchmark...")
        results = await profiler.benchmark_latency(num_samples=30)
        for k, v in results.items():
            print(f"  - {k}: {v}")

    asyncio.run(main())
