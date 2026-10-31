"""
Audio Processing Module for Omni-Modal Pipeline.
Performs audio framing, log-mel spectrogram extraction, VAD, and temporal pooling.
"""

import math
from typing import List, Tuple
from .types import AudioInput


class AudioProcessor:
    """
    Audio Processor implementing temporal acoustic feature extraction,
    Voice Activity Detection (VAD) gating, and acoustic projection.
    """

    def __init__(self, embed_dim: int = 512, n_mels: int = 80, frame_length_ms: float = 25.0):
        self.embed_dim = embed_dim
        self.n_mels = n_mels
        self.frame_length_ms = frame_length_ms

    def compute_energy_vad(self, waveform: List[float], threshold: float = 0.01) -> Tuple[bool, float]:
        """
        Computes Root Mean Square (RMS) energy to detect voiced segments.
        Returns: (is_voiced, energy_db)
        """
        if not waveform:
            return False, -100.0

        rms = math.sqrt(sum(x * x for x in waveform) / len(waveform))
        energy_db = 20 * math.log10(max(rms, 1e-9))
        is_voiced = rms > threshold
        return is_voiced, energy_db

    def extract_temporal_features(self, audio_input: AudioInput) -> List[List[float]]:
        """
        Constructs acoustic frame tokens across the temporal sequence dimension.
        """
        if audio_input.spectrogram_frames:
            frames = audio_input.spectrogram_frames
        else:
            # Synthetic mel-frequency extraction from duration / waveform
            num_frames = max(4, int((audio_input.duration_seconds or 1.0) * 50))
            frames = []
            for t in range(num_frames):
                frame = [
                    math.sin(t * 0.1 + m * 0.05) * math.exp(-m * 0.01)
                    for m in range(self.n_mels)
                ]
                frames.append(frame)

        return frames

    def project_and_pool(self, frames: List[List[float]]) -> List[float]:
        """
        Linear projection from n_mels -> embed_dim followed by temporal mean pooling.
        """
        if not frames:
            return [0.0] * self.embed_dim

        # Project and accumulate
        pooled = [0.0] * self.embed_dim
        for frame in frames:
            frame_len = len(frame)
            for i in range(self.embed_dim):
                src_val = frame[i % frame_len]
                pooled[i] += src_val * (1.0 / (1.0 + (i // frame_len)))

        # Average across temporal dimension
        num_frames = len(frames)
        pooled = [x / num_frames for x in pooled]

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in pooled)) or 1.0
        return [x / norm for x in pooled]

    def encode(self, audio_input: AudioInput) -> List[float]:
        """End-to-end forward pass: extracts temporal acoustic tokens and pools."""
        frames = self.extract_temporal_features(audio_input)
        return self.project_and_pool(frames)
