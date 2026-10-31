"""
Core type definitions for Omni-Modal Pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class ModalityType(str, Enum):
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class VisionInput:
    """Represents raw or preprocessed visual input data."""
    image_bytes: Optional[bytes] = None
    width: int = 224
    height: int = 224
    channels: int = 3
    features: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioInput:
    """Represents temporal audio input or spectrogram frames."""
    sample_rate: int = 16000
    duration_seconds: float = 0.0
    waveform: Optional[List[float]] = None
    spectrogram_frames: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextInput:
    """Represents prompt, query, or instruction tokens."""
    prompt: str = ""
    system_instruction: Optional[str] = None
    token_ids: Optional[List[int]] = None


@dataclass
class MultimodalContext:
    """Composite context carrying multimodal representations across pipeline stages."""
    session_id: str
    text_input: TextInput
    vision_input: Optional[VisionInput] = None
    audio_input: Optional[AudioInput] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionOutput:
    """Output representation resulting from cross-modal attention fusion."""
    session_id: str
    fused_embedding: List[float]
    attention_weights: Dict[str, float]
    synthesized_response: str
    latency_ms: float
    confidence_score: float
    modality_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class StreamToken:
    """Represents an incrementally streamed response token with cross-modal attribution."""
    token: str
    index: int
    is_terminal: bool = False
    source_modality: ModalityType = ModalityType.TEXT
    latency_ms: float = 0.0
