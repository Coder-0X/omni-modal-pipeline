"""
REST & WebSocket API Server for Omni-Modal Pipeline.
Powered by FastAPI with asynchronous streaming and health telemetry.
"""

from typing import Dict, List, Optional
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from src.core.orchestrator import OmniModalOrchestrator
from src.core.types import AudioInput, MultimodalContext, TextInput, VisionInput

app = FastAPI(
    title="Omni-Modal Perception Pipeline API",
    description="High-throughput cross-modal attention inference and streaming microservice.",
    version="1.0.0"
)

orchestrator = OmniModalOrchestrator()


class InferenceRequest(BaseModel):
    prompt: str = Field(..., example="Describe the spatial arrangement of objects in the scene.")
    has_image: bool = Field(default=False)
    image_width: int = Field(default=224)
    image_height: int = Field(default=224)
    has_audio: bool = Field(default=False)
    audio_duration_seconds: float = Field(default=0.0)


class InferenceResponse(BaseModel):
    session_id: str
    synthesized_response: str
    confidence_score: float
    latency_ms: float
    attention_weights: Dict[str, float]


@app.get("/health")
async def health_check():
    """System health and readiness probe."""
    return {
        "status": "healthy",
        "service": "omni-modal-pipeline",
        "embed_dim": orchestrator.embed_dim,
        "active_modalities": ["vision", "audio", "text"]
    }


@app.post("/v1/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Executes single-shot multimodal forward inference."""
    try:
        session_id = str(uuid.uuid4())
        v_input = VisionInput(width=request.image_width, height=request.image_height) if request.has_image else None
        a_input = AudioInput(duration_seconds=request.audio_duration_seconds) if request.has_audio else None

        ctx = MultimodalContext(
            session_id=session_id,
            text_input=TextInput(prompt=request.prompt),
            vision_input=v_input,
            audio_input=a_input
        )

        output = await orchestrator.execute_pipeline(ctx)

        return InferenceResponse(
            session_id=output.session_id,
            synthesized_response=output.synthesized_response,
            confidence_score=output.confidence_score,
            latency_ms=output.latency_ms,
            attention_weights=output.attention_weights
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/v1/stream")
async def stream_endpoint(websocket: WebSocket):
    """Real-time bi-directional WebSocket streaming multimodal tokens."""
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "Analyze scene")
        has_vision = data.get("has_vision", False)
        has_audio = data.get("has_audio", False)

        ctx = MultimodalContext(
            session_id=str(uuid.uuid4()),
            text_input=TextInput(prompt=prompt),
            vision_input=VisionInput() if has_vision else None,
            audio_input=AudioInput() if has_audio else None
        )

        async for stream_token in orchestrator.stream_inference(ctx):
            await websocket.send_json({
                "token": stream_token.token,
                "index": stream_token.index,
                "is_terminal": stream_token.is_terminal,
                "latency_ms": stream_token.latency_ms
            })
    except WebSocketDisconnect:
        pass
