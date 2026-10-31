"""
Interactive Streamlit Dashboard for Omni-Modal Pipeline.
Demonstrates live multimodal perception, cross-attention weight visualization, and latency telemetry.
"""

import asyncio
import streamlit as st
from src.core.orchestrator import OmniModalOrchestrator
from src.core.types import AudioInput, MultimodalContext, TextInput, VisionInput

st.set_page_config(
    page_title="Omni-Modal Perception Pipeline",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Omni-Modal Perception & Cross-Attention Pipeline")
st.markdown(
    "**End-to-End Multimodal Inference Engine** fusing visual tokens, temporal acoustic frames, "
    "and natural language queries via scaled dot-product cross-attention."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Multimodal Stream")
    prompt = st.text_area("Text Prompt / Instruction", "Analyze the visual features and auditory cues in this environment.")
    
    use_vision = st.checkbox("Include Vision Stream (Image / Camera)", value=True)
    if use_vision:
        st.file_uploader("Upload Image (Optional)", type=["png", "jpg", "jpeg"])
        img_res = st.selectbox("Resolution", ["224x224 (Standard ViT)", "384x384 (High-Res)", "512x512"])

    use_audio = st.checkbox("Include Audio Stream (Acoustic / Speech)", value=True)
    if use_audio:
        duration = st.slider("Audio Duration (seconds)", min_value=0.5, max_value=10.0, value=2.5)

    run_btn = st.button("⚡ Execute Pipeline Inference", type="primary")

with col2:
    st.subheader("📊 Cross-Modal Attention & Telemetry")
    if run_btn:
        orchestrator = OmniModalOrchestrator()
        
        ctx = MultimodalContext(
            session_id="web_demo_session",
            text_input=TextInput(prompt=prompt),
            vision_input=VisionInput() if use_vision else None,
            audio_input=AudioInput(duration_seconds=duration if use_audio else 0.0) if use_audio else None
        )

        with st.spinner("Computing Cross-Modal Multi-Head Attention..."):
            output = asyncio.run(orchestrator.execute_pipeline(ctx))

        st.success(f"Inference Completed in **{output.latency_ms} ms** | Confidence: **{output.confidence_score * 100:.1f}%**")

        st.markdown("### 🔍 Scaled Cross-Attention Weights")
        for modality, weight in output.attention_weights.items():
            st.write(f"**{modality.capitalize()} Contribution:**")
            st.progress(weight)

        st.markdown("### 📝 Synthesized Reasoning")
        st.code(output.synthesized_response, language="markdown")

st.markdown("---")
st.caption("Omni-Modal Pipeline — Open-source research artifact by @Coder-0X")
