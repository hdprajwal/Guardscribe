import os
import time
import numpy as np
import gradio as gr
import torch
from asr import SpeechToTextPipeline
from detector import ContentDetector

# Initialize models - we do this once at startup to avoid loading them on each request
print("Loading ASR models...")
whisper_pipeline = SpeechToTextPipeline(asr_model="whisper")
wav2vec_pipeline = SpeechToTextPipeline(asr_model="wav2vec", vad_mode=3)
print("ASR models loaded")

print("Loading content detector...")
content_detector = ContentDetector(
    detection_type="custom",
    threshold=0.5,
    custom_categories=["toxic"],
    enable_masking=True,
    masking_method="asterisk",
    use_token_detection=True,
)
print("Content detector loaded")


def process_audio(audio, asr_model="whisper", threshold=0.5, enable_masking=True):
    """
    Process audio input from Gradio.

    Args:
        audio (tuple): Audio data from Gradio (sample_rate, audio_array)
        asr_model (str): ASR model to use ("whisper", "wav2vec")
        threshold (float): Threshold for toxic content detection
        enable_masking (bool): Whether to mask toxic content

    Returns:
        tuple: (transcription, masked_text, detection_info)
    """
    if audio is None:
        return "No audio provided", "No audio provided", "No detection performed"

    sample_rate, audio_array = audio

    # Use pre-loaded pipeline based on selected model
    pipeline = whisper_pipeline if asr_model == "whisper" else wav2vec_pipeline

    # Update detector settings
    content_detector.threshold = threshold
    content_detector.enable_masking = enable_masking

    # Convert audio to the format needed by the ASR pipeline
    audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

    # Transcribe audio
    start_time = time.time()
    transcription = pipeline._transcribe(audio_bytes)
    asr_latency = (time.time() - start_time) * 1000  # ms

    # Detect toxic content
    detection_result = content_detector.detect(transcription)

    # Prepare outputs
    original_text = transcription if transcription else "No speech detected"
    masked_text = detection_result["masked_text"] if detection_result["detected"] else original_text

    # Prepare detection info text
    if detection_result["detected"]:
        detection_info = f"⚠️ Toxic content detected!\nFlags: {detection_result['flags']}\n"
        detection_info += f"Confidence: {detection_result['scores']}\n"
        detection_info += f"ASR Latency: {asr_latency:.2f}ms | Detection Latency: {detection_result.get('latency', 0):.2f}ms"
    else:
        detection_info = f"✓ No toxic content detected\nASR Latency: {asr_latency:.2f}ms"

    return original_text, masked_text, detection_info

# Real-time audio streaming function


def process_stream(audio_chunk, state, asr_model, threshold, enable_masking):
    """
    Process streaming audio input from microphone.

    Args:
        audio_chunk: Latest chunk of audio from the stream
        state: The current state of transcription
        asr_model: ASR model to use
        threshold: Toxic content detection threshold
        enable_masking: Whether to mask toxic content

    Returns:
        tuple: (current_transcript, masked_transcript, detection_info, updated_state)
    """
    if audio_chunk is None:
        return state["transcript"], state["masked"], state["info"], state

    # Process the audio chunk
    sample_rate, audio_array = audio_chunk
    result = process_audio(audio_chunk, asr_model, threshold, enable_masking)

    # Update the state with the new transcription
    if result[0] != "No speech detected" and result[0] != "No audio provided":
        state["transcript"] += " " + result[0]
        state["masked"] += " " + result[1]
        state["info"] = result[2]

    return state["transcript"], state["masked"], state["info"], state


# Create Gradio Interface
with gr.Blocks(title="Speech-to-Text with Toxic Content Filtering") as demo:
    gr.Markdown("# Speech-to-Text with Toxic Content Filtering")
    gr.Markdown("Record or upload audio to transcribe and filter toxic content.")

    with gr.Tabs():
        with gr.TabItem("Single Audio Processing"):
            with gr.Row():
                with gr.Column():
                    # Input Components
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"], type="numpy")

                    with gr.Row():
                        asr_model = gr.Dropdown(
                            choices=["whisper", "wav2vec"],
                            value="whisper",
                            label="ASR Model"
                        )
                        threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.1,
                            label="Detection Threshold"
                        )

                    enable_masking = gr.Checkbox(
                        value=True,
                        label="Enable Content Masking"
                    )

                    process_btn = gr.Button("Transcribe")

                with gr.Column():
                    # Output Components
                    original_text = gr.Textbox(label="Original Transcription")
                    masked_text = gr.Textbox(label="Filtered Transcription")
                    detection_info = gr.Textbox(label="Detection Information")

            # Set up event handlers
            process_btn.click(
                fn=process_audio,
                inputs=[audio_input, asr_model, threshold, enable_masking],
                outputs=[original_text, masked_text, detection_info]
            )

        with gr.TabItem("Real-time Streaming (Experimental)"):
            with gr.Row():
                with gr.Column():
                    # Real-time streaming settings
                    stream_asr_model = gr.Dropdown(
                        choices=["whisper", "wav2vec"],
                        value="whisper",
                        label="ASR Model"
                    )
                    stream_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.1,
                        label="Detection Threshold"
                    )
                    stream_enable_masking = gr.Checkbox(
                        value=True,
                        label="Enable Content Masking"
                    )

                    # Start/stop streaming button
                    stream_state = gr.State(
                        {"transcript": "", "masked": "", "info": ""})

                with gr.Column():
                    # Streaming output
                    stream_original = gr.Textbox(
                        label="Original Transcription (Streaming)")
                    stream_masked = gr.Textbox(
                        label="Filtered Transcription (Streaming)")
                    stream_info = gr.Textbox(label="Detection Information")

                    # Reset button
                    reset_btn = gr.Button("Reset Transcript")

            # Audio streaming component
            stream_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Streaming Microphone Input"
            )

            # Set up streaming event handler
            stream_audio.stream(
                fn=process_stream,
                inputs=[
                    stream_audio,
                    stream_state,
                    stream_asr_model,
                    stream_threshold,
                    stream_enable_masking
                ],
                outputs=[
                    stream_original,
                    stream_masked,
                    stream_info,
                    stream_state
                ]
            )

            # Reset button handler
            reset_btn.click(
                fn=lambda: ({"transcript": "", "masked": "", "info": ""}),
                inputs=[],
                outputs=[stream_state]
            )
            reset_btn.click(
                fn=lambda: ("", "", ""),
                inputs=[],
                outputs=[stream_original, stream_masked, stream_info]
            )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
