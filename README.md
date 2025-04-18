# GuardScribe

## Real-Time Speech-to-Text with Toxic Content Detection and Filtering

GuardScribe is a versatile speech-to-text system designed to transcribe live audio and detect potentially harmful content in real-time. The system provides a flexible pipeline with multiple ASR (Automatic Speech Recognition) options and efficient toxic content detection with automated content masking capabilities.

## Features

### Speech Recognition

- **Real-time audio transcription** from microphone input
- **Multiple ASR backends** supported:
  - OpenAI Whisper (local processing)
  - Google Speech-to-Text API (cloud-based)
  - HuggingFace Transformers (local processing with Whisper large-v3-turbo)
- **Voice Activity Detection (VAD)** to efficiently process speech segments
- **Performance metrics** including latency tracking for ASR benchmarking

### Content Filtering

- **Real-time toxic content detection** with configurable sensitivity thresholds
- **Multiple detection methods**:
  - Toxicity classification using pre-trained models
  - Custom category detection for flexible content filtering
  - Token-level detection for precise content masking
- **Automatic content masking** with multiple masking methods:
  - Asterisk masking (replacing characters with "*")
  - Redaction (replacing with "[REDACTED]")
  - Placeholder insertion (replacing with "[...]")
- **Model training tools** for building custom content detection models
- **Custom DistilBERT-based models**:
  - Text classification model based on DistilBERT for toxicity detection
  - Token classification model based on DistilBERT for precise toxic span detection

## System Architecture

GuardScribe is comprised of several components:

- `asr.py`: Core speech-to-text pipeline with multiple ASR model options
- `detector.py`: Content detection and filtering module with masking capabilities
- `app.py`: Application that integrates speech recognition with content detection
- `text_clf_model.py`: Training pipeline for DistilBERT-based text classification models
- `token_clf_model.py`: Training pipeline for DistilBERT-based token-level classification

The system uses DistilBERT as the foundation for its content detection models:

- The text classification model is fine-tuned from `distilbert-base-uncased` for detecting toxic content at the sentence level
- The token classification model is fine-tuned from the same base architecture for precise character-level detection of toxic spans

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (ideally with CUDA support for faster processing)
- PortAudio (for real-time audio capture)

### System Dependencies

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Python Dependencies

```bash
pip install sounddevice numpy torch torchaudio openai-whisper webrtcvad transformers google-cloud-speech pandas matplotlib tqdm scikit-learn
```

## Usage

### Basic Usage

Run the application to start real-time speech transcription with content filtering:

```bash
python app.py
```

### Custom Configuration

To customize the ASR model and content detection settings:

```python
from asr import SpeechToTextPipeline
from detector import ContentDetector

# Initialize ASR pipeline with Whisper
pipeline = SpeechToTextPipeline(
    asr_model="whisper",  # Options: "whisper", "google", "transformers"
    sample_rate=16000,
    frame_duration=30,
    vad_mode=3
)

# Initialize content detector
detector = ContentDetector(
    detection_type="custom",  # Options: "toxicity", "custom"
    threshold=0.5,            # Confidence threshold (0.0-1.0)
    custom_categories=["toxic"],
    enable_masking=True,
    masking_method="asterisk",  # Options: "asterisk", "redact", "placeholder"
    use_token_detection=True    # Enable more precise token-level detection
)

# Start the pipeline
pipeline.start()

# Process transcriptions
while True:
    text, latency = pipeline.get_transcription()
    if text:
        result = detector.detect(text)
        if result["detected"]:
            display_text = result["masked_text"]
        else:
            display_text = text
        print(f"Transcribed: {display_text}")
```

### Training Custom Detection Models

The system includes training pipelines for two types of DistilBERT-based models:

#### Text Classification Model

```bash
python text_clf_model.py
```

This trains a DistilBERT-based model for sentence-level toxicity classification:

- Uses the Jigsaw dataset for training
- Fine-tunes DistilBERT for binary classification (toxic vs. non-toxic)
- Includes evaluation metrics and model visualization
- Saves the trained model to `toxic_content_detector_model`

#### Token Classification Model

```bash
python token_clf_model.py
```

This trains a DistilBERT-based model for token-level toxic span detection:

- Uses specialized datasets with character-level toxic span annotations
- Fine-tunes DistilBERT for BIO tagging (Beginning, Inside, Outside of toxic spans)
- Provides span-level F1 score evaluation
- Saves the trained model to `toxic_span_distilbert`

## Benchmarking

The system includes built-in benchmarking capabilities for comparing different ASR models:

```python
from asr import benchmark_asr_models

# Compare performance of different ASR models
results = benchmark_asr_models(
    audio_file="path/to/audio.wav",
    models=["whisper", "google", "transformers"]
)
```
