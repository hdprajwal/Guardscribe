# GuardScribe

## Real-Time Speech-to-Text with Toxic Content Filtering

GuardScribe is a versatile speech-to-text system designed to transcribe live audio and detect potentially harmful content in real-time. The system provides a flexible pipeline with multiple ASR (Automatic Speech Recognition) options and efficient toxic content detection.

## Features

- **Real-time audio transcription** from microphone input
- **Multiple ASR backends** supported:
  - OpenAI Whisper (local processing)
  - Google Speech-to-Text API (cloud-based)
  - HuggingFace Transformers (local processing)
- **Voice Activity Detection (VAD)** to efficiently process speech segments

## Installation

### Prerequisites

- Python 3.8 or higher
- PortAudio (for real-time audio capture)

### System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

### Python Dependencies

```bash
pip install sounddevice numpy torch whisper webrtcvad transformers google-cloud-speech
```