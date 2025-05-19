import os
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
from google.cloud import speech
import webrtcvad
import wave
import tempfile
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2ASR:
    """Speech-to-text transcription using Facebook's Wav2Vec 2.0"""

    def __init__(self, model_name="facebook/wav2vec2-base-960h",
                 device=None):
        """
        Initialize the Wav2Vec 2.0 ASR engine.

        Args:
            model_name: Name or path of the Wav2Vec 2.0 model
            device: Device to use for inference ('cuda' or 'cpu')
        """
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize Wav2Vec2 model and processor
        print(f"Loading Wav2Vec2 model {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)

        print("Wav2Vec2 ASR engine initialized.")

    def _transcribe_array(self, audio_array):
        """
        Transcribe audio data as numpy array.

        Args:
            audio_array: Audio data as numpy array (float32, normalized to [-1.0, 1.0])

        Returns:
            str: Transcribed text
        """
        # Prepare input for the model
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest"
        ).to(self.device)

        # Perform inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # Get predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)

        # Convert ids to text
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription


class SpeechToTextPipeline:
    """
    A flexible Speech-to-Text pipeline that supports multiple ASR models:
    - OpenAI Whisper (local)
    - Google Speech-to-Text API (cloud-based)
    - Facebook Wav2Vec2 (local)
    """

    def __init__(self, asr_model="whisper", sample_rate=16000, frame_duration=30,
                 vad_mode=3, language="en", google_credentials=None):
        """
        Initialize the Speech-to-Text pipeline.

        Args:
            asr_model (str): ASR model to use ("whisper", "google")
            sample_rate (int): Audio sample rate in Hz
            frame_duration (int): Frame duration in milliseconds
            vad_mode (int): Voice Activity Detection aggressiveness (0-3)
            language (str): Language code
            google_credentials (str): Path to Google Cloud credentials file
        """
        self.asr_model = asr_model
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.language = language
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.running = False

        # Initialize Voice Activity Detection
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_size = int(sample_rate * frame_duration / 1000)

        # Initialize ASR model
        if asr_model == "whisper":
            print("Loading Whisper model...")
            self.model = whisper.load_model(
                "small", device="cuda" if torch.cuda.is_available() else "cpu")
            print("Whisper model loaded")
        elif asr_model == "google":
            if google_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials
            self.client = speech.SpeechClient()
            self.config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code=language,
                enable_automatic_punctuation=True,
            )
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=self.config,
                interim_results=True,
            )
        elif asr_model == "wav2vec":
            print("Loading Wav2Vec2 model...")
            self.model = Wav2Vec2ASR(model_name="facebook/wav2vec2-base-960h",
                                     device="cuda" if torch.cuda.is_available() else "cpu")
            print("Wav2Vec2 model loaded")

        else:
            raise ValueError(f"Unsupported ASR model: {asr_model}")

    def start(self):
        """Start the speech-to-text pipeline with audio capture."""
        if self.running:
            return

        self.running = True

        # Start audio capture thread
        self.audio_thread = threading.Thread(target=self._capture_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()

        # Start speech processing thread
        self.processing_thread = threading.Thread(target=self._process_speech)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print(f"Speech-to-Text pipeline started using {self.asr_model} model")

    def stop(self):
        """Stop the speech-to-text pipeline."""
        self.running = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1)
        print("Speech-to-Text pipeline stopped")

    def _capture_audio(self):
        """Capture audio from microphone and add to queue."""
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio capture error: {status}")
            # Add audio frame to queue
            self.audio_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            dtype='int16',
            channels=1,
            callback=callback
        ):
            while self.running:
                time.sleep(0.1)

    def _process_speech(self):
        """Process audio frames from queue and convert to text."""
        buffer = []
        silence_frames = 0

        while self.running:
            # Get audio frame from queue
            try:
                frame = self.audio_queue.get(timeout=1)

                # Check if frame contains speech
                is_speech = self._is_speech(frame)

                if is_speech:
                    buffer.append(frame)
                    silence_frames = 0
                else:
                    silence_frames += 1

                # Process buffer when we detect end of speech
                # (silence for more than 10 frames - about 300ms)
                if buffer and silence_frames > 10:
                    audio_data = b''.join(buffer)

                    # Measure transcription time for latency evaluation
                    start_time = time.time()

                    text = self._transcribe(audio_data)

                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to ms

                    if text:
                        print(
                            f"Transcribed: '{text}' (Latency: {latency:.2f}ms)")
                        self.text_queue.put((text, latency))

                    buffer = []
                    silence_frames = 0

            except queue.Empty:
                continue

    def _is_speech(self, frame):
        """Detect if a frame contains speech using Voice Activity Detection."""
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except:
            return False

    def _transcribe(self, audio_data):
        """
        Transcribe audio data to text using the selected ASR model.

        Args:
            audio_data (bytes): Raw audio data

        Returns:
            str: Transcribed text
        """
        if self.asr_model == "whisper":
            # Convert bytes to numpy array for Whisper
            audio_array = np.frombuffer(
                audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe with Whisper
            result = self.model.transcribe(audio_array, language=self.language)
            return result["text"].strip()

        elif self.asr_model == "google":
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)

                # Read the file for Google API
                with open(temp_file.name, "rb") as audio_file:
                    content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            response = self.client.recognize(config=self.config, audio=audio)

            if not response.results:
                return ""

            return response.results[0].alternatives[0].transcript.strip()
        elif self.asr_model == "wav2vec":
            # Convert bytes to numpy array for Wav2Vec2
            audio_array = np.frombuffer(
                audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe with Wav2Vec2
            result = self.model._transcribe_array(audio_array)
            return result
        return ""

    def get_transcription(self):
        """
        Get the latest transcription from the queue.

        Returns:
            tuple: (text, latency) or (None, None) if queue is empty
        """
        try:
            return self.text_queue.get(block=False)
        except queue.Empty:
            return None, None


def main():
    # Create pipeline with Whisper model
    pipeline = SpeechToTextPipeline(asr_model="whisper")
    try:
        pipeline.start()

        print("Listening for speech... (Press Ctrl+C to stop)")

        # Process transcriptions as they come in
        while True:
            text, latency = pipeline.get_transcription()
            if text and latency:
                print("Transcription: ", text)
                print("Latency: ", latency)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()


def benchmark_asr_models(audio_file, models=["whisper", "wav2vec"]):
    """
    Benchmark different ASR models on the same audio file.

    Args:
        audio_file (str): Path to audio file
        models (list): List of ASR models to benchmark
    """
    results = {}

    for model in models:
        print(f"\nBenchmarking {model}...")

        if model == "google":
            pipeline = SpeechToTextPipeline(
                asr_model=model, google_credentials="google-credentials.json")
        else:
            pipeline = SpeechToTextPipeline(asr_model=model)
        # Read audio file
        with wave.open(audio_file, 'rb') as wf:
            audio_data = wf.readframes(wf.getnframes())

        # Measure transcription time
        start_time = time.time()
        text = pipeline._transcribe(audio_data)
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to ms

        results[model] = {
            "text": text,
            "latency": latency
        }

        print(f"Model: {model}")
        print(f"Text: {text}")
        print(f"Latency: {latency:.2f}ms")

    return results


if __name__ == "__main__":
    main()
