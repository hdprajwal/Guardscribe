import time
from asr import SpeechToTextPipeline
from detector import ContentDetector


def main():
    # Create pipeline with Whisper model
    pipeline = SpeechToTextPipeline(asr_model="whisper")
    content_detector = ContentDetector(
        detection_type="custom",
        threshold=0.5,
        custom_categories=["toxic"],
        enable_masking=True,
        masking_method="astrisk",
        use_token_detection=True,
    )
    try:
        pipeline.start()

        print("Listening for speech... (Press Ctrl+C to stop)")

        # Process transcriptions as they come in
        while True:
            text, latency = pipeline.get_transcription()
            if text:
                detection_result = content_detector.detect(text)
                # Get the appropriate text to display (original or masked)
                if detection_result["detected"]:
                    display_text = detection_result["masked_text"]
                else:
                    display_text = text

                print("Transcribed: ", display_text)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
