import time
import re
import torch
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizerFast,  DistilBertForTokenClassification


class ContentDetector:
    """
    A content detection module that analyzes transcribed text for various types of content.
    Supports detection of:
    - Toxicity/harmful content
    - Custom categories

    Can also mask or redact detected content.
    """

    def __init__(self, detection_type="toxicity", threshold=0.7, custom_categories=None,
                 enable_masking=False, masking_method="asterisk",
                 toxic_word_list=None, use_token_detection=False):
        """
        Initialize the content detector.

        Args:
            detection_type (str): Type of content to detect ("toxicity" "custom")
            threshold (float): Confidence threshold for detection (0.0-1.0)
            custom_categories (list): List of custom categories to detect
            enable_masking (bool): Whether to enable content masking
            masking_method (str): Method to use for masking ("asterisk", "redact", "placeholder")
            toxic_word_list (list): List of words to mask (for word list based masking)
            use_token_detection (bool): Use token-level detection (more precise but slower)
        """
        self.detection_type = detection_type
        self.threshold = threshold
        self.custom_categories = custom_categories or []
        self.enable_masking = enable_masking
        self.masking_method = masking_method
        self.toxic_word_list = toxic_word_list or []
        self.use_token_detection = use_token_detection

        # Initialize content detection model based on type
        if detection_type == "toxicity":
            print("Loading toxicity detection model...")

            self.model = pipeline(
                "text-classification",
                model="facebook/roberta-hate-speech-dynabench-r4-target",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Toxicity detection model loaded")

            # If token detection is enabled, load token classification model
            if use_token_detection:
                print("Loading token classification model...")
                self.token_model = pipeline(
                    "token-classification",
                    model="Davlan/distilbert-base-multilingual-cased-ner-hrl",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Token classification model loaded")

            # Default toxic word list if none provided and using word list approach
            if not toxic_word_list and not use_token_detection:
                # This is a very basic example list - this should be more comprehensive
                self.toxic_word_list = [
                    "damn", "hell", "crap", "stupid", "idiot", "dumb",
                    "hate", "kill", "die", "terrible", "awful", "worst"
                ]
                print(
                    f"Using default toxic word list with {len(self.toxic_word_list)} words")

        elif detection_type == "custom":
            if not custom_categories:
                raise ValueError(
                    "Custom categories must be provided for custom detection")
            print("Loading custom content detection model...")
            # Initialize custom model based on categories
            self.model = self._init_custom_model()
            print("Custom content detection model loaded")

            # For custom categories, use custom token detection if needed
            if use_token_detection and enable_masking:
                self.token_model, self.token_tokenizer = self._init_custom_token_model()
        else:
            raise ValueError(f"Unsupported detection type: {detection_type}")

    def _init_custom_model(self):
        """Initialize custom detection model based on provided categories."""
        tokenizer = DistilBertTokenizer.from_pretrained(
            "toxic_content_detector_model")
        model = DistilBertForSequenceClassification.from_pretrained(
            "toxic_content_detector_model")
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def _init_custom_token_model(self):
        """Initialize custom token detection model."""
        try:
            # Load tokenizer and model from local path
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                "toxic_span_distilbert")
            model = DistilBertForTokenClassification.from_pretrained(
                "toxic_span_distilbert")

            device = 0 if torch.cuda.is_available() else -1
            if device >= 0:
                model = model.cuda()
            else:
                model = model.cpu()

            return model, tokenizer

        except Exception as e:
            print(f"Error loading custom token model: {e}")
            return None, None

    def detect(self, text):
        """
        Detect content in the provided text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Detection results with scores, flags, and masked text if enabled
        """
        if not text:
            return {"detected": False, "scores": {}, "flags": [], "masked_text": ""}

        start_time = time.time()

        if self.detection_type == "toxicity":
            result = self._detect_toxicity(text)
        elif self.detection_type == "custom":
            result = self._detect_custom(text)

        # Apply masking if enabled and content was detected
        if self.enable_masking and result["detected"]:
            print("Masking content...")
            result["masked_text"] = self._mask_content(text, result)
        else:
            result["masked_text"] = text

        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms

        result["latency"] = latency
        return result

    def _detect_toxicity(self, text):
        """Detect toxic content in text."""
        prediction = self.model(text)
        score = prediction[0]["score"]
        label = prediction[0]["label"]
        detected = score > self.threshold

        return {
            "detected": detected,
            "scores": {label: score},
            "flags": [label] if detected else [],
            "text": text
        }

    def _detect_custom(self, text):
        """Detect custom categories in text."""
        results = {"detected": False, "scores": {}, "flags": [], "text": text}

        # Run detection for each custom category
        for category in self.custom_categories:
            prediction = self.model(text)
            score = prediction[0]["score"] if prediction[0]["label"] == "LABEL_1" else 1 - \
                prediction[0]["score"]
            detected = score > self.threshold

            results["scores"][category] = score
            if detected:
                results["detected"] = True
                results["flags"].append(category)

        return results

    def _mask_content(self, text, detection_result):
        """
        Mask the detected content in the text.

        Args:
            text (str): Original text
            detection_result (dict): Detection results

        Returns:
            str: Masked text
        """
        # If no content detected, return original text
        if not detection_result["detected"]:
            return text

        masked_text = text

        # Method 1: Word list based masking
        if not self.use_token_detection:
            # Simple word list based approach
            words = text.split()
            masked_words = []

            for word in words:
                # Strip punctuation for checking
                clean_word = word.lower().strip('.,!?;:"\'()[]{}')

                if clean_word in self.toxic_word_list:
                    if self.masking_method == "asterisk":
                        # Replace each character with an asterisk
                        masked_word = '*' * len(word)
                    elif self.masking_method == "redact":
                        # Replace with [REDACTED]
                        masked_word = "[REDACTED]"
                    elif self.masking_method == "placeholder":
                        # Replace with [...]
                        masked_word = "[...]"
                    else:
                        masked_word = '*' * len(word)

                    masked_words.append(masked_word)
                else:
                    masked_words.append(word)

            masked_text = ' '.join(masked_words)

        # Method 2: Token-level detection and masking
        else:
            try:
                # Use token classification to identify specific entities to mask
                # Make sure model and tokenizer were loaded successfully
                if self.token_model is None or self.token_tokenizer is None:
                    raise ValueError(
                        "Token model or tokenizer not initialized properly")

                device = 0 if torch.cuda.is_available() else -1

                encoded = self.token_tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )

                # Save offset mapping and remove from inputs
                offset_mapping = encoded.pop('offset_mapping')[0]

                encoded = {k: v.to(device) for k, v in encoded.items()}

                self.token_model.cuda()
                with torch.no_grad():
                    outputs = self.token_model(**encoded)
                    predictions = torch.argmax(outputs.logits, dim=2)[0]

                # Convert token predictions to spans
                predicted_spans = []

                for i, (start, end) in enumerate(offset_mapping):
                    # Skip special tokens and padding
                    if start == 0 and end == 0:
                        continue

                    # Skip padding tokens
                    if i >= len(predictions):
                        break

                    # If token is predicted as toxic (B-toxic or I-toxic)
                    pred_class = predictions[i].item()
                    if pred_class in [0, 1]:  # 0 for B-toxic, 1 for I-toxic
                        # Add character positions to predicted spans
                        for char_idx in range(start.item(), end.item()):
                            predicted_spans.append(char_idx)

                tmp_masked_text = ""

                for i, char in enumerate(text):
                    if i in predicted_spans:
                        tmp_masked_text += f"*"
                    else:
                        tmp_masked_text += char
                masked_text = tmp_masked_text

            except Exception as e:
                print(f"Error in token-level masking: {e}")
                # Fall back to word list approach if token detection fails
                if not self.toxic_word_list:
                    return text

                # Simple word replacement
                for word in self.toxic_word_list:
                    if word.lower() in text.lower():
                        if self.masking_method == "asterisk":
                            replacement = '*' * len(word)
                        elif self.masking_method == "redact":
                            replacement = "[REDACTED]"
                        elif self.masking_method == "placeholder":
                            replacement = "[...]"
                        else:
                            replacement = '*' * len(word)

                        # Case-insensitive replacement
                        masked_text = masked_text.replace(word, replacement)

        return masked_text

    def clean_text(text):
        """Basic text cleaning function"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""


def main():
    try:
        content_detector = ContentDetector(
            detection_type="custom",
            threshold=0.5,
            custom_categories=["toxic"],
            enable_masking=True,
            masking_method="astrisk",
            use_token_detection=True,
        )

        example_text = "Go to hell, you idiot!"

        # Detect content
        detection_result = content_detector.detect(example_text)

        # Get the appropriate text to display (original or masked)
        if detection_result["detected"]:
            display_text = detection_result["masked_text"]
            print("Original text: ", example_text)
            print("Masked text: ", display_text)

            print(
                f"⚠️ FLAGGED CONTENT DETECTED: {detection_result['flags']}")
            print(
                f"Confidence scores: {detection_result['scores']}")
            print(
                f"Detection latency: {detection_result['latency']:.2f}ms")
        else:
            print("No flagged content detected")
            print("Original text: ", example_text)

    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
