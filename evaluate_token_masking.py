import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from tqdm import tqdm
import time
import json
import re
import os
import ast


class TokenLevelEvaluator:
    """Evaluator for token-level toxic span detection models"""

    def __init__(self, model_path, device=None):
        """
        Initialize the evaluator with a pre-trained token classification model

        Args:
            model_path: Path to the pre-trained DistilBERT token classification model
            device: Device to use for inference (cuda or cpu)
        """
        self.model_path = model_path

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForTokenClassification.from_pretrained(
            model_path)
        self.model.to(self.device)

        # Map ID to label
        self.id2label = {
            0: "B-toxic",  # Beginning of toxic span
            1: "I-toxic",  # Inside toxic span
            2: "O"         # Outside of toxic span (non-toxic)
        }

        # Results storage
        self.results = {
            "metrics": {},
            "predictions": [],
            "token_level_metrics": {},
            "span_level_metrics": {},
            "examples": {
                "correct": [],
                "partial": [],
                "missed": [],
                "false": []
            }
        }

    def clean_text(self, text):
        """Basic text cleaning function"""
        if isinstance(text, str):
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""

    def predict_spans(self, text):
        """
        Predict toxic spans in a given text

        Args:
            text: Text to analyze

        Returns:
            Dictionary with prediction results
        """
        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize with offsets
        start_time = time.time()
        encoded = self.tokenizer(
            cleaned_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        # Save offset mapping and remove from input to model
        offset_mapping = encoded.pop('offset_mapping')[0]

        # Move tensors to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**encoded)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)[0]

        # Calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms

        # Convert token predictions to character spans
        predicted_spans = []
        token_predictions = []

        for i, (start, end) in enumerate(offset_mapping):
            # Skip special tokens and padding
            if start == 0 and end == 0:
                continue

            # If token is predicted as toxic (B-toxic or I-toxic)
            token_pred = predictions[i].item()
            token_label = self.id2label[token_pred]
            token_predictions.append(token_label)

            if token_pred in [0, 1]:  # B-toxic or I-toxic
                # Add character positions to spans
                for char_idx in range(start.item(), end.item()):
                    predicted_spans.append(char_idx)

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "predicted_spans": predicted_spans,
            "token_predictions": token_predictions,
            "latency": latency
        }

    def evaluate_csv(self, csv_path, spans_col="spans", text_col="text", has_header=True):
        """
        Evaluate model on a CSV dataset with span annotations

        Args:
            csv_path: Path to the CSV file
            spans_col: Name of the column containing span lists (default: "spans")
            text_col: Name of the column containing text (default: "text")
            has_header: Whether the CSV has a header row (default: True)

        Returns:
            Dictionary with evaluation metrics
        """
        # Load the CSV file
        if has_header:
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(csv_path, header=None,
                             names=[spans_col, text_col])

        # Parse spans if needed
        if isinstance(df[spans_col].iloc[0], str):
            df[spans_col] = df[spans_col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Initialize metrics
        total_f1 = 0
        total_precision = 0
        total_recall = 0

        # Confusion matrix for token classification
        token_cm = {
            "B-toxic": {"B-toxic": 0, "I-toxic": 0, "O": 0},
            "I-toxic": {"B-toxic": 0, "I-toxic": 0, "O": 0},
            "O": {"B-toxic": 0, "I-toxic": 0, "O": 0}
        }

        # Process each sample
        print(f"Evaluating on dataset ({len(df)} samples)...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row[text_col]
            true_spans = row[spans_col]

            # Skip empty texts
            if not isinstance(text, str) or not text.strip():
                continue

            # Get predictions
            result = self.predict_spans(text)
            predicted_spans = result["predicted_spans"]

            # Calculate F1 for this sample
            precision, recall, f1 = self.calculate_span_f1(
                predicted_spans, true_spans)

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Categorize example based on F1 score
            if f1 == 1.0 and len(true_spans) > 0:
                # Perfect prediction of toxic spans
                example_category = "correct"
            elif f1 > 0 and f1 < 1.0:
                # Partial match
                example_category = "partial"
            elif f1 == 0 and len(true_spans) > 0:
                # Missed toxic spans completely
                example_category = "missed"
            elif len(predicted_spans) > 0 and len(true_spans) == 0:
                # False positive (predicted spans when none exist)
                example_category = "false"
            else:
                # Correctly identified no toxic spans
                example_category = "correct"

            # Store this example in appropriate category (up to 10 per category)
            if len(self.results["examples"][example_category]) < 10:
                self.results["examples"][example_category].append({
                    "text": text,
                    "true_spans": true_spans,
                    "predicted_spans": predicted_spans,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                })

            # Store individual prediction
            self.results["predictions"].append({
                "text": text,
                "true_spans": true_spans,
                "predicted_spans": predicted_spans,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "latency": result["latency"]
            })

        # Calculate average metrics
        avg_precision = total_precision / len(df)
        avg_recall = total_recall / len(df)
        avg_f1 = total_f1 / len(df)

        # Store token-level metrics
        self.results["token_level_metrics"] = {
            "confusion_matrix": token_cm
        }

        # Store span-level metrics
        self.results["span_level_metrics"] = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1
        }

        # Calculate latency statistics
        latencies = [pred["latency"] for pred in self.results["predictions"]]
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = max(latencies)

        self.results["metrics"] = {
            "span_level": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1
            },
            "latency": {
                "mean": avg_latency,
                "p95": p95_latency,
                "max": max_latency
            },
            "examples_count": {
                "correct": len([p for p in self.results["predictions"] if p["f1"] == 1.0 and len(p["true_spans"]) > 0]),
                "partial": len([p for p in self.results["predictions"] if p["f1"] > 0 and p["f1"] < 1.0]),
                "missed": len([p for p in self.results["predictions"] if p["f1"] == 0 and len(p["true_spans"]) > 0]),
                "false": len([p for p in self.results["predictions"] if len(p["predicted_spans"]) > 0 and len(p["true_spans"]) == 0]),
                "correct_empty": len([p for p in self.results["predictions"] if p["f1"] == 1.0 and len(p["true_spans"]) == 0])
            }
        }

        # Print summary
        print("\nToken-Level Classification Results:")
        print(f"Span-level Precision: {avg_precision:.4f}")
        print(f"Span-level Recall: {avg_recall:.4f}")
        print(f"Span-level F1 Score: {avg_f1:.4f}")
        print(f"Average Latency: {avg_latency:.2f} ms")

        # Example counts
        print("\nPrediction Categories:")
        for category, count in self.results["metrics"]["examples_count"].items():
            print(f"  {category}: {count}")

        return self.results["metrics"]

    def calculate_span_f1(self, predicted_spans, true_spans):
        """
        Calculate F1 score for span prediction

        Args:
            predicted_spans: List of predicted character positions
            true_spans: List of ground truth character positions

        Returns:
            Precision, recall, and F1 score
        """
        # Convert to sets
        predicted_set = set(predicted_spans)
        true_set = set(true_spans)

        # Handle special case: both predictions and ground truth are empty
        if len(predicted_set) == 0 and len(true_set) == 0:
            return 1.0, 1.0, 1.0

        # Calculate intersection (true positives)
        intersection = predicted_set.intersection(true_set)
        true_positives = len(intersection)

        # Calculate precision, recall, and F1
        precision = true_positives / \
            len(predicted_set) if len(predicted_set) > 0 else 0
        recall = true_positives / len(true_set) if len(true_set) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def visualize_span_prediction(self, text, true_spans, predicted_spans):
        """
        Create a visualization of span prediction for a given text

        Args:
            text: Original text
            true_spans: List of ground truth character positions
            predicted_spans: List of predicted character positions

        Returns:
            HTML representation of the visualization
        """
        # Convert spans to sets
        true_set = set(true_spans)
        pred_set = set(predicted_spans)

        # Create HTML with color-coded spans
        html = "<div style='font-family: monospace; white-space: pre-wrap;'>"

        for i, char in enumerate(text):
            if i in true_set and i in pred_set:
                # True positive: green
                html += f"<span style='background-color: #CCFFCC'>{char}</span>"
            elif i in true_set:
                # False negative: red
                html += f"<span style='background-color: #FFCCCC'>{char}</span>"
            elif i in pred_set:
                # False positive: yellow
                html += f"<span style='background-color: #FFFFCC'>{char}</span>"
            else:
                # True negative: no highlight
                html += char

        html += "</div>"
        return html

    def generate_visualizations(self, output_dir="token_evaluation_results"):
        """
        Generate visualizations of the evaluation results

        Args:
            output_dir: Directory to save visualizations

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)

        # Plot F1 score distribution
        plt.figure(figsize=(10, 6))
        f1_scores = [pred["f1"] for pred in self.results["predictions"]]
        sns.histplot(f1_scores, bins=20, kde=True)
        plt.xlabel('F1 Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of F1 Scores')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_distribution.png")
        plt.close()

        # Plot precision and recall scatter
        plt.figure(figsize=(10, 6))
        precisions = [pred["precision"]
                      for pred in self.results["predictions"]]
        recalls = [pred["recall"] for pred in self.results["predictions"]]
        plt.scatter(recalls, precisions, alpha=0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/precision_recall_scatter.png")
        plt.close()

        # Plot latency distribution
        plt.figure(figsize=(10, 6))
        latencies = [pred["latency"] for pred in self.results["predictions"]]
        sns.histplot(latencies, bins=30, kde=True)
        plt.axvline(self.results["metrics"]["latency"]["mean"], color='r', linestyle='--',
                    label=f'Mean: {self.results["metrics"]["latency"]["mean"]:.2f} ms')
        plt.axvline(self.results["metrics"]["latency"]["p95"], color='g', linestyle='--',
                    label=f'95th %: {self.results["metrics"]["latency"]["p95"]:.2f} ms')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png")
        plt.close()

        # Generate HTML visualizations of examples
        self.generate_html_examples(output_dir)

    def generate_html_examples(self, output_dir):
        """
        Generate HTML visualizations of example predictions

        Args:
            output_dir: Directory to save HTML file

        Returns:
            None
        """
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Toxic Span Prediction Examples</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .example { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; }
                .metrics { font-weight: bold; margin-bottom: 5px; }
                .legend { margin-top: 20px; }
                .legend span { padding: 2px 5px; margin-right: 10px; }
            </style>
        </head>
        <body>
            <h1>Toxic Span Prediction Examples</h1>
            
            <div class="legend">
                <h3>Color Legend:</h3>
                <span style="background-color: #CCFFCC">True Positive</span>
                <span style="background-color: #FFCCCC">False Negative</span>
                <span style="background-color: #FFFFCC">False Positive</span>
            </div>
        """

        # Add examples for each category
        for category in ["correct", "partial", "missed", "false"]:
            html += f"<h2>{category.capitalize()} Predictions</h2>"

            for example in self.results["examples"][category]:
                html += '<div class="example">'
                html += f'<div class="metrics">F1: {example["f1"]:.2f}, Precision: {example["precision"]:.2f}, Recall: {example["recall"]:.2f}</div>'
                html += self.visualize_span_prediction(
                    example["text"], example["true_spans"], example["predicted_spans"])
                html += '</div>'

        html += """
        </body>
        </html>
        """

        # Write to file
        with open(f"{output_dir}/examples.html", "w") as f:
            f.write(html)

    def save_results(self, output_path="token_evaluation_results.json"):
        """
        Save evaluation results to a JSON file

        Args:
            output_path: Path to save the results

        Returns:
            None
        """
        # Create a clean version without large arrays
        clean_results = {
            "metrics": self.results["metrics"],
            "examples_count": {
                "correct": len(self.results["examples"]["correct"]),
                "partial": len(self.results["examples"]["partial"]),
                "missed": len(self.results["examples"]["missed"]),
                "false": len(self.results["examples"]["false"])
            }
        }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2)

        print(f"Results saved to {output_path}")


def process_spans_csv(csv_path, has_header=True):
    """
    Process a CSV file with spans and text to ensure proper format

    Args:
        csv_path: Path to the CSV file
        has_header: Whether the CSV has a header row

    Returns:
        Processed DataFrame
    """
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None, names=["spans", "text"])

    # Ensure spans are in the correct format (list of integers)
    def parse_spans(span_str):
        if pd.isna(span_str) or span_str == '':
            return []

        try:
            if isinstance(span_str, str):
                if span_str.startswith('[') and span_str.endswith(']'):
                    return ast.literal_eval(span_str)
                else:
                    # Handle other formats if needed
                    return []
            elif isinstance(span_str, list):
                return span_str
            else:
                return []
        except:
            return []

    df["spans"] = df["spans"].apply(parse_spans)

    return df


def run_token_evaluation(model_path, csv_path, has_header=True, output_dir="./evaluation_results/toxic_span_evaluation_results"):
    """
    Run a complete evaluation of the token-level model using the specified CSV file

    Args:
        model_path: Path to the pre-trained token classification model
        csv_path: Path to the CSV file with spans and text
        has_header: Whether the CSV has a header row
        output_dir: Directory to save results and visualizations

    Returns:
        TokenLevelEvaluator instance with results
    """
    # Process the CSV file
    df = process_spans_csv(csv_path, has_header)

    # Check if there are samples with spans
    spans_count = sum(1 for spans in df["spans"] if len(spans) > 0)
    print(
        f"CSV file contains {len(df)} samples, {spans_count} with toxic spans")

    # Create evaluator
    evaluator = TokenLevelEvaluator(model_path)

    # Evaluate on the processed CSV
    evaluator.evaluate_csv(csv_path, spans_col="spans",
                           text_col="text", has_header=has_header)

    # Generate visualizations
    evaluator.generate_visualizations(output_dir)

    # Save results
    evaluator.save_results(f"{output_dir}/token_evaluation_results.json")

    return evaluator


if __name__ == "__main__":
    # Path to the fine-tuned token classification model
    model_path = "toxic_span_distilbert"

    # Path to the CSV file
    csv_path = "./data/toxic_spans/tsd_test.csv"

    # Run evaluation
    evaluator = run_token_evaluation(
        model_path=model_path,
        csv_path=csv_path,
        has_header=True,
        # Directory to save results and visualizations
        output_dir="./evaluation_results/toxic_span_evaluation_results"
    )
