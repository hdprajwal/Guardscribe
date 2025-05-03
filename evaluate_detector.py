import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
import time
import json
import re
import os


class ToxicityEvaluator:
    """Simple evaluator for toxicity detection models using a custom CSV dataset"""

    def __init__(self, model_path, threshold=0.7, device=None):
        """
        Initialize the evaluator with a pre-trained model

        Args:
            model_path: Path to the pre-trained DistilBERT model
            threshold: Classification threshold for binary decisions
            device: Device to use for inference (cuda or cpu)
        """
        self.model_path = model_path
        self.threshold = threshold

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_path)
        self.model.to(self.device)

        # Results storage
        self.results = {
            "metrics": {},
            "predictions": [],
            "error_examples": {
                "false_positives": [],
                "false_negatives": []
            }
        }

    def clean_text(self, text):
        """Basic text cleaning function"""
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""

    def predict(self, text):
        """
        Make a prediction on a single text input

        Args:
            text: Text to classify

        Returns:
            Dictionary with prediction results
        """
        # Clean text
        cleaned_text = self.clean_text(text)

        # Tokenize
        start_time = time.time()
        encoded = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            # Probability of toxic class
            toxic_prob = probabilities[0][1].item()

        # Calculate latency
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms

        # Make binary prediction
        prediction = 1 if toxic_prob >= self.threshold else 0

        return {
            "text": text,
            "toxic_probability": toxic_prob,
            "prediction": prediction,
            "latency": latency
        }

    def evaluate_csv(self, csv_path, text_col=0, label_col=1, has_header=False):
        """
        Evaluate model on a CSV dataset

        Args:
            csv_path: Path to the CSV file
            text_col: Index or name of the column containing text (default: 0)
            label_col: Index or name of the column containing labels (default: 1)
            has_header: Whether the CSV has a header row (default: False)

        Returns:
            Dictionary with evaluation metrics
        """
        # Load the CSV file
        if has_header:
            df = pd.read_csv(csv_path)
            # Get column names if they're indices
            if isinstance(text_col, int):
                text_col = df.columns[text_col]
            if isinstance(label_col, int):
                label_col = df.columns[label_col]
        else:
            df = pd.read_csv(csv_path, header=None)
            # Make sure text_col and label_col are integers
            if not isinstance(text_col, int):
                raise ValueError(
                    "text_col must be an integer for CSV without header")
            if not isinstance(label_col, int):
                raise ValueError(
                    "label_col must be an integer for CSV without header")

        texts = df[text_col].tolist()
        # convert labels to integers
        df[label_col] = df[label_col].astype(int)

        labels = df[label_col].tolist()

        # Validate labels
        for i, label in enumerate(labels):
            if label not in [0, 1]:
                print(
                    f"Warning: Invalid label {label} at index {i}, converting to 0")
                labels[i] = 0

        predictions = []
        probabilities = []
        latencies = []

        # Process each sample
        print(f"Evaluating on dataset ({len(texts)} samples)...")
        for text, label in tqdm(zip(texts, labels), total=len(texts)):
            result = self.predict(text)

            predictions.append(result["prediction"])
            probabilities.append(result["toxic_probability"])
            latencies.append(result["latency"])

            # Store misclassifications for error analysis
            if label == 0 and result["prediction"] == 1:
                self.results["error_examples"]["false_positives"].append({
                    "text": text,
                    "probability": result["toxic_probability"]
                })
            elif label == 1 and result["prediction"] == 0:
                self.results["error_examples"]["false_negatives"].append({
                    "text": text,
                    "probability": result["toxic_probability"]
                })

            # Store individual prediction
            self.results["predictions"].append({
                "text": text,
                "true_label": label,
                "prediction": result["prediction"],
                "probability": result["toxic_probability"],
                # Store latency with each prediction
                "latency": result["latency"]
            })

        # Calculate metrics
        report = classification_report(
            labels,
            predictions,
            target_names=["Non-toxic", "Toxic"],
            output_dict=True
        )

        # Calculate ROC and PR curves
        fpr, tpr, _ = roc_curve(labels, probabilities)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = auc(recall, precision)

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = max(latencies)

        # Store results
        self.results["metrics"] = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": roc_auc
            },
            "pr_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "auc": pr_auc
            },
            "latency": {
                "mean": avg_latency,
                "p95": p95_latency,
                "max": max_latency,
                "all": latencies  # Store all latency values
            }
        }

        # Print summary
        print("\nEvaluation Results:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"F1 Score (Toxic): {report['Toxic']['f1-score']:.4f}")
        print(f"Precision (Toxic): {report['Toxic']['precision']:.4f}")
        print(f"Recall (Toxic): {report['Toxic']['recall']:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"Average Latency: {avg_latency:.2f} ms")
        print(f"95th Percentile Latency: {p95_latency:.2f} ms")

        # Confusion matrix as a table
        print("\nConfusion Matrix:")
        print(f"                 Predicted Non-toxic  Predicted Toxic")
        print(f"Actual Non-toxic       {cm[0][0]}                {cm[0][1]}")
        print(f"Actual Toxic           {cm[1][0]}                {cm[1][1]}")

        # Error analysis summary
        print("\nError Analysis:")
        print(
            f"False Positives: {len(self.results['error_examples']['false_positives'])}")
        print(
            f"False Negatives: {len(self.results['error_examples']['false_negatives'])}")

        return self.results["metrics"]

    def evaluate_thresholds(self, csv_path, text_col=0, label_col=1, has_header=False, thresholds=None):
        """
        Evaluate model performance across different thresholds

        Args:
            csv_path: Path to the CSV file
            text_col: Index or name of the column containing text
            label_col: Index or name of the column containing labels
            has_header: Whether the CSV has a header row
            thresholds: List of thresholds to evaluate

        Returns:
            Dictionary with metrics at each threshold
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Load the CSV file
        if has_header:
            df = pd.read_csv(csv_path)
            # Get column names if they're indices
            if isinstance(text_col, int):
                text_col = df.columns[text_col]
            if isinstance(label_col, int):
                label_col = df.columns[label_col]
        else:
            df = pd.read_csv(csv_path, header=None)
            # Make sure text_col and label_col are integers
            if not isinstance(text_col, int):
                raise ValueError(
                    "text_col must be an integer for CSV without header")
            if not isinstance(label_col, int):
                raise ValueError(
                    "label_col must be an integer for CSV without header")

        texts = df[text_col].tolist()
        # convert labels to integers
        df[label_col] = df[label_col].astype(int)

        labels = df[label_col].tolist()

        # Get predictions once
        results = []
        print("Getting predictions for threshold analysis...")
        for text in tqdm(texts):
            result = self.predict(text)
            results.append(result)

        probabilities = [r["toxic_probability"] for r in results]

        # Evaluate at each threshold
        threshold_results = {}

        print("Evaluating performance at different thresholds...")
        for threshold in thresholds:
            predictions = [1 if p >= threshold else 0 for p in probabilities]

            report = classification_report(
                labels,
                predictions,
                target_names=["Non-toxic", "Toxic"],
                output_dict=True
            )

            threshold_results[threshold] = {
                "accuracy": report["accuracy"],
                "f1": report["Toxic"]["f1-score"],
                "precision": report["Toxic"]["precision"],
                "recall": report["Toxic"]["recall"]
            }

            print(f"Threshold {threshold:.1f}: F1={report['Toxic']['f1-score']:.4f}, "
                  f"Precision={report['Toxic']['precision']:.4f}, "
                  f"Recall={report['Toxic']['recall']:.4f}")

        self.results["threshold_analysis"] = threshold_results

        return threshold_results

    def generate_visualizations(self, output_dir="evaluation_results"):
        """
        Generate visualizations of the evaluation results

        Args:
            output_dir: Directory to save the plots

        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)

        if not self.results["metrics"]:
            print("No evaluation results to visualize")
            return

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(self.results["metrics"]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-toxic", "Toxic"],
            yticklabels=["Non-toxic", "Toxic"]
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr = self.results["metrics"]["roc_curve"]["fpr"]
        tpr = self.results["metrics"]["roc_curve"]["tpr"]
        roc_auc = self.results["metrics"]["roc_curve"]["auc"]

        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/roc_curve.png")
        plt.close()

        # Plot PR curve
        plt.figure(figsize=(8, 6))
        precision = self.results["metrics"]["pr_curve"]["precision"]
        recall = self.results["metrics"]["pr_curve"]["recall"]
        pr_auc = self.results["metrics"]["pr_curve"]["auc"]

        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pr_curve.png")
        plt.close()

        # Plot latency distribution - FIXED THIS PART
        plt.figure(figsize=(10, 6))
        # Use the latency values from metrics instead
        latencies = self.results["metrics"]["latency"]["all"]
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

        # If threshold analysis exists
        if "threshold_analysis" in self.results:
            # Plot threshold vs metrics
            plt.figure(figsize=(10, 6))
            thresholds = sorted(
                list(self.results["threshold_analysis"].keys()))
            accuracy = [self.results["threshold_analysis"]
                        [t]["accuracy"] for t in thresholds]
            f1 = [self.results["threshold_analysis"][t]["f1"]
                  for t in thresholds]
            precision = [self.results["threshold_analysis"]
                         [t]["precision"] for t in thresholds]
            recall = [self.results["threshold_analysis"][t]["recall"]
                      for t in thresholds]

            plt.plot(thresholds, accuracy, 'o-', label='Accuracy')
            plt.plot(thresholds, f1, 'o-', label='F1 Score')
            plt.plot(thresholds, precision, 'o-', label='Precision')
            plt.plot(thresholds, recall, 'o-', label='Recall')

            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Performance Metrics vs. Threshold')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/threshold_analysis.png")
            plt.close()

        # Generate error examples document
        if self.results["error_examples"]["false_positives"] or self.results["error_examples"]["false_negatives"]:
            with open(f"{output_dir}/error_examples.txt", "w") as f:
                f.write("FALSE POSITIVES (Non-toxic texts classified as toxic)\n")
                f.write("=" * 80 + "\n\n")
                for i, example in enumerate(self.results["error_examples"]["false_positives"][:10]):
                    f.write(
                        f"Example {i+1} (Probability: {example['probability']:.4f}):\n")
                    f.write(f"{example['text']}\n\n")

                f.write(
                    "\n\nFALSE NEGATIVES (Toxic texts classified as non-toxic)\n")
                f.write("=" * 80 + "\n\n")
                for i, example in enumerate(self.results["error_examples"]["false_negatives"][:10]):
                    f.write(
                        f"Example {i+1} (Probability: {example['probability']:.4f}):\n")
                    f.write(f"{example['text']}\n\n")

    def save_results(self, output_path="evaluation_results.json"):
        """
        Save evaluation results to a JSON file

        Args:
            output_path: Path to save the results

        Returns:
            None
        """
        # Create a clean version without large arrays
        clean_results = {
            "metrics": {
                "classification_report": self.results["metrics"]["classification_report"],
                "confusion_matrix": self.results["metrics"]["confusion_matrix"],
                "roc_auc": self.results["metrics"]["roc_curve"]["auc"],
                "pr_auc": self.results["metrics"]["pr_curve"]["auc"],
                "latency": {
                    "mean": self.results["metrics"]["latency"]["mean"],
                    "p95": self.results["metrics"]["latency"]["p95"],
                    "max": self.results["metrics"]["latency"]["max"]
                }
            },
            "error_analysis": {
                "false_positive_count": len(self.results["error_examples"]["false_positives"]),
                "false_negative_count": len(self.results["error_examples"]["false_negatives"])
            }
        }

        # Add threshold analysis if it exists
        if "threshold_analysis" in self.results:
            clean_results["threshold_analysis"] = self.results["threshold_analysis"]

        # Save to file
        with open(output_path, "w") as f:
            json.dump(clean_results, f, indent=2)

        print(f"Results saved to {output_path}")


def run_evaluation(model_path, csv_path, has_header=False, output_dir="./evaluation_results/toxicity_evaluation_results"):
    """
    Run a complete evaluation of the model using the specified CSV file

    Args:
        model_path: Path to the pre-trained model
        csv_path: Path to the CSV file with text and labels
        has_header: Whether the CSV has a header row
        output_dir: Directory to save results and visualizations

    Returns:
        ToxicityEvaluator instance with results
    """
    # Create evaluator
    evaluator = ToxicityEvaluator(model_path)

    # Evaluate on the CSV file
    evaluator.evaluate_csv(csv_path, text_col=0,
                           label_col=1, has_header=has_header)

    # Analyze model performance across different thresholds
    evaluator.evaluate_thresholds(
        csv_path, text_col=0, label_col=1, has_header=has_header)

    # Generate visualizations
    evaluator.generate_visualizations(output_dir)

    # Save results
    evaluator.save_results(f"{output_dir}/evaluation_results.json")

    return evaluator


if __name__ == "__main__":
    # Path to the fine-tuned DistilBERT model
    model_path = "toxic_content_detector_model"

    # Path to the CSV file
    csv_path = "./data/test_data.csv"

    # Run evaluation
    evaluator = run_evaluation(
        model_path=model_path,
        csv_path=csv_path,
        has_header=True,
        output_dir="./evaluation_results/toxicity_evaluation_results"
    )
