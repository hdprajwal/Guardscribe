import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wave
from jiwer import wer, cer

from asr import SpeechToTextPipeline


class LJSpeechDataset:
    """
    Handles the LJ Speech dataset for ASR evaluation.

    The LJ Speech dataset consists of 13,100 short audio clips of a single speaker 
    reading passages from 7 non-fiction books.

    Dataset source: https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset
    """

    def __init__(self, data_dir="./data/LJSpeech-1.1"):
        """
        Initialize the LJ Speech dataset handler.

        Args:
            data_dir (str): Directory where the LJSpeech dataset is stored
        """
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.csv")
        self.wavs_dir = os.path.join(data_dir, "wavs")

        # Ensure the dataset directory exists
        if not os.path.exists(data_dir):
            print(f"Dataset directory {data_dir} not found.")
            print("Please download the LJSpeech dataset from Kaggle:")
            print("https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset")
            print("Extract it and pass the path to the script with --data_dir")

    def load_metadata(self):
        """
        Load metadata about the audio clips.

        Returns:
            pd.DataFrame: DataFrame with metadata about the audio clips
        """
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file not found at {self.metadata_path}.")
            return None

        try:
            # LJSpeech metadata.csv has columns: id|text|normalized_text
            metadata = pd.read_csv(self.metadata_path, sep="|", header=None,
                                   names=["id", "text", "normalized_text"])

            # Add path to audio files
            metadata["audio_path"] = metadata["id"].apply(
                lambda x: os.path.join(self.wavs_dir, f"{x}.wav")
            )

            return metadata
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None

    def get_samples(self, num_samples=None, random_seed=42, split=None):
        """
        Get samples from the LJ Speech dataset.

        Args:
            num_samples (int, optional): Number of samples to return. If None, returns all samples.
            random_seed (int): Random seed for reproducibility
            split (str, optional): If provided, splits the dataset and returns the specified split
                                  ('train', 'val', 'test')

        Returns:
            list: List of sample dictionaries with audio path and ground truth text
        """
        metadata = self.load_metadata()
        if metadata is None:
            return []

        # Check if all audio files exist
        metadata = metadata[metadata["audio_path"].apply(os.path.exists)]

        # Create train/val/test split if requested
        if split is not None:
            # Set random seed for reproducibility
            np.random.seed(random_seed)

            # Generate random indices for train/val/test split (80/10/10)
            indices = np.random.permutation(len(metadata))
            train_idx = indices[:int(0.8 * len(metadata))]
            val_idx = indices[int(0.8 * len(metadata)):int(0.9 * len(metadata))]
            test_idx = indices[int(0.9 * len(metadata)):]

            if split == "train":
                metadata = metadata.iloc[train_idx]
            elif split == "val":
                metadata = metadata.iloc[val_idx]
            elif split == "test":
                metadata = metadata.iloc[test_idx]
            else:
                print(f"Invalid split {split}. Using all samples.")

        # Shuffle and sample
        if num_samples is not None and num_samples < len(metadata):
            metadata = metadata.sample(n=num_samples, random_state=random_seed)

        # Convert to list of dictionaries
        samples = []
        for _, row in metadata.iterrows():
            samples.append({
                "id": row["id"],
                "audio_path": row["audio_path"],
                "text": row["text"],
                "normalized_text": row["normalized_text"]
            })

        print(f"Loaded {len(samples)} samples from LJ Speech dataset.")
        return samples


class ASREvaluator:
    """
    Evaluator for Automatic Speech Recognition (ASR) systems using the LJ Speech dataset
    """

    def __init__(self,
                 results_path="./evaluation_results/asr_evaluation_results"):
        """
        Initialize the ASR evaluator.

        Args:
            results_path (str): Path to save evaluation results
        """
        self.results_path = results_path

        # Create results directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        # Set up metrics tracking
        self.metrics = {
            "wer_scores": [],
            "cer_scores": [],
            "latencies": [],
            "audio_durations": [],
            "rtf_values": []
        }

        # Store detailed sample results
        self.sample_results = []

    def evaluate_asr_model(self,
                           samples,
                           asr_model="whisper",
                           max_samples=None,
                           use_normalized_text=True):
        """
        Evaluate an ASR model on the LJ Speech dataset.

        Args:
            samples (list): List of sample dictionaries with audio path and ground truth text
            asr_model (str): ASR model to use
            max_samples (int, optional): Maximum number of samples to evaluate
            use_normalized_text (bool): Whether to use normalized text for evaluation

        Returns:
            dict: Evaluation results
        """
        print(f"Evaluating ASR model: {asr_model}")

        # Initialize ASR pipeline
        asr_pipeline = SpeechToTextPipeline(asr_model=asr_model)

        # Reset metrics
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []

        self.sample_results = []

        # Limit samples if needed
        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]

        # Process samples
        for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
            # Get audio and text
            audio_path = sample["audio_path"]
            ground_truth_text = sample["normalized_text"] if use_normalized_text else sample["text"]

            try:
                # Read audio file
                with wave.open(audio_path, "rb") as wf:
                    sampling_rate = wf.getframerate()
                    wf.rewind()
                    audio_data = wf.readframes(wf.getnframes())

                # Calculate audio duration
                with wave.open(audio_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    audio_duration = frames / rate

                # Measure ASR time
                asr_start_time = time.time()
                transcription = asr_pipeline._transcribe(audio_data)
                latency = (time.time() - asr_start_time) * \
                    1000  # Convert to ms

                # Normalize text for fair comparison
                normalized_transcription = self._normalize_text(transcription)
                normalized_ground_truth = self._normalize_text(
                    ground_truth_text)

                # Calculate Real-Time Factor (RTF)
                # Convert latency to seconds
                rtf = (latency / 1000) / audio_duration

                # Calculate ASR accuracy metrics
                sample_wer = wer(normalized_ground_truth,
                                 normalized_transcription)
                sample_cer = cer(normalized_ground_truth,
                                 normalized_transcription)

                # Update metrics
                self.metrics["wer_scores"].append(sample_wer)
                self.metrics["cer_scores"].append(sample_cer)
                self.metrics["latencies"].append(latency)
                self.metrics["audio_durations"].append(audio_duration)
                self.metrics["rtf_values"].append(rtf)

                # Store detailed sample result
                self.sample_results.append({
                    "id": sample["id"],
                    "ground_truth": ground_truth_text,
                    "normalized_ground_truth": normalized_ground_truth,
                    "transcription": transcription,
                    "normalized_transcription": normalized_transcription,
                    "wer": sample_wer,
                    "cer": sample_cer,
                    "latency_ms": latency,
                    "audio_duration_s": audio_duration,
                    "rtf": rtf
                })

                # Print progress every 20 samples
                if (i+1) % 20 == 0:
                    print(f"Processed {i+1}/{len(samples)} samples")

            except Exception as e:
                print(f"Error processing sample {sample['id']}: {e}")
                continue

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()

        # Save results
        self._save_results(asr_model, use_normalized_text)

        return self.metrics

    def _normalize_text(self, text):
        """
        Normalize text for fair comparison.

        Args:
            text (str): Text to normalize

        Returns:
            str: Normalized text
        """
        if text is None:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        for punct in ",.!?;:\"'()[]{}-—":
            text = text.replace(punct, " ")

        # Replace multiple spaces with a single space
        text = " ".join(text.split())

        return text

    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics from all samples."""
        # ASR metrics
        self.metrics["mean_wer"] = np.mean(self.metrics["wer_scores"])
        self.metrics["median_wer"] = np.median(self.metrics["wer_scores"])
        self.metrics["mean_cer"] = np.mean(self.metrics["cer_scores"])
        self.metrics["median_cer"] = np.median(self.metrics["cer_scores"])

        # Latency metrics
        self.metrics["mean_latency"] = np.mean(self.metrics["latencies"])
        self.metrics["median_latency"] = np.median(self.metrics["latencies"])
        self.metrics["std_latency"] = np.std(self.metrics["latencies"])
        self.metrics["mean_rtf"] = np.mean(self.metrics["rtf_values"])
        self.metrics["median_rtf"] = np.median(self.metrics["rtf_values"])

        # Calculate WER distribution
        wer_scores = np.array(self.metrics["wer_scores"])
        self.metrics["wer_percentile_25"] = np.percentile(wer_scores, 25)
        self.metrics["wer_percentile_75"] = np.percentile(wer_scores, 75)
        self.metrics["wer_percentile_90"] = np.percentile(wer_scores, 90)

        # Count perfect transcriptions (WER = 0)
        self.metrics["perfect_transcriptions"] = np.sum(wer_scores == 0)
        self.metrics["perfect_transcription_rate"] = np.mean(wer_scores == 0)

    def _save_results(self, asr_model, use_normalized_text):
        """
        Save evaluation results to disk.

        Args:
            asr_model (str): ASR model used
            use_normalized_text (bool): Whether normalized text was used for evaluation
        """
        # Create a descriptive name for the run
        text_type = "normalized" if use_normalized_text else "raw"
        run_name = f"{asr_model}_{text_type}_ljspeech"

        # Save metrics
        metrics_file = os.path.join(
            self.results_path, f"{run_name}_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert numpy values to native Python types for JSON serialization
            serializable_metrics = {
                key: value if not isinstance(value, np.ndarray) and not isinstance(value, np.number)
                else value.item() if isinstance(value, np.number)
                else value.tolist()
                for key, value in self.metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)

        # Save detailed sample results
        samples_file = os.path.join(
            self.results_path, f"{run_name}_samples.json")
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_results, f, indent=2)

        print(f"Results saved to {metrics_file} and {samples_file}")

        # Print summary
        self._print_results_summary()

    def _print_results_summary(self):
        """Print a summary of the evaluation results."""
        print("\n=== ASR Evaluation Results Summary ===")
        print(f"Mean WER: {self.metrics['mean_wer']:.4f}")
        print(f"Median WER: {self.metrics['median_wer']:.4f}")
        print(f"Mean CER: {self.metrics['mean_cer']:.4f}")
        print(f"Median CER: {self.metrics['median_cer']:.4f}")
        print(
            f"Perfect Transcription Rate: {self.metrics['perfect_transcription_rate']:.4f} ({self.metrics['perfect_transcriptions']} samples)")

        print(f"\nLatency Metrics:")
        print(f"Mean Latency: {self.metrics['mean_latency']:.2f} ms")
        print(f"Median Latency: {self.metrics['median_latency']:.2f} ms")
        print(f"Std Latency: {self.metrics['std_latency']:.2f} ms")
        print(f"Mean Real-Time Factor: {self.metrics['mean_rtf']:.4f}")

    def generate_plots(self):
        """Generate visualization plots for the evaluation results."""
        print("Generating visualization plots...")

        # Get all result files
        result_files = [f for f in os.listdir(
            self.results_path) if f.endswith('_metrics.json')]

        if not result_files:
            print("No result files found for plotting.")
            return

        # Load all results
        all_results = {}
        for file in result_files:
            run_name = file.replace('_metrics.json', '')
            with open(os.path.join(self.results_path, file), 'r', encoding='utf-8') as f:
                all_results[run_name] = json.load(f)

        # Create plots
        self._plot_wer_cer_comparison(all_results)
        self._plot_latency_comparison(all_results)
        self._plot_wer_distribution()
        self._plot_latency_by_duration()
        self._plot_wer_vs_length()

        print(f"Plots saved to {self.results_path}")

    def _plot_wer_cer_comparison(self, all_results):
        """Plot WER and CER comparison across models."""
        plt.figure(figsize=(10, 6))

        runs = list(all_results.keys())
        wer_values = [all_results[run]['mean_wer'] for run in runs]
        cer_values = [all_results[run]['mean_cer'] for run in runs]

        x = np.arange(len(runs))
        width = 0.35

        plt.bar(x - width/2, wer_values, width, label='WER')
        plt.bar(x + width/2, cer_values, width, label='CER')

        plt.xlabel('ASR Model')
        plt.ylabel('Error Rate')
        plt.title('ASR Error Rates on LJ Speech')
        plt.xticks(x, [run.split('_')[0] for run in runs], rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_path, 'wer_cer_comparison.png'))
        plt.close()

    def _plot_latency_comparison(self, all_results):
        """Plot latency and RTF comparison across models."""
        plt.figure(figsize=(10, 6))

        runs = list(all_results.keys())
        latency_values = [all_results[run]['mean_latency'] for run in runs]
        rtf_values = [all_results[run]['mean_rtf'] for run in runs]

        # Primary y-axis for latency
        ax1 = plt.gca()
        ax1.bar(runs, latency_values, color='blue', alpha=0.7)
        ax1.set_xlabel('ASR Model')
        ax1.set_ylabel('Latency (ms)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        plt.xticks(rotation=45)

        # Secondary y-axis for RTF
        ax2 = ax1.twinx()
        ax2.plot(runs, rtf_values, 'r-o', linewidth=2)
        ax2.set_ylabel('Real-Time Factor', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('ASR Latency and Real-Time Factor')
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_path, 'latency_comparison.png'))
        plt.close()

    def _plot_wer_distribution(self):
        """Plot WER distribution for the most recently evaluated model."""
        if not self.metrics["wer_scores"]:
            print("No WER scores available for plotting distribution.")
            return

        plt.figure(figsize=(10, 6))

        plt.hist(self.metrics["wer_scores"], bins=20, alpha=0.7)
        plt.axvline(self.metrics["mean_wer"], color='r', linestyle='dashed',
                    linewidth=2, label=f'Mean WER: {self.metrics["mean_wer"]:.4f}')
        plt.axvline(self.metrics["median_wer"], color='g', linestyle='dashed',
                    linewidth=2, label=f'Median WER: {self.metrics["median_wer"]:.4f}')

        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Word Error Rates')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_path, 'wer_distribution.png'))
        plt.close()

    def _plot_latency_by_duration(self):
        """Plot latency vs audio duration for the most recently evaluated model."""
        if not self.metrics["latencies"] or not self.metrics["audio_durations"]:
            print("No latency or duration data available for plotting.")
            return

        plt.figure(figsize=(10, 6))

        plt.scatter(self.metrics["audio_durations"],
                    self.metrics["latencies"], alpha=0.7)

        # Add trend line
        z = np.polyfit(self.metrics["audio_durations"],
                       self.metrics["latencies"], 1)
        p = np.poly1d(z)
        plt.plot(self.metrics["audio_durations"], p(self.metrics["audio_durations"]), "r--",
                 label=f'Trend line: y={z[0]:.2f}x + {z[1]:.2f}')

        plt.xlabel('Audio Duration (seconds)')
        plt.ylabel('Processing Latency (ms)')
        plt.title('ASR Latency vs Audio Duration')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_path, 'latency_by_duration.png'))
        plt.close()

    def _plot_wer_vs_length(self):
        """Plot WER vs text length for the most recently evaluated model."""
        if not self.sample_results:
            print("No sample results available for plotting.")
            return

        plt.figure(figsize=(10, 6))

        # Calculate text lengths
        text_lengths = [len(s["normalized_ground_truth"].split())
                        for s in self.sample_results]
        wer_scores = [s["wer"] for s in self.sample_results]

        plt.scatter(text_lengths, wer_scores, alpha=0.7)

        # Add trend line
        z = np.polyfit(text_lengths, wer_scores, 1)
        p = np.poly1d(z)
        plt.plot(text_lengths, p(text_lengths), "r--",
                 label=f'Trend line: y={z[0]:.4f}x + {z[1]:.4f}')

        plt.xlabel('Text Length (words)')
        plt.ylabel('Word Error Rate (WER)')
        plt.title('WER vs Text Length')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_path, 'wer_vs_length.png'))
        plt.close()


def analyze_error_cases(samples_file, n_worst=10, output_file=None):
    """
    Analyze the worst-performing samples and output analysis.

    Args:
        samples_file (str): Path to the samples JSON file
        n_worst (int): Number of worst samples to analyze
        output_file (str, optional): Path to save analysis results
    """
    print(f"Analyzing error cases from {samples_file}...")

    # Load samples
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Sort by WER (descending)
    samples_sorted = sorted(samples, key=lambda x: x["wer"], reverse=True)
    worst_samples = samples_sorted[:n_worst]

    # Analyze worst samples
    analysis = []
    for i, sample in enumerate(worst_samples):
        analysis.append({
            "rank": i + 1,
            "id": sample["id"],
            "wer": sample["wer"],
            "ground_truth": sample["ground_truth"],
            "transcription": sample["transcription"],
            "audio_duration_s": sample["audio_duration_s"],
            "potential_issues": identify_potential_issues(
                sample["ground_truth"],
                sample["transcription"],
                sample["wer"]
            )
        })

    # Print analysis to console
    print(f"\n=== Top {n_worst} Worst Performing Samples ===")
    for item in analysis:
        print(
            f"\nRank {item['rank']} (WER: {item['wer']:.4f}, Duration: {item['audio_duration_s']:.2f}s)")
        print(f"Sample ID: {item['id']}")
        print(f"Ground truth: {item['ground_truth']}")
        print(f"Transcription: {item['transcription']}")
        print(f"Potential issues: {', '.join(item['potential_issues'])}")

    # Save analysis if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        print(f"Error analysis saved to {output_file}")

    return analysis


def identify_potential_issues(ground_truth, transcription, wer):
    """
    Identify potential issues in a transcription.

    Args:
        ground_truth (str): Ground truth text
        transcription (str): ASR transcription
        wer (float): Word error rate

    Returns:
        list: List of potential issues
    """
    issues = []

    # Check for completely wrong transcription
    if wer > 0.8:
        issues.append("Complete mistranscription")

    # Check for length difference
    gt_words = ground_truth.split()
    tr_words = transcription.split()

    if len(tr_words) < len(gt_words) * 0.7:
        issues.append("Significant content missing")
    elif len(tr_words) > len(gt_words) * 1.3:
        issues.append("Excessive content (possible hallucination)")

    # Check for potential issues with long utterances
    if len(gt_words) > 50 and wer > 0.3:
        issues.append("Long utterance (possible memory limitations)")

    # Check for potential numbers/dates issues
    has_numbers = any(c.isdigit() for c in ground_truth)
    numbers_missing = has_numbers and not any(
        c.isdigit() for c in transcription)
    if has_numbers and numbers_missing and wer > 0.3:
        issues.append("Numbers/dates not properly recognized")

    # Check for potential proper noun issues
    # Look for capitalized words (other than at the beginning of sentences)
    proper_nouns = [word for word in gt_words if word[0].isupper()
                    and not word.isupper()]
    if proper_nouns and wer > 0.3:
        issues.append("Potential proper noun recognition issues")

    # If no specific issues identified
    if not issues and wer > 0.5:
        issues.append("General recognition errors")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ASR models on LJ Speech dataset')
    parser.add_argument('--results_dir', type=str, default='./evaluation_results/asr_evaluation_results',
                        help='Path to save evaluation results')
    parser.add_argument('--data_dir', type=str, default='./data/LJSpeech-1.1',
                        help='Path to the LJSpeech dataset')
    parser.add_argument('--asr_models', type=str, nargs='+',
                        default=['wav2vec'],
                        help='ASR models to evaluate')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--split', type=str, default=None,
                        help='Dataset split to use (train, val, test, or None for all)')
    parser.add_argument('--normalized_text', action='store_true',
                        help='Use normalized text for evaluation')
    parser.add_argument('--generate_plots', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--analyze_errors', action='store_true',
                        help='Analyze worst-performing samples')

    args = parser.parse_args()

    # Initialize dataset
    dataset = LJSpeechDataset(data_dir=args.data_dir)

    # Get samples
    samples = dataset.get_samples(
        num_samples=args.max_samples, split=args.split)
    if not samples:
        print("No samples found in the dataset. Exiting.")
        return

    # Initialize evaluator
    evaluator = ASREvaluator(results_path=args.results_dir)

    # Evaluate each ASR model
    for asr_model in args.asr_models:
        evaluator.evaluate_asr_model(
            samples=samples,
            asr_model=asr_model,
            max_samples=args.max_samples,
            use_normalized_text=args.normalized_text
        )

        # Analyze errors if requested
        if args.analyze_errors:
            text_type = "normalized" if args.normalized_text else "raw"
            samples_file = os.path.join(
                args.results_dir, f"{asr_model}_{text_type}_ljspeech_samples.json")
            output_file = os.path.join(
                args.results_dir, f"{asr_model}_error_analysis.json")
            analyze_error_cases(samples_file, n_worst=10,
                                output_file=output_file)

    # Generate plots if requested
    if args.generate_plots:
        evaluator.generate_plots()


if __name__ == "__main__":
    main()
