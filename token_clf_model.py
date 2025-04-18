import pandas as pd
import numpy as np
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


def load_data(file_path):
    """Load data from CSV file"""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """Preprocess the data - convert spans from string to list of integers"""
    # Function to safely convert spans string to list
    def parse_spans(span_str):
        if pd.isna(span_str) or span_str == '':
            return []

        try:
            if isinstance(span_str, str):
                # Handle different possible formats
                if span_str.startswith('[') and span_str.endswith(']'):
                    # Regular list format: [1, 2, 3]
                    return ast.literal_eval(span_str)
                elif ',' in span_str:
                    # Comma-separated format without brackets: 1, 2, 3
                    return [int(x.strip()) for x in span_str.split(',') if x.strip().isdigit()]
                elif ' ' in span_str:
                    # Space-separated format: 1 2 3
                    return [int(x) for x in span_str.split() if x.isdigit()]
                else:
                    # Single number: 42
                    return [int(span_str)] if span_str.isdigit() else []
            elif isinstance(span_str, list):
                # Already a list
                return span_str
            else:
                return []
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse spans: {span_str}")
            return []

    # Apply the parsing function to the spans column
    df['spans'] = df['spans'].apply(parse_spans)

    # Print statistics
    total_spans = sum(len(spans) for spans in df['spans'])
    avg_spans_per_sample = total_spans / len(df) if len(df) > 0 else 0

    print(f"Total spans: {total_spans}")
    print(f"Average spans per sample: {avg_spans_per_sample:.2f}")
    print(
        f"Samples with no spans: {sum(len(spans) == 0 for spans in df['spans'])}")

    return df


def create_token_labels(text, spans, tokenizer):
    """Create token-level labels from character-level spans"""
    # Ensure spans is a list of integers
    if not isinstance(spans, list):
        spans = []

    # Convert spans to a set for faster lookup
    spans_set = set(spans)

    # Tokenize text with offset mapping
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding=False,  # We'll handle padding in the Dataset class
        add_special_tokens=True
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    offset_mapping = encoded['offset_mapping']

    # Create labels for each token
    labels = []
    prev_is_toxic = False

    for i, (start, end) in enumerate(offset_mapping):
        # Special tokens like [CLS] and [SEP]
        if start == 0 and end == 0:
            # -100 is ignored by PyTorch's CrossEntropyLoss
            labels.append(-100)
            prev_is_toxic = False
            continue

        # Check if token overlaps with any toxic span
        # A token is toxic if any of its characters are in the toxic spans
        is_toxic = False
        if end > start:  # Skip empty tokens
            for char_pos in range(start, end):
                if char_pos in spans_set:
                    is_toxic = True
                    break

        # BIO tagging scheme: B-toxic for beginning, I-toxic for inside, O for outside
        if is_toxic:
            if prev_is_toxic:
                labels.append(1)  # I-toxic (continuation)
            else:
                labels.append(0)  # B-toxic (beginning)
            prev_is_toxic = True
        else:
            labels.append(2)  # O (outside)
            prev_is_toxic = False

    # Sanity check: make sure all lists have the same length
    assert len(input_ids) == len(attention_mask) == len(labels), \
        f"Length mismatch: {len(input_ids)}, {len(attention_mask)}, {len(labels)}"

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class ToxicSpanDataset(Dataset):
    def __init__(self, texts, spans, tokenizer, max_length=512):
        self.texts = texts
        self.spans = spans
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        spans = self.spans[idx]

        # Create token-level labels
        encoding = create_token_labels(text, spans, self.tokenizer)

        # Make sure all tensors have same length by padding
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        labels = encoding['labels']

        # Pad or truncate to max_length
        if len(input_ids) < self.max_length:
            # Padding
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + \
                [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            # -100 is ignored by loss function
            labels = labels + [-100] * padding_length
        else:
            # Truncation
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels)
        }


def train_model(train_dataset, val_dataset, num_labels=3):
    """Fine-tune DistilBERT for token classification"""
    # Initialize model
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_labels)

    # Define training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Use DataLoader with appropriate batch_size and collate_fn
    # Default collate_fn works fine since we've standardized lengths in the Dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8
    )

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Add tqdm progress bar for training
        train_progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for batch in train_progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            current_loss = loss.item()
            train_loss += current_loss
            train_progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Validation
        model.eval()
        val_loss = 0

        # Add tqdm progress bar for validation
        val_progress_bar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")

        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                current_loss = outputs.loss.item()
                val_loss += current_loss

                # Update validation progress bar
                val_progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')

    return model

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')

    return model


def predict_spans(text, model, tokenizer, device):
    """Predict toxic spans for a given text"""
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

    offset_mapping = encoded.pop('offset_mapping')[0]

    encoded = {k: v.to(device) for k, v in encoded.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        predictions = torch.argmax(outputs.logits, dim=2)[
            0]  # Get predictions for the first (and only) batch

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

    return predicted_spans


def f1_score(predicted_spans, ground_truth_spans):
    """Calculate F1 score for span detection"""
    # Ensure inputs are lists
    if not isinstance(predicted_spans, list):
        predicted_spans = []
    if not isinstance(ground_truth_spans, list):
        ground_truth_spans = []

    predicted_set = set(predicted_spans)
    ground_truth_set = set(ground_truth_spans)

    # Handle special case: both prediction and ground truth are empty
    if len(predicted_set) == 0 and len(ground_truth_set) == 0:
        return 1.0, 1.0, 1.0  # Perfect score

    # True positives: character positions that are in both predictions and ground truth
    true_positives = len(predicted_set & ground_truth_set)

    # False positives: character positions that are in predictions but not in ground truth
    false_positives = len(predicted_set - ground_truth_set)

    # False negatives: character positions that are in ground truth but not in predictions
    false_negatives = len(ground_truth_set - predicted_set)

    # Calculate precision, recall, and F1
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def main():
    # Set reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading and preprocessing data...")
    # Load and preprocess data
    data_path = './data/toxic_spans/data/tsd_train.csv'
    df = load_data(data_path)
    df = preprocess_data(df)

    # Print sample to verify data format
    print(f"Total samples: {len(df)}")
    print("\nSample data:")
    for i in range(min(3, len(df))):
        print(f"Text: {df['text'].iloc[i][:100]}...")
        print(f"Spans: {df['spans'].iloc[i][:20]}...")
        print("-" * 50)

    # Split data into train and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(
        f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased')

    # Create datasets with a consistent max_length
    max_length = 128  # Reduced from 512 for faster training
    print(f"Creating datasets with max_length={max_length}...")

    train_dataset = ToxicSpanDataset(
        train_df['text'].tolist(),
        train_df['spans'].tolist(),
        tokenizer,
        max_length=max_length
    )

    val_dataset = ToxicSpanDataset(
        val_df['text'].tolist(),
        val_df['spans'].tolist(),
        tokenizer,
        max_length=max_length
    )

    # Check a sample from the dataset to verify formatting
    sample = train_dataset[0]
    print("\nSample from dataset:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")

    # Fine-tune model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = train_model(train_dataset, val_dataset)

    # Evaluate model on validation set
    print("Evaluating model...")
    val_predictions = []
    val_ground_truth = val_df['spans'].tolist()

    # Add progress bar for evaluation
    eval_progress_bar = tqdm(
        enumerate(val_df['text'].tolist()),
        desc="Predicting",
        total=len(val_df)
    )

    for i, text in eval_progress_bar:
        predicted_spans = predict_spans(text, model, tokenizer, device)
        val_predictions.append(predicted_spans)

    # Calculate F1 score
    print("Calculating F1 score...")
    total_f1 = 0
    f1_scores = []

    # Add progress bar for calculating F1 scores
    f1_progress_bar = tqdm(
        zip(val_predictions, val_ground_truth),
        desc="Calculating F1",
        total=len(val_predictions)
    )

    for pred, truth in f1_progress_bar:
        precision, recall, f1 = f1_score(pred, truth)
        f1_scores.append(f1)
        total_f1 += f1

    average_f1 = total_f1 / len(val_predictions)
    print(f'Validation F1 Score: {average_f1:.4f}')

    # Save model
    print("Saving model...")
    model.save_pretrained('toxic_span_distilbert')
    tokenizer.save_pretrained('toxic_span_distilbert')

    print('Model saved to toxic_span_distilbert/')

    # Demo prediction on a sample text
    print("\nDemo prediction:")
    sample_text = val_df['text'].iloc[0]
    sample_truth = val_df['spans'].iloc[0]

    print(f"Text: {sample_text}")
    print(f"True toxic spans: {sample_truth}")

    predicted_spans = predict_spans(sample_text, model, tokenizer, device)
    print(f"Predicted toxic spans: {predicted_spans}")

    # Highlight toxic spans in the text
    highlighted_text = ""
    for i, char in enumerate(sample_text):
        if i in predicted_spans:
            highlighted_text += f"[{char}]"
        else:
            highlighted_text += char

    print(f"Text with highlighted toxic spans: {highlighted_text}")


if __name__ == '__main__':
    main()
