import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# Set seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_NAME = 'distilbert-base-uncased'


def load_jigsaw_data(file_path='train.csv'):
    """
    Load the Jigsaw dataset.

    The Jigsaw dataset contains text with binary labels for different categories of toxicity:
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Create a combined 'any_toxic' column that's True if any toxicity type is True
    toxicity_columns = ['toxic', 'severe_toxic',
                        'obscene', 'threat', 'insult', 'identity_hate']
    df['any_toxic'] = df[toxicity_columns].any(axis=1).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Percentage of toxic comments: {df['any_toxic'].mean() * 100:.2f}%")

    # Display distribution of toxicity types
    print("\nDistribution of toxicity types:")
    for col in toxicity_columns:
        print(f"{col}: {df[col].sum()} ({df[col].mean() * 100:.2f}%)")

    return df


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


class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return predictions, actual_labels


def redact_toxic_content(text, model, tokenizer, threshold=0.5, replacement="[REDACTED]"):
    """
    Identify and redact toxic content in the given text.

    Args:
        text: The input text to check and potentially redact
        model: The trained toxicity detection model
        tokenizer: The tokenizer used with the model
        threshold: Confidence threshold for toxicity detection
        replacement: What to replace toxic content with

    Returns:
        Tuple of (redacted_text, is_toxic, confidence)
    """
    model.eval()

    # Clean and prepare the text
    cleaned_text = clean_text(text)

    # Tokenize the text
    encoding = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get model prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        toxicity_score = probs[0][1].item()  # Probability of being toxic

    # Determine if the text is toxic based on the threshold
    is_toxic = toxicity_score >= threshold

    # Redact if toxic
    redacted_text = replacement if is_toxic else text

    return {
        'original_text': text,
        'redacted_text': redacted_text,
        'is_toxic': is_toxic,
        'confidence': toxicity_score
    }


def train_toxic_content_model(df):
    """Train the BERT-based toxicity detection model"""
    # Prepare data
    df['cleaned_text'] = df['comment_text'].apply(clean_text)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['cleaned_text'].values,
        df['any_toxic'].values,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df['any_toxic'].values
    )

    # Load tokenizer and create datasets
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = JigsawDataset(
        train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = JigsawDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # Binary classification: toxic or not
    )
    model = model.to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    history = {'train_loss': [], 'val_accuracy': []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device)
        history['train_loss'].append(train_loss)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        predictions, actual_labels = evaluate(model, val_loader, device)

        # Print metrics
        report = classification_report(
            actual_labels,
            predictions,
            target_names=['Non-toxic', 'Toxic'],
            digits=4
        )
        print(f"Validation Report:\n{report}")

        # Calculate accuracy
        accuracy = (np.array(predictions) == np.array(actual_labels)).mean()
        history['val_accuracy'].append(accuracy)
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Save the model
    model_save_path = 'toxic_content_detector_model'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    return model, tokenizer


def demo_redaction(model, tokenizer):
    """Demonstrate the toxic content detection and redaction"""
    test_texts = [
        "This is a normal comment about the weather today.",
        "You are such a stupid idiot, I hate you!",
        "I respectfully disagree with your opinion on this matter.",
        "Go kill yourself, nobody likes you anyway."
    ]

    print("\nTesting redaction on example texts:")
    for text in test_texts:
        result = redact_toxic_content(text, model, tokenizer)
        print(f"\nOriginal: {result['original_text']}")
        print(f"Redacted: {result['redacted_text']}")
        print(
            f"Toxic: {result['is_toxic']} (Confidence: {result['confidence']:.4f})")


def redact_dataset(df, model, tokenizer, text_column='comment_text'):
    """Apply redaction to an entire dataset"""
    results = []

    for text in tqdm(df[text_column], desc="Redacting dataset"):
        result = redact_toxic_content(text, model, tokenizer)
        results.append(result)

    # Create a dataframe with the results
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    df = load_jigsaw_data("./data/jigsaw/train.csv")

    # Train model
    model, tokenizer = train_toxic_content_model(df)

    # Demo the model
    demo_redaction(model, tokenizer)

    # Optional: Process a test dataset
    test_df = pd.read_csv("./data/jigsaw/test.csv")
    redacted_df = redact_dataset(test_df, model, tokenizer)
    redacted_df.to_csv("redacted_results.csv", index=False)
