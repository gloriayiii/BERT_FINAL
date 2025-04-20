import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import time
import os
from datasets import load_dataset
from bertviz import head_view, model_view
from transformers import AutoModel, AutoTokenizer


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed()

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare datasets
def load_and_prepare_datasets():
    print("Loading datasets...")

    # Load IMDB dataset
    imdb_dataset = load_dataset("stanfordnlp/imdb")

    # Load SST-2 dataset
    sst2_dataset = load_dataset("stanfordnlp/sst2")

    # Load Twitter Sentiment dataset
    twitter_dataset = load_dataset("carblacac/twitter-sentiment-analysis")

    print("Datasets loaded successfully")

    return {
        "imdb": imdb_dataset,
        "sst2": sst2_dataset,
        "twitter": twitter_dataset
    }


# Custom Dataset class for BERT inputs
class BERTSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize and prepare for BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_dataloader(dataset, tokenizer, batch_size=16, train_size=0.8, split_seed=42):
    """
    Prepare dataloaders from dataset with robust field detection and error handling.

    Args:
        dataset: Hugging Face dataset or dictionary
        tokenizer: BERT tokenizer
        batch_size: Batch size for dataloader
        train_size: Proportion of data to use for training
        split_seed: Random seed for data splitting

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    # Extract texts and labels with flexible field detection
    if isinstance(dataset, dict):  # Hugging Face dataset format
        if 'train' in dataset and 'test' in dataset:
            # Identify the text field name
            text_field = None
            for field in ['text', 'sentence', 'content', 'review']:
                if field in dataset['train'].column_names:
                    text_field = field
                    break

            if text_field is None:
                raise ValueError(f"No text field found in dataset. Available fields: {dataset['train'].column_names}")

            # Check for label field
            if 'label' not in dataset['train'].column_names:
                raise ValueError(
                    f"No 'label' field found in dataset. Available fields: {dataset['train'].column_names}")

            # Extract data
            train_texts = dataset['train'][text_field]
            train_labels = dataset['train']['label']

            test_texts = dataset['test'][text_field]
            test_labels = dataset['test']['label']

            # Create validation set from training set
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=(1 - train_size), random_state=split_seed
            )
        else:
            # If no predefined split, identify available splits
            available_splits = list(dataset.keys())
            print(f"Available dataset splits: {available_splits}")

            # Choose the first available split
            split_name = available_splits[0]

            # Identify the text field name
            text_field = None
            for field in ['text', 'sentence', 'content', 'review']:
                if field in dataset[split_name].column_names:
                    text_field = field
                    break

            if text_field is None:
                raise ValueError(
                    f"No text field found in dataset['{split_name}']. Available fields: {dataset[split_name].column_names}")

            # Check for label field
            if 'label' not in dataset[split_name].column_names:
                raise ValueError(
                    f"No 'label' field found in dataset['{split_name}']. Available fields: {dataset[split_name].column_names}")

            # Extract all data
            all_texts = dataset[split_name][text_field]
            all_labels = dataset[split_name]['label']

            # First split for train
            train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                all_texts, all_labels, train_size=train_size, random_state=split_seed
            )

            # Second split for val and test
            val_texts, test_texts, val_labels, test_labels = train_test_split(
                temp_texts, temp_labels, train_size=0.5, random_state=split_seed
            )
    else:
        raise ValueError("Unsupported dataset format. Expected a Hugging Face dataset or dictionary.")

    # Debug information
    print(f"Dataset splits - Train: {len(train_texts)}, Validation: {len(val_texts)}, Test: {len(test_texts)}")

    # Create datasets
    train_dataset = BERTSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = BERTSentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = BERTSentimentDataset(test_texts, test_labels, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=batch_size
    )

    return train_dataloader, val_dataloader, test_dataloader

# Model Architecture: Standard BERT for Sentiment Classification
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=2, dropout_rate=0.3):
        super(BERTSentimentClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # Extract attention weights
        attentions = outputs.attentions

        # Rest of your forward method
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, attentions


# Hybrid BERT with BiLSTM
class HybridBERTLSTM(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=2,
                 hidden_size=768, lstm_hidden_size=256, dropout_rate=0.3):
        super(HybridBERTLSTM, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        # Set hidden size explicitly from BERT config
        self.hidden_size = self.bert.config.hidden_size  # Get actual BERT hidden size

        # Bidirectional LSTM layer after BERT
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,  # Use BERT's actual hidden size
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Final classifier - make sure dimensions match with LSTM output
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)  # BiLSTM output is 2x hidden size

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs, including all hidden states
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True  # Add this to get attention weights
        )

        # Use the sequence output from BERT's last layer
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Pass through BiLSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)

        # Concatenate the final forward and backward hidden states
        lstm_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # Apply dropout and classify
        lstm_final = self.dropout(lstm_final)
        logits = self.classifier(lstm_final)

        return logits, outputs.attentions

# Aspect-Based Sentiment Analysis
class AspectBasedSentimentAnalysis(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", num_sentiment_classes=2,
                 num_aspect_classes=5, hidden_size=768, dropout_rate=0.3):
        super(AspectBasedSentimentAnalysis, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # Shared feature layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_classes)
        self.aspect_classifier = nn.Linear(hidden_size, num_aspect_classes)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract features
        pooled_output = outputs.pooler_output
        features = self.feature_extractor(pooled_output)

        # Task-specific predictions
        sentiment_logits = self.sentiment_classifier(features)
        aspect_logits = self.aspect_classifier(features)

        return {
            'sentiment_logits': sentiment_logits,
            'aspect_logits': aspect_logits,
            'attentions': outputs.attentions
        }


# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=4, learning_rate=2e-5):
    # Initialize optimizerxx
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training history
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    # Train the model
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')

        # Training
        model.train()
        train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            model.zero_grad()

            # Get model outputs (handle different return types)
            if isinstance(model, AspectBasedSentimentAnalysis):
                outputs = model(input_ids, attention_mask)
                logits = outputs['sentiment_logits']
            else:
                logits, _ = model(input_ids, attention_mask)

            # Calculate loss and backpropagate
            loss = criterion(logits, labels)
            loss.backward()

            # Clip gradients and update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        training_history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                if isinstance(model, AspectBasedSentimentAnalysis):
                    outputs = model(input_ids, attention_mask)
                    logits = outputs['sentiment_logits']
                else:
                    logits, _ = model(input_ids, attention_mask)

                # Calculate loss
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Get predictions
                _, preds = torch.max(logits, dim=1)

                # Store predictions and true labels
                val_preds.extend(preds.cpu().tolist())
                val_true_labels.extend(labels.cpu().tolist())

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(val_true_labels, val_preds)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_accuracy)

    return training_history


# Evaluation function
def evaluate_model(model, test_dataloader):
    model.eval()

    # Collect predictions and true labels
    all_preds = []
    all_true_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            if isinstance(model, AspectBasedSentimentAnalysis):
                outputs = model(input_ids, attention_mask)
                logits = outputs['sentiment_logits']
            else:
                logits, _ = model(input_ids, attention_mask)

            # Get predictions
            _, preds = torch.max(logits, dim=1)

            # Store predictions, true labels, and logits
            all_preds.extend(preds.cpu().tolist())
            all_true_labels.extend(labels.cpu().tolist())
            all_logits.extend(F.softmax(logits, dim=1).cpu().tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, all_preds, average='weighted'
    )

    # Calculate confusion matrix
    cm = confusion_matrix(all_true_labels, all_preds)

    # ROC curve (for binary classification)
    if len(set(all_true_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_true_labels, [logit[1] for logit in all_logits])
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    }


# Visualization functions
def plot_training_history(history):
    """Plot the training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('images/training_history.png')
    plt.close()


def plot_confusion_matrix(cm, classes):
    """Plot the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('images/confusion_matrix.png')
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot the ROC curve for binary classification."""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('images/roc_curve.png')
    plt.close()


# Attention visualization function
def visualize_attention(model, tokenizer, text, layer_index=-1, head_index=0):
    """Visualize attention for a given text input."""
    # Tokenize the input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get model predictions and attention weights
    model.eval()
    with torch.no_grad():
        if isinstance(model, AspectBasedSentimentAnalysis):
            outputs = model(input_ids, attention_mask)
            attention = outputs['attentions']
        else:
            _, attention = model(input_ids, attention_mask)

    # Extract attention weights from the specified layer and head
    if attention is not None:
        attention_weights = attention[layer_index][0, head_index].cpu().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        # Remove padding tokens for visualization
        valid_tokens = []
        for i, token in enumerate(tokens):
            if token == '[PAD]':
                continue
            if attention_mask[0, i] == 1:
                valid_tokens.append(token)

        # Adjust attention weights matrix to match valid tokens
        attention_weights = attention_weights[:len(valid_tokens), :len(valid_tokens)]

        # Create attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights,
                    xticklabels=valid_tokens,
                    yticklabels=valid_tokens,
                    cmap='viridis')
        plt.title(f'Attention Weights for Layer {layer_index}, Head {head_index}')
        plt.tight_layout()
        plt.savefig('images/attention_heatmap.png')
        plt.close()

        return attention_weights
    else:
        print("No attention weights available")
        return None


# def run_interactive_testing(model, tokenizer, device=None):
#     """
#     Interactive function to test the model on custom text input.
#
#     Args:
#         model: Trained sentiment analysis model
#         tokenizer: BERT tokenizer
#         device: PyTorch device (defaults to the model's device)
#     """
#     if device is None:
#         device = next(model.parameters()).device
#
#     # Set model to evaluation mode
#     model.eval()
#
#     print("\n" + "=" * 50)
#     print("INTERACTIVE SENTIMENT ANALYSIS TESTING")
#     print("=" * 50)
#     print("Type your text below or 'quit' to exit:")
#
#     while True:
#         # Get user input
#         user_input = input("\nEnter text to analyze: ")
#
#         if user_input.lower() == 'quit':
#             break
#
#         if not user_input.strip():
#             print("Please enter some text to analyze.")
#             continue
#
#         # Tokenize input
#         encoding = tokenizer.encode_plus(
#             user_input,
#             add_special_tokens=True,
#             max_length=128,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         # Move to device
#         input_ids = encoding['input_ids'].to(device)
#         attention_mask = encoding['attention_mask'].to(device)
#
#         # Get prediction
#         with torch.no_grad():
#             # Handle different model return types
#             if hasattr(model, 'forward') and 'AspectBasedSentimentAnalysis' in model.__class__.__name__:
#                 outputs = model(input_ids, attention_mask)
#                 logits = outputs['sentiment_logits']
#             else:
#                 logits, attention = model(input_ids, attention_mask)
#
#             # Convert logits to probabilities
#             probs = torch.nn.functional.softmax(logits, dim=1)
#
#             # Get prediction
#             _, prediction = torch.max(logits, dim=1)
#
#             # Get confidence
#             confidence = probs[0][prediction.item()].item() * 100
#
#         # Display results
#         sentiment = "Positive" if prediction.item() == 1 else "Negative"
#         print(f"\nSentiment: {sentiment}")
#         print(f"Confidence: {confidence:.2f}%")
#
#         # Display most important words (using attention if available)
#         if 'attention' in locals() and attention is not None:
#             try:
#                 # Get tokens
#                 tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#
#                 # Get attention from last layer, first head
#                 attn_weights = attention[-1][0, 0].cpu().numpy()
#
#                 # Filter out padding tokens
#                 valid_tokens = []
#                 attn_values = []
#
#                 for i, token in enumerate(tokens):
#                     if token == '[PAD]' or token == '[CLS]' or token == '[SEP]':
#                         continue
#                     if encoding['attention_mask'][0, i] == 1:
#                         valid_tokens.append(token)
#                         attn_values.append(attn_weights[i])
#
#                 # Get top 5 tokens with highest attention
#                 if len(valid_tokens) > 0:
#                     # Sort by attention value
#                     token_attention = sorted(zip(valid_tokens, attn_values),
#                                              key=lambda x: x[1], reverse=True)
#
#                     # Display top 5 or fewer
#                     top_count = min(5, len(token_attention))
#                     print("\nMost important words:")
#                     for token, attn in token_attention[:top_count]:
#                         print(f"  - {token}: {attn:.4f}")
#             except Exception as e:
#                 print(f"Could not analyze token importance: {e}")


# def run_sample_testing(model, tokenizer):
#     """Test the model with sample texts and print results."""
#     print("\n" + "=" * 50)
#     print("SAMPLE SENTIMENT ANALYSIS RESULTS")
#     print("=" * 50)
#
#     # Sample texts to try
#     samples = [
#         "This movie was amazing and I loved every minute of it!",
#         "The worst film I've ever seen, a complete waste of time.",
#         "Good acting but the story was boring and predictable.",
#         "I didn't expect much from this film.",
#         "The cinematography utilized advanced techniques."
#     ]
#
#     for sample in samples:
#         print(f"\nInput: {sample}")
#
#         # Tokenize
#         encoding = tokenizer.encode_plus(
#             sample,
#             add_special_tokens=True,
#             max_length=128,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )
#
#         # Move to device
#         device = next(model.parameters()).device
#         input_ids = encoding['input_ids'].to(device)
#         attention_mask = encoding['attention_mask'].to(device)
#
#         # Get prediction
#         with torch.no_grad():
#             # Handle different model return types
#             if hasattr(model, 'forward') and 'AspectBasedSentimentAnalysis' in model.__class__.__name__:
#                 outputs = model(input_ids, attention_mask)
#                 logits = outputs['sentiment_logits']
#             else:
#                 logits, _ = model(input_ids, attention_mask)
#
#             # Convert logits to probabilities
#             probs = torch.nn.functional.softmax(logits, dim=1)
#
#             # Get prediction
#             _, prediction = torch.max(logits, dim=1)
#
#             # Get confidence
#             confidence = probs[0][prediction.item()].item() * 100
#
#         # Display results
#         sentiment = "Positive" if prediction.item() == 1 else "Negative"
#         print(f"Sentiment: {sentiment}")
#         print(f"Confidence: {confidence:.2f}%")
#         print("-" * 40)

def run_sample_testing(model, tokenizer):
    """Test the model with sample texts and print results, with support for aspect-based analysis."""
    print("\n" + "=" * 50)
    print("SAMPLE SENTIMENT ANALYSIS RESULTS")
    print("=" * 50)

    # Sample texts to try with diverse aspect mentions
    samples = [
        "This movie was amazing and I loved every minute of it!",
        "The worst film I've ever seen, a complete waste of time.",
        "Good acting but the story was boring and predictable.",
        "The cinematography was beautiful but the dialogue was terrible.",
        "The characters were well-developed and the plot was engaging.",
        "Great special effects, but poor character development."
    ]

    # Common aspects in movie reviews
    movie_aspects = ["acting", "plot", "cinematography", "script", "direction",
                     "special effects", "dialogue", "soundtrack", "characters",
                     "story", "pacing", "editing"]

    is_aspect_model = hasattr(model, 'forward') and 'AspectBasedSentimentAnalysis' in model.__class__.__name__

    for sample in samples:
        print(f"\nInput: {sample}")

        # Tokenize
        encoding = tokenizer.encode_plus(
            sample,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        device = next(model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get prediction
        with torch.no_grad():
            # Handle different model return types
            if is_aspect_model:
                outputs = model(input_ids, attention_mask)
                sentiment_logits = outputs['sentiment_logits']
                aspect_logits = outputs['aspect_logits']

                # Convert sentiment logits to probabilities
                sentiment_probs = torch.nn.functional.softmax(sentiment_logits, dim=1)

                # Get sentiment prediction
                _, sentiment_prediction = torch.max(sentiment_logits, dim=1)

                # Get sentiment confidence
                sentiment_confidence = sentiment_probs[0][sentiment_prediction.item()].item() * 100

                # Get aspect predictions
                aspect_probs = torch.nn.functional.sigmoid(aspect_logits)
                aspect_predictions = (aspect_probs > 0.5).int()

                # Map aspect indices to aspect names (for simulation)
                # In a real implementation, you would have actual aspect labels
                predicted_aspects = []
                for i, is_present in enumerate(aspect_predictions[0]):
                    if is_present.item() == 1 and i < len(movie_aspects):
                        # Get aspect confidence
                        aspect_conf = aspect_probs[0][i].item() * 100
                        # Simulate aspect sentiment (could be part of the actual model output)
                        # Here we're simulating by checking if aspect words appear near positive/negative words
                        aspect_sentiment = "Positive"
                        aspect_words = movie_aspects[i].split()

                        # Simplified sentiment for each aspect based on closeness to sentiment words
                        if any(neg_word in sample.lower() for neg_word in
                               ["terrible", "poor", "boring", "waste", "worst"]):
                            if any(movie_aspects[i] in sample.lower() for aspect in aspect_words):
                                aspect_sentiment = "Negative"

                        predicted_aspects.append((movie_aspects[i], aspect_sentiment, aspect_conf))
            else:
                logits, _ = model(input_ids, attention_mask)

                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)

                # Get prediction
                _, prediction = torch.max(logits, dim=1)

                # Get confidence
                confidence = probs[0][prediction.item()].item() * 100

        # Display results
        if is_aspect_model:
            sentiment = "Positive" if sentiment_prediction.item() == 1 else "Negative"
            print(f"Overall Sentiment: {sentiment}")
            print(f"Confidence: {sentiment_confidence:.2f}%")

            if predicted_aspects:
                print("\nDetected Aspects:")
                for aspect, aspect_sentiment, aspect_conf in predicted_aspects:
                    print(f"  - {aspect}: {aspect_sentiment} (confidence: {aspect_conf:.2f}%)")
            else:
                print("No specific aspects detected.")
        else:
            sentiment = "Positive" if prediction.item() == 1 else "Negative"
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}%")

        print("-" * 40)


def run_interactive_testing_with_aspects(model, tokenizer, device=None):
    """Interactive function to test the model with support for aspect-based sentiment analysis."""
    if device is None:
        device = next(model.parameters()).device

    # Determine if this is an aspect-based model
    is_aspect_model = hasattr(model, 'forward') and 'AspectBasedSentimentAnalysis' in model.__class__.__name__

    # Common aspects in movie reviews (for aspect-based models)
    movie_aspects = ["acting", "plot", "cinematography", "script", "direction",
                     "special effects", "dialogue", "soundtrack", "characters",
                     "story", "pacing", "editing"]

    # Set model to evaluation mode
    model.eval()

    print("\n" + "=" * 50)
    print("INTERACTIVE SENTIMENT ANALYSIS TESTING")
    print("=" * 50)
    print("Type your text below or 'quit' to exit:")

    while True:
        # Get user input
        user_input = input("\nEnter text to analyze: ")

        if user_input.lower() == 'quit':
            break

        if not user_input.strip():
            print("Please enter some text to analyze.")
            continue

        # Tokenize input
        encoding = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get prediction
        with torch.no_grad():
            # Handle different model return types
            if is_aspect_model:
                outputs = model(input_ids, attention_mask)
                sentiment_logits = outputs['sentiment_logits']
                aspect_logits = outputs['aspect_logits']
                attention = outputs.get('attentions', None)

                # Convert sentiment logits to probabilities
                sentiment_probs = torch.nn.functional.softmax(sentiment_logits, dim=1)

                # Get sentiment prediction
                _, sentiment_prediction = torch.max(sentiment_logits, dim=1)

                # Get sentiment confidence
                sentiment_confidence = sentiment_probs[0][sentiment_prediction.item()].item() * 100

                # Get aspect predictions
                aspect_probs = torch.nn.functional.sigmoid(aspect_logits)
                aspect_predictions = (aspect_probs > 0.5).int()

                # Map aspect indices to aspect names
                predicted_aspects = []
                for i, is_present in enumerate(aspect_predictions[0]):
                    if is_present.item() == 1 and i < len(movie_aspects):
                        aspect_conf = aspect_probs[0][i].item() * 100

                        # Simulate aspect sentiment (could be part of model output)
                        aspect_sentiment = "Positive"
                        aspect_words = movie_aspects[i].split()

                        # Simplified sentiment for each aspect
                        if any(neg_word in user_input.lower() for neg_word in
                               ["terrible", "poor", "boring", "waste", "worst"]):
                            if any(movie_aspects[i] in user_input.lower() for aspect in aspect_words):
                                aspect_sentiment = "Negative"

                        predicted_aspects.append((movie_aspects[i], aspect_sentiment, aspect_conf))
            else:
                logits, attention = model(input_ids, attention_mask)

                # Convert logits to probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)

                # Get prediction
                _, prediction = torch.max(logits, dim=1)

                # Get confidence
                confidence = probs[0][prediction.item()].item() * 100

        # Display results
        if is_aspect_model:
            sentiment = "Positive" if sentiment_prediction.item() == 1 else "Negative"
            print(f"\nOverall Sentiment: {sentiment}")
            print(f"Confidence: {sentiment_confidence:.2f}%")

            if predicted_aspects:
                print("\nDetected Aspects:")
                for aspect, aspect_sentiment, aspect_conf in predicted_aspects:
                    print(f"  - {aspect}: {aspect_sentiment} (confidence: {aspect_conf:.2f}%)")
            else:
                print("No specific aspects detected.")
        else:
            sentiment = "Positive" if prediction.item() == 1 else "Negative"
            print(f"\nSentiment: {sentiment}")
            print(f"Confidence: {confidence:.2f}%")

        # Display most important words (using attention if available)
        if attention is not None:
            try:
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

                # Get attention from last layer, first head
                attn_weights = attention[-1][0, 0].cpu().numpy()

                # Filter out padding tokens
                valid_tokens = []
                attn_values = []

                for i, token in enumerate(tokens):
                    if token == '[PAD]' or token == '[CLS]' or token == '[SEP]':
                        continue
                    if encoding['attention_mask'][0, i] == 1:
                        valid_tokens.append(token)
                        attn_values.append(attn_weights[i])

                # Get top 5 tokens with highest attention
                if len(valid_tokens) > 0:
                    # Sort by attention value
                    token_attention = sorted(zip(valid_tokens, attn_values),
                                             key=lambda x: x[1], reverse=True)

                    # Display top 5 or fewer
                    top_count = min(5, len(token_attention))
                    print("\nMost important words:")
                    for token, attn in token_attention[:top_count]:
                        print(f"  - {token}: {attn:.4f}")
            except Exception as e:
                print(f"Could not analyze token importance: {e}")


def main():
    # Load datasets
    datasets = load_and_prepare_datasets()

    # Select dataset for training (can be changed)
    selected_dataset = "imdb"
    print(f"Selected dataset: {selected_dataset}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare_dataloader(
        datasets[selected_dataset], tokenizer, batch_size=16
    )

    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode text
    inputs = tokenizer("This is a test sentence", return_tensors='pt')
    outputs = model(**inputs, output_attentions=True)

    # Get tokens and attention
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attention = outputs.attentions

    # Visualize attention
    head_view(attention, tokens)

    # Initialize model (choose one of the three model architectures)
    model_type = "standard"  # Options: "standard", "hybrid", "aspect"

    if model_type == "standard":
        model = BERTSentimentClassifier(num_classes=2).to(device)
    elif model_type == "hybrid":
        model = HybridBERTLSTM(num_classes=2).to(device)
    elif model_type == "aspect":
        model = AspectBasedSentimentAnalysis(num_sentiment_classes=2).to(device)
    else:
        raise ValueError("Invalid model type")

    print(f"Model initialized: {model_type}")

    # Train model
    print("Starting training...")
    history = train_model(model, train_dataloader, val_dataloader, epochs=5)

    # Plot training history
    plot_training_history(history)

    # Evaluate model
    print("Evaluating model...")
    evaluation = evaluate_model(model, test_dataloader)

    # Print evaluation metrics
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1 Score: {evaluation['f1']:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(evaluation['confusion_matrix'], classes=['Negative', 'Positive'])

    # Plot ROC curve for binary classification
    if evaluation['roc']['fpr'] is not None:
        plot_roc_curve(
            evaluation['roc']['fpr'],
            evaluation['roc']['tpr'],
            evaluation['roc']['auc']
        )

    # Visualize attention weights for a sample text
    sample_text = "This movie was absolutely fantastic. The acting was superb."
    attention_weights = visualize_attention(model, tokenizer, sample_text)

    # save the model
    model_save_path = "standard_model_test.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\nTesting model with sample texts...")
    run_sample_testing(model, tokenizer)

    # Optional interactive testing
    print("\nWould you like to test the model interactively? (y/n)")
    response = input()
    if response.lower() == 'y':
        run_interactive_testing_with_aspects(model, tokenizer)

    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()