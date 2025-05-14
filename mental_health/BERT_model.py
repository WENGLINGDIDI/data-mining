import pandas as pd
import numpy as np
import re
import string
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff
import warnings

warnings.filterwarnings('ignore')


# Data Cleaning Function (from the notebook)
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower()  # Convert to lowercase

    # Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove newlines
    text = text.replace("\n", " ")

    # Remove words containing numbers
    text = re.sub(r'\b\w*\d\w*\b', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Load and preprocess data
def load_and_clean_data(file_path='Combined Data.csv'):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Fill missing values with empty string
    df.fillna('', inplace=True)

    # Apply text preprocessing
    df['cleaned_comment'] = df['statement'].apply(preprocess_text)

    return df


# Prepare data for BERT
def prepare_bert_data(df, max_length=128):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['status'])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        df['cleaned_comment'], y_encoded, random_state=42, test_size=0.2
    )

    # Tokenize data
    def tokenize_data(texts):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    # Tokenize training and testing data
    x_train_ids, x_train_masks = tokenize_data(x_train)
    x_test_ids, x_test_masks = tokenize_data(x_test)

    # Convert labels to tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (x_train_ids, x_train_masks, y_train_tensor,
            x_test_ids, x_test_masks, y_test_tensor, label_encoder)


# Train and evaluate BERT model
def train_and_evaluate_bert(train_data, test_data, num_labels, epochs=3, batch_size=16):
    # Unpack data
    x_train_ids, x_train_masks, y_train_tensor = train_data
    x_test_ids, x_test_masks, y_test_tensor = test_data

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train_ids, x_train_masks, y_train_tensor)
    test_dataset = TensorDataset(x_test_ids, x_test_masks, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"BERT Accuracy: {accuracy:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm_fig = ff.create_annotated_heatmap(
        z=cm,
        x=list(label_encoder.classes_),
        y=list(label_encoder.classes_),
        annotation_text=cm,
        colorscale='Viridis'
    )
    cm_fig.update_layout(title='Confusion Matrix for BERT')
    cm_fig.show()

    return accuracy


# Main execution
if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data()

    # Prepare data for BERT
    (x_train_ids, x_train_masks, y_train_tensor,
     x_test_ids, x_test_masks, y_test_tensor, label_encoder) = prepare_bert_data(df)

    # Train and evaluate
    train_data = (x_train_ids, x_train_masks, y_train_tensor)
    test_data = (x_test_ids, x_test_masks, y_test_tensor)
    accuracy = train_and_evaluate_bert(
        train_data,
        test_data,
        num_labels=len(label_encoder.classes_)
    )
