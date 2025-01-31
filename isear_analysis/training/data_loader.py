from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import torch
class CustomDataset(Dataset):
    """
    Custom dataset class for tokenizing text data.

    Attributes:
    - dataset: DataFrame loaded from CSV
    - tokenizer: Tokenizer for text processing
    - max_length: Maximum token length
    """
    
    def __init__(self, dataset_path, text_column_name, label_column_name, tokenizer_name, max_length=512):
        self.label_encoder = LabelEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_column = text_column_name
        self.label_column = label_column_name
        self.max_length = max_length
        self.dataset = self.load_dataset(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.iloc[idx][self.text_column]
        label = self.dataset.iloc[idx][self.label_column]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(label), dtype=torch.long)  # Convert label to tensor
        }

    def load_dataset(self, dataset_path):
        """
        Load dataset from a CSV file.

        Args:
        - dataset_path: str

        Returns:
        - Pandas DataFrame
        """
        try:
            df = pd.read_csv(dataset_path)
            df[self.label_column] = self.label_encoder.fit_transform(df[self.label_column])
            return df
        except FileNotFoundError:
            print(f"Data not found at path: {dataset_path}")
            return pd.DataFrame()  # Return empty DataFrame to prevent errors