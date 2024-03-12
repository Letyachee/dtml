import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


def load_stock_data(data_dir, start_date, end_date):
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_data = []
    labels = []

    for stock_file in stock_files:
        data_path = os.path.join(data_dir, stock_file)
        data = pd.read_csv(data_path)
        data['date'] = pd.to_datetime(data['date'])
        filtered_data = data[(data['date'] >= start_date) & (data['date'] < end_date)]

        # Drop 'date' and 'label'
        features = filtered_data.drop(columns=['date', 'label']).values
        label = filtered_data['label'].values

        all_data.append(features)
        
        if not "SPY" in stock_file:
            labels.append(label)

    # lists to tensors
    features_tensor = torch.tensor(all_data, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    return features_tensor, labels_tensor



class StocksDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, seq_len):
        """
        features: Tensor of shape (num_stocks, seq_len, feature_dim)
        labels: Tensor of shape (num_stocks, seq_len)
        seq_len: The sequence length we want to use for generating batches
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.num_stocks = features.shape[0]
        self.total_seq_len = features.shape[1]

    def __len__(self):
        return (self.total_seq_len - self.seq_len) #+ 1

    def __getitem__(self, idx):
        """
        Returns a batch of features of size (batch_size, seq_len, num_stocks, feature_dim) and a batch of labels (batch_size, num_stocks-1)
        num_stock-1 in labels is due to the process of Multi-level context aggregation (we remove SPY from the features when it goes throgh DTML)
        """
        batch_features = self.features[:, idx:idx+self.seq_len, :]
        batch_labels = self.labels[:, idx+self.seq_len]  # Label for the last time step in the sequence
        return batch_features.permute(1, 0, 2), batch_labels # Rearrange dimensions to match (seq_len, num_stocks, feature_dim)


class StockDataModule(LightningDataModule):
    def __init__(self, data_dir, train_date, val_date, test_date, batch_size=32, seq_len=10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        self.train_date = train_date
        self.val_date = val_date
        self.test_date = test_date
        
    def setup(self, stage=None):
        # Load and preprocess data
        if stage == 'fit' or stage is None:
            train_features, train_labels = load_stock_data(self.data_dir, *self.train_date)
            self.train_dataset = StocksDataset(train_features, train_labels, self.seq_len)
            
            val_features, val_labels = load_stock_data(self.data_dir, *self.val_date)
            self.val_dataset = StocksDataset(val_features, val_labels, self.seq_len)
            
        if stage == 'test' or stage is None:
            test_features, test_labels = load_stock_data(self.data_dir, *self.test_date)
            self.test_dataset = StocksDataset(test_features, test_labels, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=11)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)