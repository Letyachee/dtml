import os
import pandas as pd
import numpy as np

def market_index(data_dir, market_index_name):
    stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for index, file_name in enumerate(stock_files):
        if market_index_name in file_name:
            return index

def stock_feat_num_dims(data_dir):
    stocks = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    file_path = os.path.join(data_dir, stocks[-1])
    feat_dim = pd.read_csv(file_path).shape[1] - 2 # - date and label
    num_stocks = len(stocks)
    
    return feat_dim, num_stocks


#########
def all_labels(data_dir):
    stocks = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    all_labels = []
    for s in stocks:
        file_path = os.path.join(data_dir, s)
        lab = pd.read_csv(file_path)['label'].values
        all_labels.extend(lab)
    
    return np.array(all_labels)

def calculate_class_ratio(data_dir):
    labels_array = all_labels(data_dir)
    zeros = np.sum(labels_array == 0)
    ones = np.sum(labels_array == 1)
    ratio = zeros / ones if ones > 0 else 0
    return ratio