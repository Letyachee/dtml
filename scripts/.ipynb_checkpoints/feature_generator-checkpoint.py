import pandas as pd
from pathlib import Path

# This from https://github.com/simonjisu/DTML-pytorch
def load_single_tick(p, pct=0.55):
    def longterm_trend(x, k):
        return (x.rolling(k).sum().div(k*x) - 1) * 100

    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])#, format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    if 'Unnamed' in df.columns:
        df.drop(columns=df.columns[7], inplace=True)
    if 'Original_Open' in df.columns:
        df.rename(columns={'Open': 'Adj Open', 'Original_Open': 'Open'}, inplace=True)
    # Open, High, Low
    z1 = (df.loc[:, ['Open', 'High', 'Low']].div(df['Close'], axis=0) - 1).rename(
        columns={'Open': 'open', 'High': 'high', 'Low': 'low'}) * 100
    # Close
    z2 = df[['Close']].pct_change().rename(columns={'Close': 'close'}) * 100
    # Adj Close
    z3 = df[['Adj Close']].pct_change().rename(columns={'Adj Close': 'adj_close'}) * 100

    z4 = []
    for k in [5, 10, 15, 20, 25, 30]:
        z4.append(df[['Adj Close']].apply(longterm_trend, k=k).rename(columns={'Adj Close': f'zd{k}'}))

    df_pct = pd.concat([df['Date'], z1, z2, z3] + z4, axis=1).rename(columns={'Date': 'date'})
    cols_max = df_pct.columns[df_pct.isnull().sum() == df_pct.isnull().sum().max()]
    df_pct = df_pct.loc[~df_pct[cols_max].isnull().values, :]

    # from https://arxiv.org/abs/1810.09936
    # Examples with movement percent ≥ 0.55% and ≤ −0.5% are 
    # identified as positive and negative examples, respectively
    df_pct['label'] = (df_pct['close'] >= pct).astype(int)
    return df_pct



def generate_and_save_features_files(raw_data_dir, features_data_dir, pct=0.55, file_pattern = "*.csv"):
    # Create the destination folder if it doesn't exist
    features_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a list of all files matching the pattern in the source folder
    source_files = list(raw_data_dir.glob(file_pattern))
    
    # Loop through each file in the list
    for file_path in source_files:
        # Apply the processing function to the file
        df_processed = load_single_tick(file_path, pct)
        
        # Save the processed data to a new file in the destination folder
        # The file name remains the same, but the path changes to the destination folder
        destination_file_path = features_data_dir / file_path.name
        df_processed.to_csv(destination_file_path, index=False)