import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

class TradingSimulator:
    def __init__(self, model, data_dir, market_index_name, start_date, end_date, seq_len, best_stock_number=3, budget=10_000, fees=0.0, close_price_index = 3):
        self.model = model
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.seq_len = seq_len
        self.budget = budget
        self.fees = fees
        self.best_stock_number = best_stock_number
        self.close_price_index = close_price_index # Index of the columns 'Close' (close is actually df.Close.pct_change())
        self.market_index_name = market_index_name
        self.data = None  # This will hold the generated data tensor
        self.filtered_dates = None  # Store filtered dates from the dataset
        self.spy_data = None  # Store SPY stock data if available
        self.budget_dynamics = [budget]  # Initialize budget dynamics with the initial budget
        self.budget_dynamics_market = [budget]

    def load_and_process_data(self):
        """
        Loads data from CSV files in the specified directory and processes it
        into a tensor for the given date range. Also extracts and stores
        filtered dates and SPY stock data if available.
        """
        all_data = []
        stock_files = os.listdir(self.data_dir)

        for stock_file in stock_files:
            data_path = os.path.join(self.data_dir, stock_file)
            if os.path.isfile(data_path):
                data = pd.read_csv(data_path)
                data['date'] = pd.to_datetime(data['date'])
                filtered_data = data[(data['date'] >= self.start_date) & (data['date'] < self.end_date)]

                if self.filtered_dates is None:
                    self.filtered_dates = filtered_data['date']

                features = filtered_data.drop(columns=['date', 'label']).values
                all_data.append(features)

                if self.market_index_name in stock_file.upper():  # Assuming stock file names are case-insensitive
                    self.market_index_data = features

        if all_data:
            self.data = torch.tensor(all_data, dtype=torch.float).permute(1, 0, 2)
        else:
            self.data = torch.tensor([])

    def predict_stock_growth(self, t):
        """
        Predicts stock growth using the provided model at time t.

        Parameters:
        - t: The current time index.

        Returns:
        Probabilities of growth for each stock.
        """
        inputs = self.data[t:t + self.seq_len, :, :].unsqueeze(0)
        with torch.no_grad():
            logits = self.model(inputs)
        probs = torch.sigmoid(logits)
        return probs

    def calculate_budget_change(self):
        """
        Calculates the change in budget over time based on stock growth predictions.
        Dynamically updates the budget_dynamics list with new budget values, 
        reflecting the budget at the end of each timestep accurately.
        """
        for t in range(self.data.shape[0] - self.seq_len):
            # Predict stock growth probabilities at the current time index
            probabilities = self.predict_stock_growth(t)
            # Sort the probabilities and get indices of the top stocks
            top_stocks_indices = probabilities.argsort(descending=True).squeeze()[:self.best_stock_number]

            # Use the latest value in budget_dynamics for profit calculation
            current_budget = self.budget_dynamics[-1]

            budget_change = 0
            for stock_index in top_stocks_indices:
                # Calculate the percentage change in price for the selected stock
                price_change_percentage = self.get_close_price_change(stock_index, t + self.seq_len)
                profit = (current_budget / self.best_stock_number) * price_change_percentage / 100
                budget_change += profit

            # Calculate fees based on the current budget
            fees = self.calculate_fees(current_budget)
            updated_budget = current_budget + budget_change - fees
            self.budget_dynamics.append(updated_budget)

            # Market investing data calculation
            current_budget_market = self.budget_dynamics_market[-1]
            price_change_percentage_market = self.get_close_price_change_market(t + self.seq_len)
            profit_market = current_budget_market * price_change_percentage_market / 100
            self.budget_dynamics_market.append(profit_market + current_budget_market)



    def get_close_price_change(self, stock_index, t):
        """
        Calculates the percentage change in closing price for a specified stock and time index.

        Parameters:
        - stock_index: The index of the stock in the data.
        - t: The time index for calculating the price change.

        Returns:
        The percentage change in closing price between t and t + 1.
        """
        price_change = self.data[t, stock_index,  self.close_price_index]
    
        return price_change
    
    def get_close_price_change_market(self, t):
        price_change = self.market_index_data[t, self.close_price_index]
        return price_change
    
    def calculate_fees(self, transaction_vol):
        return transaction_vol * self.fees * 2 / 100 
            
    def setup(self):
        self.load_and_process_data()
        self.calculate_budget_change()
        
        
    def plot_budget_dynamics(self):
        """
        Plots the budget dynamics over time.
        """
        if len(self.budget_dynamics) != len(self.filtered_dates.iloc[self.seq_len - 1:]):
            print("Error: Date range and budget dynamics length do not match.")
            return

        plt.figure(figsize=(10, 6))
        # Model investing plot
        plt.plot(self.filtered_dates.iloc[self.seq_len - 1:].values, self.budget_dynamics, label='Model Investing', linestyle='-')
        # Market index investing plot
        plt.plot(self.filtered_dates.iloc[self.seq_len - 1:].values, self.budget_dynamics_market, label='Market Index Investing', linestyle='-')

        plt.title('Budget Dynamics Over Time')
        plt.xlabel('Date')
        plt.ylabel('Budget')
        plt.xticks(rotation=45)
        plt.legend()  
        plt.tight_layout()
        plt.show()