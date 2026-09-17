# environment.py
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler

class TradingEnvironment:
    """
    Trading environment for the reinforcement learning agent.
    """

    def __init__(self, metadata_file, image_dir, initial_balance=10000, transaction_cost_percent=0.001, lookback_window=100, hold_reward_factor=0.0001):
        """
        Initializes the TradingEnvironment.

        Args:
            metadata_file (str): Path to the metadata CSV file.
            image_dir (str): Directory containing the images.
            initial_balance (float): Starting balance for the agent.
            transaction_cost_percent (float): Transaction cost as a percentage of the trade value.
            lookback_window (int): how much data is in one image.
            hold_reward_factor (float): Factor to scale the hold reward.
        """
        self.metadata = pd.read_csv(metadata_file)
        self.image_dir = image_dir
        self.initial_balance = initial_balance
        self.transaction_cost_percent = transaction_cost_percent
        self.lookback_window = lookback_window
        self.current_balance = initial_balance
        self.current_position = 0  # 0: no position, 1: long position
        self.data_index = 0
        self.done = False
        self.entry_price = 0  # To track the price when we buy
        self.shares_held = 0
        self.hold_reward_factor = hold_reward_factor # new variable for holding position

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
        
        # Create a separate scaler for 'Close' prices
        self.close_price_scaler = MinMaxScaler()
        # we will get the unscaled data from the 'metadata.csv' file.
        self.unscaled_close_prices = pd.read_csv(metadata_file)['Close'].values.reshape(-1, 1)
        self.close_price_scaler.fit(self.unscaled_close_prices)
        
        # Calculate the number of data points and how much data is in the entire dataframe
        self.num_data_points = len(self.metadata)
        self.total_data_points = self.num_data_points - self.lookback_window
        print("total data points in environment", self.total_data_points)

        self.images = {}  # Store pre-loaded images here
        self._preload_images()

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            tuple: (state, info)
        """
        self.current_balance = self.initial_balance
        self.current_position = 0
        self.data_index = 0 # reset the data index to the start
        self.done = False
        self.entry_price = 0
        self.shares_held = 0
        return self._get_state(), {}
    
    def _preload_images(self):
        """Loads and transforms all images at initialization."""
        print("Pre-loading images...")
        for index, row in self.metadata.iterrows():
            img_name = os.path.join(self.image_dir, row["image_path"])
            try:
                image = Image.open(img_name).convert("L")
                if self.transform:
                    image = self.transform(image)
                self.images[index] = image
            except Exception as e:
                print(f"Error pre-loading image {img_name}: {e}")
                self.images[index] = None  # Or handle the error as you see fit
        
        return


    def _get_state(self):
        """
         Gets the current state (faster version).
        """
        if self.data_index >= self.total_data_points:
            self.done = True
            return None, None
        
       # Retrieve pre-loaded image
        image = self.images.get(self.data_index)  # Get the pre-loaded image
        if image is None:
            print("Image is None")
            self.done = True
            return None, None

        # Load numerical data
        numerical_data = self.metadata.iloc[self.data_index, 2:].values.astype('float32')
        
        # Check the shape and adjust if necessary
        if numerical_data.shape[0] != 17:
            print("Error: expected 17 numerical features, received", numerical_data.shape[0])
            self.done = True
            return None, None

        numerical_data = torch.from_numpy(numerical_data)
        return image, numerical_data

    def step(self, action):
        """
        Takes a step in the environment based on the given action.

        Args:
            action (int): Action to take (0: hold, 1: buy, 2: sell).

        Returns:
            tuple: (next_state, reward, done, truncated, info)
        """
        if self.done:
            raise Exception("Environment is done, reset first")

        # Get current unnormalized price
        current_price = self.unscaled_close_prices[self.data_index][0]
        print("current ", current_price, self.current_balance, self.current_position)
        last_balance = self.current_balance
        reward = 0 # initialize reward to zero
        
        # --- Action Validation ---
        # Prevent selling when there's no position
        if action == 2 and self.current_position == 0:
            print("Invalid action: Cannot sell without a long position.")
            action = 0 # force the action to hold.
            reward = -0.1 # negative reward for invalid action.
        # Prevent buying when there is an existing long position
        elif action == 1 and self.current_position == 1:
            print("Invalid action: Cannot buy while already in a long position.")
            action = 0 # force action to hold
            reward = -0.1 # negative reward for invalid action.

        # Handle actions
        transaction_cost = 0
        if action == 1 and self.current_position == 0:  # Buy
            # Calculate transaction cost
            
            if self.current_balance > 0:
              # buy as many shares as possible
              self.shares_held = (self.current_balance * (1 - self.transaction_cost_percent) )/ current_price
              self.entry_price = current_price
              transaction_cost = self.current_balance * self.transaction_cost_percent
              self.current_balance -= transaction_cost
              self.current_balance -= self.shares_held * current_price
              self.current_position = 1
              print(f"bought at: {self.entry_price}, shares:{self.shares_held}")
            #   reward += 0.1 # positive reward for buying

            else:
                print("insufficient balance to buy")
                action = 0
                reward = -0.1 # negative reward for insufficient balance


        elif action == 2 and self.current_position == 1:  # Sell
            # Calculate transaction cost
            if self.shares_held > 0:
                transaction_cost = (self.shares_held * current_price) * self.transaction_cost_percent
                self.current_balance += (self.shares_held * current_price) * (1 - self.transaction_cost_percent)
                
                # Calculate reward based on the selling price compared to the entry price
                reward += (current_price - self.entry_price) * self.shares_held # Calculate profit/loss
                
                print(f"sold at: {current_price}, profit/loss: {reward}")
                
                self.entry_price = 0  # Reset entry price after selling
                self.shares_held = 0
                self.current_position = 0
            else:
              print("no shares to sell")
              action = 0
              reward = -0.1 # negative reward for no shares to sell
        
        # Calculate reward for holding position
        elif self.current_position == 1 and action == 0: # if we are holding a long position
            price_diff = current_price - self.entry_price
            reward += price_diff * self.shares_held * self.hold_reward_factor # calculate if its positive or negative

        print("reward", reward)
        # Move to the next state
        self.data_index += 1
        next_state = self._get_state()
        
        # if the state is none, set done to true
        if next_state[0] is None:
            self.done = True

        # done is true if we have gone through all the data
        truncated = False # add truncated in the future
        
        # Info for debug
        info = {
            "balance": self.current_balance,
            "position": self.current_position,
            "data_index": self.data_index,
            "current_price": current_price,
            "reward": reward,
            "action": action # add the action to the info.
        }

        return next_state[0], next_state[1], reward, self.done, truncated, info
