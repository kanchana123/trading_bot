# rl_trainer.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import mplfinance as mpf
import matplotlib.pyplot as plt
from environment import TradingEnvironment  # Import the environment
from network import DQN  # Import the DQN model from network.py
from sklearn.preprocessing import MinMaxScaler
from experience_store import ExperienceStore # Import ExperienceStore

def train_model(model, target_model, env, optimizer, experience_store, device, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes, model_save_path):
    """
    Trains the DQN model.

    Args:
        model (nn.Module): The DQN model.
        target_model (nn.Module): The target DQN model.
        env (TradingEnvironment): The trading environment.
        optimizer (optim.Optimizer): Optimizer.
        experience_store (ExperienceStore): The experience store.
        device (str): Device to train on ('cuda' or 'cpu').
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        epsilon_start (float): Starting epsilon for epsilon-greedy.
        epsilon_end (float): Ending epsilon for epsilon-greedy.
        epsilon_decay (float): Epsilon decay rate.
        num_episodes (int): number of episodes to train for.
        model_save_path (str): The path to save the model weights.
    """
    epsilon = epsilon_start
    criterion = nn.MSELoss()
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        
        # Check if initial state is None
        if state[0] is None:
           print("Initial state is None, skipping episode.")
           continue
        
        done = False
        total_reward = 0

        while not done:
            state_image, state_numerical_data = state  # unpack the state here, inside the loop

            # choose action
            if random.random() > epsilon:
                with torch.no_grad():
                    q_values = model(state_image.unsqueeze(0).to(device), state_numerical_data.unsqueeze(0).to(device))
                    action = q_values.argmax().item()
            else:
                action = random.randrange(3)
            # take action
            next_state_image, next_state_numerical_data, reward, done, truncated, info = env.step(action)
            action = info["action"]

            # save to memory
            experience_store.push((state_image, state_numerical_data), action, reward, (next_state_image, next_state_numerical_data), done)
            
            total_reward += reward
            
            # update the states
            state = (next_state_image, next_state_numerical_data)

            if len(experience_store) >= batch_size:
                # get training data
                batch_data = experience_store.sample(batch_size)
                if batch_data is None:
                    continue # skip this loop if we have not enough data.

                state_images, state_numerical_data, action_batch, reward_batch, next_state_images, next_state_numerical_data, done_batch = batch_data

                
                # Check for empty tensors after sampling. This can happen if there are not enough transitions in the buffer
                if len(state_images) == 0 or len(state_numerical_data) == 0 or len(next_state_images) == 0 or len(next_state_numerical_data) == 0:
                   continue # skip this loop if we have empty tensors.

                state_images = state_images.to(device)
                state_numerical_data = state_numerical_data.to(device)
                next_state_images = next_state_images.to(device)
                next_state_numerical_data = next_state_numerical_data.to(device)
                action_batch = action_batch.to(device)
                reward_batch = reward_batch.to(device)
                done_batch = done_batch.to(device)
                
                # calculate target values
                with torch.no_grad():
                    next_q_values = target_model(next_state_images, next_state_numerical_data).max(1)[0]
                    
                    if len(next_q_values) != len(done_batch):
                        target_q_values = reward_batch
                    else:
                        target_q_values = reward_batch + (gamma * next_q_values * (~done_batch)) # (~done_batch) is same as (1- done_batch)

                # get q values
                q_values = model(state_images, state_numerical_data).gather(1, action_batch.unsqueeze(1)).squeeze(1)

                # back prop
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #update target network
        if episode % 100 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"episode: {episode+1}/{num_episodes}, total reward: {total_reward}, epsilon: {epsilon}")
        
        # Save model weights after each episode
        torch.save(model.state_dict(), model_save_path)
        print(f"Model weights saved to {model_save_path}")

def test_model(model, env, device, num_episodes=10):
    """
    Tests the trained model.

    Args:
        model (nn.Module): The trained DQN model.
        env (TradingEnvironment): The trading environment.
        device (str): Device to test on ('cuda' or 'cpu').
        num_episodes (int): number of episodes to test for.
    """
    model.eval()  # Set the model to evaluation mode
    total_test_reward = 0
    buy_sell_data = [] # list for buy sell data
    
    with torch.no_grad():
        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            
            # Check if initial state is None
            if state[0] is None:
                print("Initial state is None, skipping test episode.")
                continue

            done = False
            episode_reward = 0
            while not done:
                state_image, state_numerical_data = state
                q_values = model(state_image.unsqueeze(0).to(device), state_numerical_data.unsqueeze(0).to(device))
                action = q_values.argmax().item()

                next_state_image, next_state_numerical_data, reward, done, truncated, info = env.step(action)
                
                # Append buy/sell info
                if info["action"] == 1 or info["action"] == 2: # check for info["action"]
                    buy_sell_data.append([info["data_index"], info["current_price"], info["action"]]) # check for info["action"]
                
                episode_reward += reward
                state = (next_state_image, next_state_numerical_data)
            
            total_test_reward += episode_reward
            print(f"Test Episode: {episode+1}/{num_episodes}, Total Reward: {episode_reward}")

    avg_test_reward = total_test_reward / num_episodes if num_episodes > 0 else 0
    print(f"Average Test Reward over {num_episodes} episodes: {avg_test_reward}")
    
    # Plot buy/sell data
    if len(buy_sell_data) > 0:
        plot_buy_sell(env.metadata, buy_sell_data)
    else:
        print("No buy/sell data found to plot")

def plot_buy_sell(metadata, buy_sell_data):
    """
    Plots the buy/sell markers on a candlestick chart.

    Args:
        metadata (pd.DataFrame): The test metadata.
        buy_sell_data (list): List of [data_index, current_price, action].
    """
    # Extract the OHLC data
    ohlc_data = metadata[["Open", "High", "Low", "Close"]].copy()
    ohlc_data.index = pd.to_datetime(metadata.index)
    
    # Create buy/sell lists
    buy_markers = []
    sell_markers = []
    for data_index, current_price, action in buy_sell_data:
        if action == 1:  # Buy
            buy_markers.append((metadata.iloc[data_index].name, current_price))
        elif action == 2:  # Sell
            sell_markers.append((metadata.iloc[data_index].name, current_price))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the candlestick chart
    mpf.plot(ohlc_data, type="candle", ax=ax, style="yahoo", xrotation=0)

    # Add buy markers
    buy_times, buy_prices = zip(*buy_markers) if buy_markers else ([], [])
    ax.scatter(buy_times, buy_prices, marker="^", color="green", label="Buy", s=100)

    # Add sell markers
    sell_times, sell_prices = zip(*sell_markers) if sell_markers else ([], [])
    ax.scatter(sell_times, sell_prices, marker="v", color="red", label="Sell", s=100)

    ax.legend()
    plt.show()

def main():
    """Main function to load data, create model, and train/evaluate."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths
    metadata_file = 'strategies/RL/data/metadata.csv'
    image_dir = '/Users/kanchannannavare/Documents/RL_dataset/data/charts'
    model_save_path = 'strategies/RL/data/model_weights.pth'  # Path to save/load model weights

    # Hyperparameters
    num_episodes = 10
    batch_size = 32
    learning_rate = 0.001
    num_actions = 3
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    replay_buffer_capacity = 10000
    lookback_window=75
    transaction_cost_percent = 0.001
    initial_balance = 10000
    test_size = 0.2  # 20% of the data for testing

    # --- Data Preprocessing ---
    metadata_df = pd.read_csv(metadata_file).head(100)
    print(metadata_df.shape)
    # Add 'time' column
    metadata_df['time'] = metadata_df['timestamp'].apply(lambda x: int(x.split('_')[1].replace('-', '')))

   
    # Split data into train and test
    train_df, test_df = train_test_split(metadata_df, test_size=test_size, shuffle=False)

    # Save the processed metadata
    train_df.to_csv('strategies/RL/data/train_metadata.csv', index=False)
    test_df.to_csv('strategies/RL/data/test_metadata.csv', index=False)

    # --- End Data Preprocessing ---

    # Create environment
    train_env = TradingEnvironment('strategies/RL/data/train_metadata.csv', image_dir, initial_balance=initial_balance, transaction_cost_percent=transaction_cost_percent, lookback_window=lookback_window)
    test_env = TradingEnvironment('strategies/RL/data/test_metadata.csv', image_dir, initial_balance=initial_balance, transaction_cost_percent=transaction_cost_percent, lookback_window=lookback_window)

    # Normalize numerical data
    numerical_cols = ["Open", "High", "Low", "Close", "Volume", "SMA", "Upper", "Lower", "Open_HA", "High_HA", "Low_HA", "Close_HA", "Open_R", "High_R", "Low_R", "Close_R", "time"]
    scaler = MinMaxScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    test_df[numerical_cols] = scaler.fit_transform(test_df[numerical_cols])

    # Create model, loss function, and optimizer
    model = DQN(num_actions).to(device)
    target_model = DQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load model weights if they exist
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Model weights loaded from {model_save_path}")
    else:
        print("No saved model weights found. Starting with a new model.")
        
    # Initialize ExperienceStore instead of ReplayBuffer
    experience_store_file = "strategies/RL/data/experience_store.pkl"  # Choose a file name
    experience_store = ExperienceStore(replay_buffer_capacity, experience_store_file)
    
    # Train the model
    train_model(model, target_model, train_env, optimizer, experience_store, device, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes, model_save_path)
    
    # Test the model
    test_model(model, test_env, device, num_episodes=10)

if __name__ == "__main__":
    main()
