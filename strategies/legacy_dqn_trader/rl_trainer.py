# rl_trainer.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mplfinance as mpf
import matplotlib.pyplot as plt
import logging
import time
from network import DQN  # Import the DQN model from network.py
from experience_store import ExperienceStore  # Import ExperienceStore
from process_metadata import MetadataProcessor  # Import MetadataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_model(model, target_model, metadata_processor, optimizer, experience_store, device, batch_size, gamma,
                num_episodes, model_save_path, train_metadata_file):
    """
    Trains the DQN model in batches.

    Args:
        model (nn.Module): The DQN model.
        target_model (nn.Module): The target DQN model.
        metadata_processor (MetadataProcessor): The MetadataProcessor instance.
        optimizer (optim.Optimizer): Optimizer.
        experience_store (ExperienceStore): The experience store.
        device (str): Device to train on ('cuda' or 'cpu').
        batch_size (int): Batch size for training.
        gamma (float): Discount factor.
        num_episodes (int): Number of episodes to train for.
        model_save_path (str): The path to save the model weights.
        train_metadata_file (str): The path to the train metadata file.
    """
    criterion = nn.MSELoss()
    logging.info("Starting training...")
    for episode in tqdm(range(num_episodes)):
        logging.info(f"Starting episode {episode + 1}/{num_episodes}")
        episode_start_time = time.time()

        # Process metadata for each episode
        metadata_processing_start_time = time.time()
        processed_metadata = metadata_processor.process_metadata_for_episode()
        processed_metadata.to_csv(train_metadata_file, index=False)
        metadata_processing_end_time = time.time()
        logging.info(f"Metadata processing time for episode {episode + 1}: {metadata_processing_end_time - metadata_processing_start_time:.4f} seconds")

        if processed_metadata is None:
            logging.warning(f"Initial state is None, skipping episode {episode + 1}.")
            continue

        total_reward = 0
        episode_experiences = []
        experience_store_adding_start_time = time.time()

        for i in range(len(processed_metadata)):
            logging.debug(f"Processing row {i + 1}/{len(processed_metadata)} in episode {episode + 1}")
            state_image = torch.tensor(processed_metadata.loc[i, 'image_tensor']).to(device)
            state_numerical_data = torch.tensor(processed_metadata.loc[i, 'numerical_data_normalized']).to(device)
            action = processed_metadata.loc[i, 'action']
            reward = processed_metadata.loc[i, 'reward']
            target_q = processed_metadata.loc[i, 'target_q']

            # next state is in the next row
            if i < len(processed_metadata) - 1:
                next_state_image = torch.tensor(processed_metadata.loc[i + 1, 'image_tensor']).to(device)
                next_state_numerical_data = torch.tensor(processed_metadata.loc[i + 1, 'numerical_data_normalized']).to(device)
                done = False
            else:
                next_state_image = torch.empty(1, 320, 320).to(device)
                next_state_numerical_data = torch.empty(1, 17).to(device)
                done = True

            total_reward += reward

            # Save experience to a list to add all of them to the store after going through one episode
            episode_experiences.append(((state_image, state_numerical_data), action, reward,
                                        (next_state_image, next_state_numerical_data), done))

        # Push all episode experiences to the memory at once
        logging.debug(f"Pushing {len(episode_experiences)} experiences to memory for episode {episode + 1}")
        for experience in episode_experiences:
            experience_store.push(*experience)
        experience_store_adding_end_time = time.time()
        logging.info(f"Adding to experience store time for episode {episode + 1}: {experience_store_adding_end_time - experience_store_adding_start_time:.4f} seconds")

        # Training step (now outside of the loop over rows)
        if len(experience_store) >= batch_size:
            num_training_batches = len(experience_store) // batch_size
            logging.info(f"Starting {num_training_batches} training batches for episode {episode + 1}")
            for batch_num in range(num_training_batches):
                batch_training_start_time = time.time()

                # get training data
                batch_data = experience_store.sample(batch_size)
                if batch_data is None:
                    logging.warning(f"Skipping training batch due to insufficient data in episode {episode + 1}")
                    continue  # skip this loop if we have not enough data.

                state_images, state_numerical_data, action_batch, reward_batch, next_state_images, next_state_numerical_data, done_batch = batch_data

                # Check for empty tensors after sampling. This can happen if there are not enough transitions in the buffer
                if len(state_images) == 0 or len(state_numerical_data) == 0 or len(next_state_images) == 0 or len(
                        next_state_numerical_data) == 0:
                    logging.warning(f"Skipping training batch due to empty tensors in episode {episode + 1}")
                    continue  # skip this loop if we have empty tensors.

                # If we have an empty tensors we do not want to process them, so skip this loop.
                if len(state_images) != batch_size or len(next_state_images) != batch_size:
                    logging.warning(f"Skipping training batch due to incorrect batch size in episode {episode + 1}")
                    continue

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
                    target_q_values = reward_batch + (gamma * next_q_values * (~done_batch))  # (~done_batch) is same as (1- done_batch)

                # get q values
                q_values = model(state_images, state_numerical_data).gather(1, action_batch.unsqueeze(1)).squeeze(1)

                # back prop
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_training_end_time = time.time()
                logging.info(
                    f"Finished training batch {batch_num + 1} for episode {episode + 1}: {batch_training_end_time - batch_training_start_time:.4f} seconds")
            logging.info(f"Finished training batches for episode {episode + 1}")

        # update target network
        if episode % 10 == 0:
            logging.info(f"Updating target model at episode {episode + 1}")
            target_model.load_state_dict(model.state_dict())

        logging.info(f"Finished episode {episode + 1}/{num_episodes}, total reward: {total_reward}")

        # Save model weights after each episode
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model weights saved to {model_save_path}")

        episode_end_time = time.time()
        logging.info(f"Total time for episode {episode + 1}: {episode_end_time - episode_start_time:.4f} seconds")


def test_model(model, metadata_processor, device, test_transactions_file, num_episodes=1):
    """
    Tests the trained model.

    Args:
        model (nn.Module): The trained DQN model.
        metadata_processor (MetadataProcessor): The MetadataProcessor instance.
        device (str): Device to test on ('cuda' or 'cpu').
        test_transactions_file (str): Path to save the test transactions CSV file.
        num_episodes (int): Number of episodes to test for.
    """
    logging.info("Starting testing...")
    model.eval()  # Set the model to evaluation mode
    total_test_reward = 0
    transactions = []

    with torch.no_grad():
        for episode in tqdm(range(num_episodes)):
            logging.info(f"Starting test episode {episode + 1}/{num_episodes}")
            processed_metadata = metadata_processor.process_metadata_for_episode()
            if processed_metadata is None:
                logging.warning(f"Initial state is None, skipping test episode {episode + 1}.")
                continue
            episode_reward = 0
            for i in range(len(processed_metadata)):
                state_image = torch.tensor(processed_metadata.loc[i, 'image_tensor']).to(device)
                state_numerical_data = torch.tensor(processed_metadata.loc[i, 'numerical_data_normalized']).to(device)
                action = processed_metadata.loc[i, 'action']
                reward = processed_metadata.loc[i, 'reward']

                # Log transaction details
                if action == 1 or action == 2:  # Buy or Sell action
                    logging.info(
                        f"Test episode {episode + 1}: Transaction at row {i + 1}, Action: {'Buy' if action == 1 else 'Sell'}, Price: {processed_metadata.iloc[i]['Close']}, Reward: {reward}")
                    transactions.append({
                        "data_index": i,
                        "current_price": processed_metadata.iloc[i]["Close"],  # Assuming "Close" is available in your metadata
                        "action": "Buy" if action == 1 else "Sell",
                        "reward": reward
                    })

                episode_reward += reward

            total_test_reward += episode_reward
            logging.info(f"Finished test episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")

    avg_test_reward = total_test_reward / num_episodes if num_episodes > 0 else 0
    logging.info(f"Average Test Reward over {num_episodes} episodes: {avg_test_reward}")

    # Save transactions to CSV
    if transactions:
        transactions_df = pd.DataFrame(transactions)
        transactions_df.to_csv(test_transactions_file, index=False)
        logging.info(f"Test transactions saved to {test_transactions_file}")

        # Plot buy/sell data
        plot_buy_sell(processed_metadata, transactions_df)
    else:
        logging.info("No transactions found during testing.")


def plot_buy_sell(metadata, transactions_df):
    """
    Plots the buy/sell markers on a candlestick chart.

    Args:
        metadata (pd.DataFrame): The test metadata.
        transactions_df (list): DataFrame of the transactions.
    """
    # Extract the OHLC data
    ohlc_data = metadata[["Open", "High", "Low", "Close"]].copy()
    ohlc_data.index = pd.to_datetime(metadata.index)

    # Create buy/sell lists
    buy_markers = []
    sell_markers = []
    for _, row in transactions_df.iterrows():
        data_index = row["data_index"]
        current_price = row["current_price"]
        action = row["action"]
        if action == "Buy":  # Buy
            buy_markers.append((metadata.iloc[data_index].name, current_price))
        elif action == "Sell":  # Sell
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
    logging.info(f"Using device: {device}")

    # Define paths
    metadata_file = 'strategies/RL/data/metadata.csv'
    image_dir = '/Users/kanchannannavare/Documents/RL_dataset/data/charts'
    model_save_path = 'strategies/RL/data/model_weights.pth'  # Path to save/load model weights
    train_metadata_file = 'strategies/RL/data/train_metadata.csv'
    test_metadata_file = 'strategies/RL/data/test_metadata.csv'
    test_transactions_file = 'strategies/RL/data/test_transactions.csv'

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
    transaction_cost_percent = 0.001
    initial_balance = 10000
    test_size = 0.2  # 20% of the data for testing
    replay_buffer_capacity = 10000 

    # --- Data Preprocessing ---
    logging.info("Starting data preprocessing...")
    metadata_df = pd.read_csv(metadata_file)

    # Split data into train and test
    train_df, test_df = train_test_split(metadata_df, test_size=test_size, shuffle=False)

    # Save the processed metadata
    train_df.to_csv(train_metadata_file, index=False)
    test_df.to_csv(test_metadata_file, index=False)
    logging.info(f"Data preprocessing completed. Train data saved to {train_metadata_file}, Test data saved to {test_metadata_file}")
    # --- End Data Preprocessing ---

    # Create metadata processor
    logging.info("Creating MetadataProcessors...")
    train_metadata_processor = MetadataProcessor(train_metadata_file, image_dir, model_save_path, device=device,
                                               transaction_cost_percent=transaction_cost_percent,
                                               epsilon_start=epsilon_start,
                                               epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                                                 test_mode=False)
    test_metadata_processor = MetadataProcessor(test_metadata_file, image_dir, model_save_path, device=device,
                                              transaction_cost_percent=transaction_cost_percent,
                                              epsilon_start=epsilon_start,
                                              epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                                                test_mode=True)

    # Create model, target model, and optimizer
    logging.info("Creating model, target model, and optimizer...")
    model = DQN(num_actions).to(device)
    target_model = DQN(num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load model weights if they exist
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info(f"Model weights loaded from {model_save_path}")

    # Initialize ExperienceStore instead of ReplayBuffer
    logging.info("Initializing ExperienceStore...")
    experience_store_file = "strategies/RL/data/experience_store.pkl"  # Choose a file name
    experience_store = ExperienceStore(replay_buffer_capacity, experience_store_file)

    train_model(model, target_model, train_metadata_processor, optimizer, experience_store, device, batch_size, gamma,
                num_episodes, model_save_path, train_metadata_file)
    
if __name__ == "__main__":
    main()