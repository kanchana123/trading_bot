# # rl_trainer.py
# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# import random
# from environment import TradingEnvironment  # Import the environment
# from network import DQN  # Import the DQN model from network.py
# from sklearn.preprocessing import MinMaxScaler

# class ReplayBuffer:
#     """Replay buffer for storing and sampling experiences."""

#     def __init__(self, capacity):
#         """Initializes the ReplayBuffer.

#         Args:
#             capacity (int): Maximum number of experiences to store.
#         """
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0

#     def push(self, state, action, reward, next_state, done):
#         """Saves an experience to the buffer.

#         Args:
#             state (tuple): Current state (image, numerical data).
#             action (int): Action taken.
#             reward (float): Reward received.
#             next_state (tuple): Next state (image, numerical data).
#             done (bool): Whether the episode is done.
#         """
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         """Samples a batch of experiences from the buffer.

#         Args:
#             batch_size (int): Number of experiences to sample.

#         Returns:
#             tuple: A tuple of (state_batch, action_batch, reward_batch, next_state_batch, done_batch).
#         """
#         batch = random.sample(self.buffer, batch_size)
#         state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

#         # Convert states to tensors and handle None values
#         state_images, state_numerical_data = zip(*[(s[0], s[1]) if s is not None else (None, None) for s in state_batch])
#         next_state_images, next_state_numerical_data = zip(*[(s[0], s[1]) if s is not None else (None, None) for s in next_state_batch])

#         # Create empty tensors if all elements are None
#         state_images_list = [img for img in state_images if img is not None]
#         next_state_images_list = [img for img in next_state_images if img is not None]
#         state_numerical_data_list = [num for num in state_numerical_data if num is not None]
#         next_state_numerical_data_list = [num for num in next_state_numerical_data if num is not None]

#         # Check if any images are available
#         if state_images_list:
#             state_images = torch.stack(state_images_list)
#         else:
#             state_images = torch.empty(0, 1, 320, 320)

#         if next_state_images_list:
#             next_state_images = torch.stack(next_state_images_list)
#         else:
#             next_state_images = torch.empty(0, 1, 320, 320)
        
#         if state_numerical_data_list:
#             state_numerical_data = torch.stack(state_numerical_data_list)
#         else:
#             state_numerical_data = torch.empty(0, 17)
        
#         if next_state_numerical_data_list:
#             next_state_numerical_data = torch.stack(next_state_numerical_data_list)
#         else:
#             next_state_numerical_data = torch.empty(0, 17)

#         action_batch = torch.tensor(action_batch, dtype=torch.long)
#         reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
#         done_batch = torch.tensor(done_batch, dtype=torch.bool)

#         return (
#             state_images,
#             state_numerical_data,
#             action_batch,
#             reward_batch,
#             next_state_images,
#             next_state_numerical_data,
#             done_batch,
#         )

#     def __len__(self):
#         """Returns the current number of experiences in the buffer."""
#         return len(self.buffer)

# def train_model(model, target_model, env, optimizer, replay_buffer, device, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes):
#     """
#     Trains the DQN model.

#     Args:
#         model (nn.Module): The DQN model.
#         target_model (nn.Module): The target DQN model.
#         env (TradingEnvironment): The trading environment.
#         optimizer (optim.Optimizer): Optimizer.
#         replay_buffer (ReplayBuffer): The replay buffer.
#         device (str): Device to train on ('cuda' or 'cpu').
#         batch_size (int): Batch size for training.
#         gamma (float): Discount factor.
#         epsilon_start (float): Starting epsilon for epsilon-greedy.
#         epsilon_end (float): Ending epsilon for epsilon-greedy.
#         epsilon_decay (float): Epsilon decay rate.
#         num_episodes (int): number of episodes to train for.
#     """
#     epsilon = epsilon_start
#     criterion = nn.MSELoss()
#     for episode in tqdm(range(num_episodes)):
#         state, _ = env.reset()
        
#         # Check if initial state is None
#         if state[0] is None:
#            print("Initial state is None, skipping episode.")
#            continue
        
#         done = False
#         total_reward = 0

#         while not done:
#             state_image, state_numerical_data = state  # unpack the state here, inside the loop

#             # choose action
#             if random.random() > epsilon:
#                 with torch.no_grad():
#                     q_values = model(state_image.unsqueeze(0).to(device), state_numerical_data.unsqueeze(0).to(device))
#                     action = q_values.argmax().item()
#             else:
#                 action = random.randrange(3)
            
#             # take action
#             next_state_image, next_state_numerical_data, reward, done, truncated, info = env.step(action)

#             # save to memory
#             if next_state_image is not None:
#                 replay_buffer.push((state_image, state_numerical_data), action, reward, (next_state_image, next_state_numerical_data), done)
#             else:
#                 replay_buffer.push((state_image, state_numerical_data), action, reward, (None, None), done)
            
#             total_reward += reward
            
#             # update the states
#             state = (next_state_image, next_state_numerical_data)

#             if len(replay_buffer) >= batch_size:
#                 # get training data
#                 state_images, state_numerical_data, action_batch, reward_batch, next_state_images, next_state_numerical_data, done_batch = replay_buffer.sample(batch_size)
                
#                 # Check for empty tensors after sampling. This can happen if there are not enough transitions in the buffer
#                 if len(state_images) == 0 or len(state_numerical_data) == 0 or len(next_state_images) == 0 or len(next_state_numerical_data) == 0:
#                    continue # skip this loop if we have empty tensors.
                
#                 # If we have an empty tensors we do not want to process them, so skip this loop.
#                 if len(state_images) != batch_size or len(next_state_images) != batch_size:
#                     continue

#                 state_images = state_images.to(device)
#                 state_numerical_data = state_numerical_data.to(device)
#                 next_state_images = next_state_images.to(device)
#                 next_state_numerical_data = next_state_numerical_data.to(device)
#                 action_batch = action_batch.to(device)
#                 reward_batch = reward_batch.to(device)
#                 done_batch = done_batch.to(device)
                
#                 # calculate target values
#                 with torch.no_grad():
#                     next_q_values = target_model(next_state_images, next_state_numerical_data).max(1)[0]
#                     target_q_values = reward_batch + (gamma * next_q_values * (~done_batch)) # (~done_batch) is same as (1- done_batch)

#                 # get q values
#                 q_values = model(state_images, state_numerical_data).gather(1, action_batch.unsqueeze(1)).squeeze(1)

#                 # back prop
#                 loss = criterion(q_values, target_q_values)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         #update target network
#         if episode % 100 == 0:
#             target_model.load_state_dict(model.state_dict())
        
#         # update epsilon
#         epsilon = max(epsilon_end, epsilon * epsilon_decay)
#         print(f"episode: {episode+1}/{num_episodes}, total reward: {total_reward}, epsilon: {epsilon}")

# def test_model(model, env, device, num_episodes=10):
#     """
#     Tests the trained model.

#     Args:
#         model (nn.Module): The trained DQN model.
#         env (TradingEnvironment): The trading environment.
#         device (str): Device to test on ('cuda' or 'cpu').
#         num_episodes (int): number of episodes to test for.
#     """
#     model.eval()  # Set the model to evaluation mode
#     total_test_reward = 0
    
#     with torch.no_grad():
#         for episode in tqdm(range(num_episodes)):
#             state, _ = env.reset()
            
#             # Check if initial state is None
#             if state[0] is None:
#                 print("Initial state is None, skipping test episode.")
#                 continue

#             done = False
#             episode_reward = 0
#             while not done:
#                 state_image, state_numerical_data = state
#                 q_values = model(state_image.unsqueeze(0).to(device), state_numerical_data.unsqueeze(0).to(device))
#                 action = q_values.argmax().item()

#                 next_state_image, next_state_numerical_data, reward, done, truncated, info = env.step(action)
                
#                 episode_reward += reward
#                 state = (next_state_image, next_state_numerical_data)
            
#             total_test_reward += episode_reward
#             print(f"Test Episode: {episode+1}/{num_episodes}, Total Reward: {episode_reward}")

#     avg_test_reward = total_test_reward / num_episodes if num_episodes > 0 else 0
#     print(f"Average Test Reward over {num_episodes} episodes: {avg_test_reward}")

# def main():
#     """Main function to load data, create model, and train/evaluate."""
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Define paths
#     metadata_file = 'strategies/RL/data/metadata.csv'
#     image_dir = 'strategies/RL/data/charts'

#     # Hyperparameters
#     num_episodes = 100
#     batch_size = 32
#     learning_rate = 0.001
#     num_actions = 3
#     gamma = 0.99
#     epsilon_start = 1.0
#     epsilon_end = 0.01
#     epsilon_decay = 0.995
#     replay_buffer_capacity = 10000
#     lookback_window=75
#     transaction_cost_percent = 0.001
#     initial_balance = 10000
#     test_size = 0.2  # 20% of the data for testing

#     # --- Data Preprocessing ---
#     metadata_df = pd.read_csv(metadata_file)

#     # Add 'time' column
#     metadata_df['time'] = metadata_df['timestamp'].apply(lambda x: int(x.split('_')[1].replace('-', '')))

#     # Normalize numerical data
#     numerical_cols = ["Open", "High", "Low", "Close", "Volume", "SMA", "Upper", "Lower", "Open_HA", "High_HA", "Low_HA", "Close_HA", "Open_R", "High_R", "Low_R", "Close_R", "time"]
#     scaler = MinMaxScaler()
#     metadata_df[numerical_cols] = scaler.fit_transform(metadata_df[numerical_cols])

#     # Split data into train and test
#     train_df, test_df = train_test_split(metadata_df, test_size=test_size, shuffle=False)

#     # Save the processed metadata
#     train_df.to_csv('strategies/RL/data/train_metadata.csv', index=False)
#     test_df.to_csv('strategies/RL/data/test_metadata.csv', index=False)

#     # --- End Data Preprocessing ---

#     # Create environment
#     train_env = TradingEnvironment('strategies/RL/data/train_metadata.csv', image_dir, initial_balance=initial_balance, transaction_cost_percent=transaction_cost_percent, lookback_window=lookback_window)
#     test_env = TradingEnvironment('strategies/RL/data/test_metadata.csv', image_dir, initial_balance=initial_balance, transaction_cost_percent=transaction_cost_percent, lookback_window=lookback_window)

#     # Create model, loss function, and optimizer
#     model = DQN(num_actions).to(device)
#     target_model = DQN(num_actions).to(device)
#     target_model.load_state_dict(model.state_dict())
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
#     # Train the model
#     train_model(model, target_model, train_env, optimizer, replay_buffer, device, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, num_episodes)
    
#     # Test the model
#     test_model(model, test_env, device, num_episodes=10)

# if __name__ == "__main__":
#     main()
