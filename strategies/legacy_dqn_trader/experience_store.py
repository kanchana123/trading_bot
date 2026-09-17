# experience_store.py
import torch
import random
import os
import pickle


class ExperienceStore:
    """
    A persistent experience store that can save and load experiences to/from a file.
    """

    def __init__(self, capacity, store_file="experience_store.pkl"):
        """
        Initializes the ExperienceStore.

        Args:
            capacity (int): Maximum number of experiences to store.
            store_file (str): Path to the file where experiences will be stored.
        """
        self.capacity = capacity
        self.store_file = store_file
        self.buffer = []  # Start with an empty buffer
        self.load()  # Try to load existing experiences on initialization

    def push(self, state, action, reward, next_state, done):
        """
        Saves an experience to the buffer and periodically saves it to the file.

        Args:
            state (tuple): Current state (image, numerical data).
            action (int): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state (image, numerical data).
            done (bool): Whether the episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))
        # self.buffer = self.buffer[-self.capacity:]
        self.save()  # Save to file after adding each experience

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: A tuple of (state_batch, action_batch, reward_batch, next_state_batch, done_batch) or None if not enough samples
        """
        if len(self.buffer) < batch_size:
            return None  # Not enough experiences for a full batch

        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_images, state_numerical_data = zip(*state_batch)
        next_state_images, next_state_numerical_data = zip(*next_state_batch)

        # Handle potential None values in images and numerical data (at the end of an episode)
        state_images = [img for img in state_images if img is not None]
        next_state_images = [img for img in next_state_images if img is not None]
        state_numerical_data = [num for num in state_numerical_data if num is not None]
        next_state_numerical_data = [num for num in next_state_numerical_data if num is not None]
        
        # Stack the lists or create empty tensors if empty
        state_images = torch.stack(state_images) if state_images else torch.empty(0, 1, 320, 320)
        next_state_images = torch.stack(next_state_images) if next_state_images else torch.empty(0, 1, 320, 320)
        state_numerical_data = torch.stack(state_numerical_data) if state_numerical_data else torch.empty(0, 17)
        next_state_numerical_data = torch.stack(next_state_numerical_data) if next_state_numerical_data else torch.empty(0, 17)
        

        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.bool)

        return (
            state_images,
            state_numerical_data,
            action_batch,
            reward_batch,
            next_state_images,
            next_state_numerical_data,
            done_batch,
        )

    def __len__(self):
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)

    def save(self):
        """Saves the experiences to the store file."""
        try:
            os.makedirs(os.path.dirname(self.store_file), exist_ok=True)
            with open(self.store_file, "wb") as f:
                pickle.dump(self.buffer, f)
        except Exception as e:
            print(f"Error saving experience store to {self.store_file}: {e}")

    def load(self):
        """Loads experiences from the store file."""
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, "rb") as f:
                    self.buffer = pickle.load(f)
                print(f"Loaded {len(self.buffer)} experiences from {self.store_file}")
            except Exception as e:
                print(f"Error loading experience store from {self.store_file}: {e}")
