import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import random
import torch
import torchvision.transforms as transforms
from network import DQN  # Assuming DQN model is defined in network.py
import logging
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MetadataProcessor:
    def __init__(self, metadata_file=None, image_dir=None, model_path=None, device="cpu", transaction_cost_percent=0.001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, test_mode=False):
        """
        Initializes the MetadataProcessor.

        Args:
            metadata_file (str, optional): Path to the metadata CSV file.
            image_dir (str, optional): Directory containing the images.
            model_path (str, optional): Path to the trained DQN model.
            device (str): Device to run on ('cuda' or 'cpu').
            transaction_cost_percent (float): Transaction cost percentage.
            epsilon_start (float): Starting epsilon value for epsilon-greedy strategy.
            epsilon_end (float): Ending epsilon value for epsilon-greedy strategy.
            epsilon_decay (float): Decay rate for epsilon.
            test_mode (bool): If True, the agent will only choose actions from the model.
        """
        self.metadata_file = metadata_file
        self.image_dir = image_dir
        self.transaction_cost_percent = transaction_cost_percent
        self.scaler = MinMaxScaler()  # Initialize the scaler for reward calculation (unnormalized)
        self.numerical_data_scaler = MinMaxScaler() # create a new scaler for the numerical data to pass to the model. (normalized)
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        self.device = torch.device(device)
        self.num_actions = 3  # hold, buy, sell
        self.numerical_cols = ["Open", "High", "Low", "Close", "Volume", "SMA", "Upper", "Lower", "Open_HA", "High_HA",
                               "Low_HA", "Close_HA", "Open_R", "High_R", "Low_R", "Close_R", "time"]

        if model_path:
            self.model = self._load_model(model_path)

        if metadata_file:
            self.metadata = self._load_and_preprocess_metadata() # for unnormalized data.
            self.numerical_data_scaler.fit(self.metadata[self.numerical_cols])  # fit the new scaler with the metadata. for normalized data.
            self.unnormalized_metadata = self.metadata.copy()
            self.numerical_data_scaler.fit(self.unnormalized_metadata[self.numerical_cols])
        else:
            self.metadata = None
            self.unnormalized_metadata = None
            self.numerical_data_scaler = None

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.image_batch = None  # Initialize image_batch to None
        self.first_episode = True  # Initialize first_episode to True
        self.test_mode = test_mode

    def _load_model(self, model_path):
        """Loads the trained DQN model."""
        model = DQN(self.num_actions).to(self.device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            logging.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return model

    def _load_and_preprocess_metadata(self):
        """Loads and preprocesses the metadata CSV."""
        metadata = pd.read_csv(self.metadata_file)

        # Add 'time' column
        metadata['time'] = metadata['timestamp'].apply(lambda x: int(x.split('_')[1].replace('-', '')))
        
        return metadata

    def _load_and_transform_image(self, image_path):
        """Loads and transforms an image to a grayscale tensor."""
        try:
            full_image_path = os.path.join(self.image_dir, image_path)
            image = Image.open(full_image_path)

            # Check if the image is RGBA and convert to RGB if needed
            if image.mode == "RGBA":
                image = image.convert("RGB")

            image = self.transform(image)
            return image
        except Exception as e:
            logging.error(f"Error loading or transforming image {image_path}: {e}")
            return None

    def _process_image_batch(self, image_paths):
        """Loads and transforms a batch of images to grayscale tensors."""
        image_tensors = []
        for image_path in image_paths:
            image_tensor = self._load_and_transform_image(image_path)
            if image_tensor is not None:
                image_tensors.append(image_tensor)
            else:
                # Handle the case where image loading fails. Here, we'll skip this image.
                logging.warning(f"Skipping image: {image_path}")

        if not image_tensors:
            return None  # Return None if all images failed to load

        # Stack the tensors to create a batch
        return torch.stack(image_tensors).to(self.device)

    def _get_normalized_numerical_data(self, metadata):
        """
        Normalizes the numerical data using the numerical_data_scaler.

        Args:
            metadata (pd.DataFrame): DataFrame containing the numerical data.

        Returns:
            torch.Tensor: Normalized numerical data as a tensor.
        """
        numerical_data_batch_unscaled = metadata[self.numerical_cols].values.astype(np.float32)
        numerical_data_batch = self.numerical_data_scaler.transform(numerical_data_batch_unscaled)
        return torch.tensor(numerical_data_batch).to(self.device)
    
    def get_train_df(self):
        """
        Processes the metadata to create the initial train_df.

        Returns:
            pd.DataFrame: The processed metadata DataFrame.
        """
        metadata = self.metadata.copy()  # Create a copy to avoid modifying the original
        # Initialize image_tensor and numerical_data_normalized
        metadata['image_tensor'] = None  # Initialize image_tensor
        metadata['numerical_data_normalized'] = None

        # Load and transform all images in one batch only in the first episode
        if self.first_episode:
            self.image_batch = self._process_image_batch(metadata['image_path'].tolist())
            if self.image_batch is None:
                logging.error("No images were loaded correctly. Cannot continue.")
                return None
            # add image_tensor to the metadata
            metadata['image_tensor'] = self.image_batch.tolist()  # Convert the tensor to list to save it as data in the dataframe.

        # Load all numerical data into one tensor
        # Transform the numerical data using the numerical_data_scaler
        # numerical_data_batch = self._get_normalized_numerical_data(metadata)
        # metadata['numerical_data_normalized'] = numerical_data_batch.tolist()

        return metadata

    def process_metadata_for_episode(self, df = None, initial_balance=10000, gamma=0.9):
        """
        Processes the metadata to prepare data for one episode using batch processing.

        Args:
            df (pd.DataFrame): metadata DataFrame to calculate the new q targets.
            initial_balance (float): The starting balance for the trading simulation.
            gamma (float): discount factor.

        Returns:
            pd.DataFrame: The processed metadata DataFrame.
        """
        if df is None:
            # Get the unnormalized metadata if df is None
            metadata = self.unnormalized_metadata.copy()
            # Initialize new columns
            metadata['action'] = 0
            metadata['reward'] = 0
            metadata['current_balance'] = initial_balance
            metadata['current_position'] = 0
            metadata['entry_price'] = 0
            metadata['q_action'] = None
            metadata['max_q_value'] = None
            metadata['q_values'] = None
            metadata['target_q'] = 0

            # Load all numerical data into one tensor
            # Transform the numerical data using the numerical_data_scaler
            numerical_data_batch = self._get_normalized_numerical_data(metadata)
            metadata['numerical_data_normalized'] = numerical_data_batch.tolist()
            
            # add image_tensor to the metadata
            metadata['image_tensor'] = self.image_batch.tolist()  # Convert the tensor to list to save it as data in the dataframe.
        else:
            metadata = df.copy()
            # Initialize new columns
            metadata['action'] = 0
            metadata['reward'] = 0
            metadata['current_balance'] = initial_balance
            metadata['current_position'] = 0
            metadata['entry_price'] = 0
            metadata['q_action'] = None
            metadata['max_q_value'] = None
            metadata['q_values'] = None
            metadata['target_q'] = 0


        # Get q values from the model for the entire batch
        with torch.no_grad():
            if self.first_episode and self.image_batch is not None:
                q_values_batch = self.model(self.image_batch, torch.stack([torch.tensor(numerical_data).to(self.device) for numerical_data in metadata["numerical_data_normalized"].tolist()]))
            else:
                q_values_batch = self.model(torch.stack([torch.tensor(img).to(self.device) for img in metadata["image_tensor"].tolist()]), torch.stack([torch.tensor(numerical_data).to(self.device) for numerical_data in metadata["numerical_data_normalized"].tolist()]))

            metadata['q_values'] = q_values_batch.tolist()  # Convert q_values to a list to save in the dataframe.
            # Get the best q action for each row
            metadata['q_action'] = q_values_batch.argmax(dim=1).tolist()
            # Get the max q value for each row
            metadata['max_q_value'] = q_values_batch.max(dim=1)[0].tolist()

        current_position = 0
        entry_price = 0
        if not self.test_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = -1 # force to choose from the model.

        for i in range(len(metadata)):
            # Get current unnormalized price
            current_unnormalized_price = self.unnormalized_metadata.iloc[i]["Close"] # self.unnormalized_metadata is a copy of the initial metadata
            last_balance = metadata.loc[i, 'current_balance']
            valid_action = False

            # Epsilon-greedy strategy
            if random.random() > self.epsilon:
                action = metadata.loc[i, 'q_action']  # choose q action
            else:
                action = random.randrange(3)  # choose random action
            metadata.loc[i, 'action'] = action

            if action == 1 and current_position == 0:  # Buy
                transaction_cost = metadata.loc[i, 'current_balance'] * self.transaction_cost_percent
                if metadata.loc[i, 'current_balance'] > transaction_cost:
                    metadata.loc[i, 'current_balance'] -= transaction_cost
                    entry_price = current_unnormalized_price
                    current_position = 1
                    metadata.loc[i, 'current_position'] = current_position
                    metadata.loc[i, 'entry_price'] = entry_price
                    metadata.loc[i, 'reward'] = -transaction_cost
                    valid_action = True
                else:
                    # insufficient balance
                    logging.warning("insufficient balance")
                    metadata.loc[i, 'reward'] = 0
                    metadata.loc[i, 'action'] = 0  # change action to hold if there is insufficient balance.

            elif action == 2 and current_position == 1:  # Sell
                transaction_cost = metadata.loc[i, 'current_balance'] * self.transaction_cost_percent
                if metadata.loc[i, 'current_balance'] > transaction_cost:
                    metadata.loc[i, 'current_balance'] -= transaction_cost
                    metadata.loc[i, 'current_balance'] += current_unnormalized_price
                    current_position = 0
                    metadata.loc[i, 'current_position'] = current_position
                    reward = current_unnormalized_price - entry_price
                    metadata.loc[i, 'reward'] = reward - transaction_cost
                    entry_price = 0
                    metadata.loc[i, 'entry_price'] = entry_price
                    valid_action = True
                else:
                    # insufficient balance
                    logging.warning("insufficient balance")
                    metadata.loc[i, 'reward'] = 0
                    metadata.loc[i, 'action'] = 0  # change action to hold if there is insufficient balance.
            elif action == 0:  # Hold
                metadata.loc[i, 'current_position'] = current_position
                if current_position == 1:
                    reward = current_unnormalized_price - entry_price
                else:
                    reward = 0
                metadata.loc[i, 'reward'] = reward
                valid_action = True

        # shift the q_values to get the next state
        metadata['next_max_q_value'] = metadata['max_q_value'].shift(-1).fillna(0)
        # calculate target q
        metadata['target_q'] = metadata['reward'] + (gamma * metadata['next_max_q_value'])

        # drop unneeded columns
        metadata = metadata.drop(['current_position', 'entry_price', 'next_max_q_value'], axis=1)

        # Set first_episode to false after the first episode
        self.first_episode = False
        
        #return the new normalized numerical data
        numerical_data_batch = self._get_normalized_numerical_data(metadata)
        metadata['numerical_data_normalized'] = numerical_data_batch.tolist()

        return metadata

if __name__ == "__main__":

    metadata_file = 'strategies/RL/data/metadata.csv'
    image_dir = '/Users/kanchannannavare/Documents/RL_dataset/data/charts'
    model_save_path = os.path.join("strategies/RL/data/model_weights_colab.pth")

    num_episodes = 10
    batch_size = 32
    learning_rate = 0.001
    num_actions = 3
    gamma = 0.99
    replay_buffer_capacity = 10000
    transaction_cost_percent = 0.001
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    device = "cpu"

    metadata_processor = MetadataProcessor(metadata_file=metadata_file, image_dir=image_dir, model_path=None, device=device,
                                           transaction_cost_percent=transaction_cost_percent,
                                           epsilon_start=epsilon_start,
                                           epsilon_end=epsilon_end, epsilon_decay=epsilon_decay)
    # metadata_processor.metadata = metadata_processor.metadata
    print(metadata_processor.metadata.shape)
    chunk_size = 1000
    original_metadata = metadata_processor.metadata
    all_data = []
    for i in range(0, original_metadata.shape[0]//1000, chunk_size):
        chunk = original_metadata[i:i+chunk_size]
        print(chunk.head(1))
        print(chunk.tail(1))
        metadata_processor.metadata = chunk
        start_time = time.time()
        train_df = metadata_processor.get_train_df()
        print("time taken: ", time.time()-start_time)

        all_data.append(train_df)

    data = pd.concat(all_data, ignore_index=True)
    data.to_csv("rl_data.csv")
    print(train_df.head())
    print(train_df.columns.tolist())
    print(train_df.head(5)['image_tensor'])
    
    
