#network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers for image processing (grayscale)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)  # input is 320x320x1 (grayscale)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out((1, 320, 320))  # Adjusted for grayscale

        # Fully connected layers for image and numerical data processing
        self.fc_image = nn.Linear(conv_out_size, 512)
        self.fc_numerical = nn.Linear(17, 512)  # 17 numerical inputs
        self.fc_combined = nn.Linear(512 + 512, 512)
        self.fc_out = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        """Calculates the output size of the convolutional layers."""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, image, numerical_data):
        """
        Forward pass of the DQN.

        Args:
            image (torch.Tensor): Image tensor of shape (batch_size, 1, 320, 320).
            numerical_data (torch.Tensor): Numerical data tensor of shape (batch_size, 17).

        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Image processing
        conv_out = F.relu(self.conv1(image))
        conv_out = F.relu(self.conv2(conv_out))
        conv_out = F.relu(self.conv3(conv_out))
        conv_out = conv_out.view(conv_out.size()[0], -1)  # Flatten the output
        image_features = F.relu(self.fc_image(conv_out))

        # Numerical data processing
        numerical_features = F.relu(self.fc_numerical(numerical_data))

        # Concatenate image and numerical features
        combined_features = torch.cat((image_features, numerical_features), dim=1)

        # Output layer
        combined_features = F.relu(self.fc_combined(combined_features))
        q_values = self.fc_out(combined_features)

        return q_values
    
if __name__ == "__main__":
    # Example usage:
    num_actions = 3  # Example: Buy, Hold, Sell
    model = DQN(num_actions)

    # Example input data (grayscale)
    batch_size = 32
    image_input = torch.randn(batch_size, 1, 320, 320)  # Batch of 32 grayscale images
    numerical_input = torch.randn(batch_size, 17)  # Batch of 32 sets of numerical data

    # Forward pass
    q_values = model(image_input, numerical_input)
    print(q_values.shape)  # Should print: torch.Size([32, 3])
