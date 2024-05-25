import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as weights
from typing import Callable


class VisualOdometryModel(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        lstm_dropout: float = 0.2
    ) -> None:

        super(VisualOdometryModel, self).__init__()

        # Load pre-trained ResNet model
        self.cnn_model = resnet(weights=weights.DEFAULT)
        resnet_output = list(self.cnn_model.children())[-1].in_features
        self.cnn_model.fc = nn.Identity()

        # Freeze the weights of the ResNet layers
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # LSTM
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            resnet_output,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout
        )


        # FC to generate the translation (3) and rotation (4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*pow(2, int(bidirectional)), hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 7)
        )

    def resnet_transforms(self) -> Callable:
        return weights.DEFAULT.transforms(antialias=True)

    def forward(self, x: torch.TensorType) -> torch.TensorType:

        # CNN feature extraction
        batch_size, seq_length, channels, height, width = x.size()
        features = x.view(batch_size * seq_length, channels, height, width)

        with torch.no_grad():
            features = self.cnn_model(features)

        # Use LSTM
        features = features.view(batch_size, seq_length, -1)
        lstm_out_hidden, lstm_out_cell = self.lstm(features)

        # Last time step
        return self.fc(lstm_out_hidden[:, -1, :])
