import os
import torch
from torch import nn
import torchvision.models as models

DEVICE = "cpu"


class CustomCNNModel(nn.Module):
    def __init__(self, input_shape: int = 3, hidden_units: int = 64, output_shape: int = 6):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_units),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Keep attribute name 'classifer' to match serialized state_dict keys in custom_cnn_model.pth
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_units * 37 * 37, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        x = self.classifer(x)
        return x


def load_custom_cnn() -> nn.Module:
    url = "https://github.com/vikassinngh123/AI-ML-Learning/releases/download/v1.0/custom_cnn_model.pth"
    filename = "custom_cnn_model.pth"
    if not os.path.exists(filename):
        torch.hub.download_url_to_file(url, filename)

    loaded_model = CustomCNNModel(input_shape=3, hidden_units=64, output_shape=6).to(DEVICE)
    loaded_model.load_state_dict(
        torch.load(filename, map_location=torch.device(DEVICE), weights_only=False)
    )
    loaded_model.eval()
    return loaded_model


def load_resnet18() -> nn.Module:
    url = "https://github.com/vikassinngh123/AI-ML-Learning/releases/download/v1.0/resnet18_model.pth"
    filename = "resnet18_model.pth"
    if not os.path.exists(filename):
        torch.hub.download_url_to_file(url, filename)

    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=128, out_features=6),
    ).to(DEVICE)

    resnet18.load_state_dict(
        torch.load(filename, map_location=torch.device(DEVICE), weights_only=False)
    )
    resnet18.eval()
    return resnet18