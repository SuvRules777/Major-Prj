import torch
import torch.nn as nn


class FishBiomassCNN(nn.Module):
    """Custom CNN for fish biomass regression using image inputs.

    The network is slightly deeper than the SimpleBiomassCNN in src/model_comparison.py
    and is intended to act as a stronger baseline CNN model.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After four 2x2 pools, spatial size is img_size / 16
        feature_size = img_size // 16
        flattened_dim = 256 * feature_size * feature_size

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.features(input_tensor)
        output = self.regressor(features)
        return output
