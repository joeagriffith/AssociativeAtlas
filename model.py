import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CNN(nn.Module):
    def __init__(self, num_features=8, out_channels=128):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(5, num_features, 5, 2, 2),
            ConvBlock(num_features, num_features*2, 5, 2, 1),
            ConvBlock(num_features*2, num_features*4, 3, 2, 0),
            ConvBlock(num_features*4, num_features*4, 3, 1, 0),
            ConvBlock(num_features*4, num_features*4, 3, 1, 0),
        ])

        self.lin_layers = nn.Sequential(
            nn.Linear(num_features*4, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.flatten(1)
        x = self.lin_layers(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=256, out_channels=128):
        super().__init__()
        self.embed = nn.Linear(50*5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Sequential(
            nn.Linear(d_model, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.csl_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        # x: (N, C, H, W)
        # convert to (N, W, C*H)
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)
        x = self.embed(x)

        # add csl token
        csl_token = self.csl_token.expand(x.shape[0], -1, -1)
        x = torch.cat((csl_token, x), dim=1)

        x = self.transformer_encoder(x)

        x = x[:, 0, :]
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x