import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2)
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x / torch.norm(x, p=2, dim=1).unsqueeze(-1) # l2 norm


class GlobalCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5 * 5, 32)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 2) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return x / torch.norm(x, p=2, dim=-1).unsqueeze(-1) # l2 norm


class CrossAttentionCNN(nn.Module):

    def __init__(self, backbone: nn.Module, global_cnn: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.glob_cnn = global_cnn
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=1, batch_first=True)
        self.fc1 = nn.Linear(32, 10)


    def forward(self, crop, img):
        h_crop = self.backbone(crop).unsqueeze(1)
        h_glob = self.glob_cnn(img)

        h, attn_weights = self.attn(query=h_crop, key=h_glob, value=h_glob)
        y_hat = self.fc1(h.squeeze(1))

        return y_hat


    @classmethod
    def build(cls):
        backbone = Backbone()
        glob_cnn = GlobalCNN()
        model = CrossAttentionCNN(backbone, glob_cnn)
        return model
