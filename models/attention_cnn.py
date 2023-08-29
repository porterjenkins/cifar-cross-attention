import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2)
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x / torch.norm(x, p=2, dim=1).unsqueeze(-1) # l2 norm


class NaiveAggregator(nn.Module):

    def __init__(self, dim):
        super(NaiveAggregator, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(5*5, dim)

    def forward(self, x):
        x = torch.flatten(x, 2)  # flatten all dimensions except batch
        h = self.relu(self.fc1(x))
        return h

class SqueezeExciteAggregator(nn.Module):
    def __init__(self, dim):
        super(SqueezeExciteAggregator, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16, dim)

    def forward(self, x: torch.Tensor):
        avg_x = torch.mean(x, dim=[2, 3])
        f_avg = self.relu(self.fc1(avg_x))
        return f_avg.unsqueeze(1)


class GlobalCNN(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.feature_map_aggregator = SqueezeExciteAggregator(dim)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        h = self.feature_map_aggregator(x)
        return h / torch.norm(h, p=2, dim=-1).unsqueeze(-1) # l2 norm

class TransformerMlp(nn.Module):
    def __init__(self, dim, dropout_prob, fc_dims):
        super(TransformerMlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, fc_dims),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_dims, dim),
        )

    def forward(self, x):
        return self.net(x)

class CrossAttentionCNN(nn.Module):

    def __init__(self, dim: int, backbone: nn.Module, global_cnn: nn.Module, mlp: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.glob_cnn = global_cnn
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)
        self.mlp = mlp

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)

        self.fc_out = nn.Linear(dim, 10)


    def forward(self, crop, img):
        h_crop = self.backbone(crop).unsqueeze(1)
        h_glob = self.glob_cnn(img)
        h_cross, attn_weights = self.attn(query=h_crop, key=h_glob, value=h_glob)
        h = self.bn1((h_crop + h_cross).squeeze(1))

        h2 = self.mlp(h)
        h2 = self.bn2(h + h2)

        y = self.fc_out(h2)

        return y


    @classmethod
    def build(cls, device: torch.device):
        dim = 64
        backbone = Backbone(dim)
        glob_cnn = GlobalCNN(dim)
        mlp = TransformerMlp(dim=dim, dropout_prob=0.0, fc_dims=32)
        model = CrossAttentionCNN(dim, backbone, glob_cnn, mlp).to(device)
        return model
