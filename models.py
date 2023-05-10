import torch.nn as nn
from config import *


def get_gram_matrix(features):
    b, n, w, h = features.shape
    assert b == 1, 'Batch size must be 1'

    flattened = features.view(n, w * h)     # Vectorize each feature map
    factor = 2 * n * w * h                  # Normalization factor described in paper
    return torch.matmul(flattened, flattened.transpose(0, 1)) / factor


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

        # Normalization parameters used during VGG training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, content_features):
        super().__init__()
        self.content_features = content_features.detach()
        self.loss = None

    def forward(self, x):
        self.loss = torch.sum((self.content_features - x) ** 2) / 2
        return x


class StyleLoss(nn.Module):
    def __init__(self, style_features):
        super().__init__()
        self.style_gram = get_gram_matrix(style_features.detach())
        self.loss = None

    def forward(self, x):
        x_gram = get_gram_matrix(x)
        self.loss = torch.sum((self.style_gram - x_gram) ** 2)
        return x


class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(Normalize())

    def forward(self, x):
        return


if __name__ == '__main__':
    import torch

    # Check that dimensions are correct
    test = torch.rand((1, 16, 32, 32))
    ContentLoss(test).forward(test)
    StyleLoss(test).forward(test)
