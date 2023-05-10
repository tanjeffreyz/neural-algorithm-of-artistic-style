import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONTENT_IMAGE = ''
STYLE_IMAGE = ''

CONTENT_WEIGHT = 1E-3
STYLE_WEIGHT = 1

LEARNING_RATE = 1E-3
