import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONTENT_IMAGE = 'tubingen.jpg'
STYLE_IMAGE = 'starry_night.jpg'

CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

CONTENT_WEIGHT = 1E-6
STYLE_WEIGHT = 1

LEARNING_RATE = 1E-3
NUM_ITERS = 1000
