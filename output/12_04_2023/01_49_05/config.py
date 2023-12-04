import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONTENT_IMAGE = 'tubingen.jpg'
STYLE_IMAGE = 'composition.jpg'

CONTENT_LAYERS = ['relu_4']
STYLE_LAYERS = ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1E6

NUM_ITERS = 600

# 0: start with white_noise
# 1: start with content_image
# 2: start with content_image + white_noise
MODE = 0
