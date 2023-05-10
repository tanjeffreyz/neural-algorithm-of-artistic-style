import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONTENT_IMAGE = 'tubingen.jpg'
STYLE_IMAGE = 'picasso.jpg'
SCALE_FACTOR = 0.5          # Scales both dimensions of the content image

CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1E7

NUM_ITERS = 1000
