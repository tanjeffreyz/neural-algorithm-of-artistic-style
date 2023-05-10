import torch
import os
import numpy as np
from torchvision.io import read_image, write_png
from models import NeuralStyleTransfer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from config import *


writer = SummaryWriter()
now = datetime.now()

# Load input images and normalize RGB to be in range [0, 1]
content_img = read_image(os.path.join('input', 'content', CONTENT_IMAGE)).type(torch.FloatTensor)
content_img /= 256
content_img = torch.unsqueeze(content_img, 0).to(DEVICE)

style_img = read_image(os.path.join('input', 'style', STYLE_IMAGE)).type(torch.FloatTensor)
style_img /= 256
style_img = torch.unsqueeze(style_img, 0).to(DEVICE)

# Build the model
model = NeuralStyleTransfer(content_img, style_img, CONTENT_LAYERS, STYLE_LAYERS)

# Optimizing content image to fit the style, freeze model weights
content_img.requires_grad_(True)
model.requires_grad_(False)

# Optimizer
optimizer = torch.optim.Adam(
    [content_img],            # Optimizing content image, not model weights!
    lr=LEARNING_RATE
)

# Create folders for this run
root = os.path.join(
    'output',
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
if not os.path.isdir(root):
    os.makedirs(root)

# Metrics
content_losses = np.empty((2, 0))
style_losses = np.empty((2, 0))
total_losses = np.empty((2, 0))


def save():
    np.save(os.path.join(root, 'content_losses'), content_losses)
    np.save(os.path.join(root, 'style_losses'), style_losses)
    np.save(os.path.join(root, 'total_losses'), total_losses)
    img = content_img.cpu()
    img *= 256      # Convert back to RGB values
    img = torch.squeeze(img.type(torch.uint8), dim=0)   # Cast back to byte tensor and remove batch dimension
    write_png(img, os.path.join(root, 'result.png'))


# Train
for i in tqdm(range(NUM_ITERS), desc='Iteration'):
    with torch.no_grad():
        content_img.clamp_(0, 1)

    optimizer.zero_grad()
    model.forward(content_img)

    # Calculate total loss from specified layers
    style_loss = 0
    for layer in model.style_loss_layers:
        style_loss += layer.loss

    content_loss = 0
    for layer in model.content_loss_layers:
        content_loss += layer.loss

    total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
    total_loss.backward()

    # Update the content image
    optimizer.step()

    # Occasionally save preliminary results
    if i % 10 == 0:
        writer.add_scalar('Loss/content', content_loss.item(), i)
        writer.add_scalar('Loss/style', style_loss.item(), i)
        writer.add_scalar('Loss/total', total_loss.item(), i)
        np.append(content_losses, [[i], [content_loss.item()]], axis=1)
        np.append(style_losses, [[i], [style_loss.item()]], axis=1)
        np.append(total_losses, [[i], [total_loss.item()]], axis=1)
        save()

# Save final result
with torch.no_grad():
    content_img.clamp_(0, 1)
save()
