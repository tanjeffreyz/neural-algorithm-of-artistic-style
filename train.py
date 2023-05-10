import torch
import shutil
import os
import numpy as np
from torchvision.io import read_image, write_png
from models import NeuralStyleTransfer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
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

# Can either start from content_image or random white noise
if USE_WHITE_NOISE:
    result = torch.rand(content_img.shape).to(DEVICE)
else:
    result = content_img.clone().contiguous()      # LBGFS requires gradients to be contiguously


# Optimizing content image to fit the style, freeze model weights
result.requires_grad_(True)
model.requires_grad_(False)

# Optimizer
# optimizer = torch.optim.Adam([result], lr=1E-3)
optimizer = torch.optim.LBFGS([result])

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


def finish():
    np.save(os.path.join(root, 'content_losses'), content_losses)
    np.save(os.path.join(root, 'style_losses'), style_losses)
    np.save(os.path.join(root, 'total_losses'), total_losses)

    with torch.no_grad():
        result.clamp_(0, 1)
    img = result.cpu()
    img *= 256      # Convert back to RGB values
    img = torch.squeeze(img.type(torch.uint8), dim=0)   # Cast back to byte tensor and remove batch dimension

    content_name, _ = os.path.splitext(CONTENT_IMAGE)
    style_name, _ = os.path.splitext(STYLE_IMAGE)
    write_png(img, os.path.join(root, f'{content_name}-{style_name}.png'))

    # Save configuration as well
    shutil.copyfile('config.py', os.path.join(root, 'config.py'))


# Train
i = [0]
while i[0] < NUM_ITERS:
    def closure():
        with torch.no_grad():
            result.clamp_(0, 1)

        optimizer.zero_grad()
        model.forward(result)

        # Calculate total loss from specified layers
        content_loss = 0
        for layer in model.content_loss_layers:
            content_loss += layer.loss
        content_loss *= CONTENT_WEIGHT

        style_loss = 0
        for layer in model.style_loss_layers:
            style_loss += layer.loss
        style_loss *= STYLE_WEIGHT

        total_loss = content_loss + style_loss
        total_loss.backward()

        # Occasionally save and print losses
        if i[0] % 10 == 0:
            writer.add_scalar('Loss/content', content_loss.item(), i[0])
            writer.add_scalar('Loss/style', style_loss.item(), i[0])
            writer.add_scalar('Loss/total', total_loss.item(), i[0])
            np.append(content_losses, [[i[0]], [content_loss.item()]], axis=1)
            np.append(style_losses, [[i[0]], [style_loss.item()]], axis=1)
            np.append(total_losses, [[i[0]], [total_loss.item()]], axis=1)

            print(f'\n[~] Iteration {i[0]}:')
            print(f'    content_loss = {content_loss.item()}')
            print(f'    style_loss = {style_loss.item()}')
            print(f'    total_loss = {total_loss.item()}')

        i[0] += 1
        return content_loss + style_loss

    # Update the content image
    optimizer.step(closure)

# Save final result
finish()
