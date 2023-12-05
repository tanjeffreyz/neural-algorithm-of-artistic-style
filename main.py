import torch
import os
import json
import imageio
from torchvision.io import read_image, write_png
from torch.utils.tensorboard import SummaryWriter
from models import StyleTransfer


USE_CACHE = False
config = {
    'content': {
        'name': 'tubingen',
        'layers': [10],     # conv4_2
        'weight': 1
    },
    'style': {
        'name': 'starry_night',
        'layers': [1, 3, 5, 9, 13],     # conv1_1 to conv5_1
        'weight': 1E6
    },
    'num_iters': 600
}

# Load config if already exists
file_name = config['content']['name'] + '_' + config['style']['name']
if USE_CACHE:
    config_path = os.path.join('results', file_name + '.json')
    if os.path.exists(config_path):
        print(f'[~] Using existing configuration: {config_path}')
        with open(config_path, 'r') as file:
            config = json.load(file)


#######################
#   Training Script   #
#######################
writer = SummaryWriter()
device = ('cuda' if torch.cuda.is_available() else 'cpu')

content_path = os.path.join('data', 'content', config['content']['name'] + '.jpg')
style_path = os.path.join('data', 'style', config['style']['name'] + '.jpg')

c_img = read_image(content_path).type(torch.float32) / 255
s_img = read_image(style_path).type(torch.float32) / 255

# Add batch dimension to images and move to GPU
c_img = c_img.unsqueeze(0).to(device)
s_img = s_img.unsqueeze(0).to(device)

# Paper states that starting from c_img or white noise are equivalent
result = torch.rand(c_img.shape).to(device)
result.requires_grad_(True)

# Initialize rest of training boilerplate
model = StyleTransfer(
    c_img, s_img,
    c_layers=config['content']['layers'],
    s_layers=config['style']['layers']
)

# optimizer = torch.optim.Adam(       # Optimizing image, not model weights!!!
#     [result],
#     lr=config['learning_rate']
# )
optimizer = torch.optim.LBFGS([result])

# Training loop
FPS = 30
DURATION = 2        # Duration in seconds

first_frame = torch.clamp(result, 0, 1)
first_frame = (first_frame.cpu() * 255).type(torch.uint8).squeeze(0)
frames = [first_frame.numpy().transpose(1, 2, 0)]


def lbfgs_closure():
    global i

    with torch.no_grad():
        result.clamp_(0, 1)

    optimizer.zero_grad()
    model(result)

    # Calculate content and style losses
    c_loss = 0
    for layer in model.content_losses:
        c_loss += layer.loss
    s_loss = 0
    for layer in model.style_losses:
        s_loss += layer.loss
    c_loss *= config['content']['weight']
    s_loss *= config['style']['weight']
    total_loss = c_loss + s_loss
    total_loss.backward()

    writer.add_scalar('Loss/total', total_loss.item(), i)
    writer.add_scalar('Loss/content', c_loss.item(), i)
    writer.add_scalar('Loss/style', s_loss.item(), i)

    # Save intermediate images
    if i % (config['num_iters'] // (FPS * DURATION)) == 0:
        frame = torch.clamp(result, 0, 1)
        frame = (frame.cpu() * 255).type(torch.uint8).squeeze(0)
        frames.append(frame.numpy().transpose(1, 2, 0))

        print(f'\nIteration {i}:')
        print(f'Content loss: {c_loss.item()}')
        print(f'Style loss: {s_loss.item()}')
        print(f'Total loss: {total_loss.item()}')
    i += 1
    return total_loss


i = 0
while i < config['num_iters']:
    optimizer.step(lbfgs_closure)

# Write final image to results
result = torch.clamp(result, 0, 1)
result = (result.cpu() * 255).type(torch.uint8).squeeze(0)
write_png(result, os.path.join('results', file_name + '.png'))

# Save progression of results over course of training
frames.append(result.numpy().transpose(1, 2, 0))
padding = int(FPS * 0.5)
frames = [frames[0] for _ in range(padding)] + frames + [frames[-1] for _ in range(padding)]
imageio.mimsave(os.path.join('results', file_name + '.gif'), frames, loop=0, duration=int(1000 / FPS))

# Save configuration for reference too
with open(os.path.join('results', file_name + '.json'), 'w') as file:
    json.dump(config, file, indent=2)
