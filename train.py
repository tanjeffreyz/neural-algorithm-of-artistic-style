import torch
import os
import argparse
import numpy as np
import torchvision.transforms as T
from models import SimpleAutoencoder, ConvolutionalAutoencoder
from torchvision.datasets.lfw import LFWPeople
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm


writer = SummaryWriter()
now = datetime.now()

transform = T.Compose([
    T.ToTensor(),
    T.Resize((64, 64)),
    T.Grayscale(),
])
train_set = LFWPeople(
    root='data',
    split='train',
    download=True,
    transform=transform
)
test_set = LFWPeople(
    root='data',
    split='test',
    download=True,
    transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1E-3
)

# Loss Function
loss_function = torch.nn.BCELoss()

# Create folders for this run
root = os.path.join(
    'models',
    str(model.__class__.__name__),
    now.strftime('%m_%d_%Y'),
    now.strftime('%H_%M_%S')
)
weight_dir = os.path.join(root, 'weights')
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir)

# Metrics
train_losses = np.empty((2, 0))
test_losses = np.empty((2, 0))


def save_metrics():
    np.save(os.path.join(root, 'train_losses'), train_losses)
    np.save(os.path.join(root, 'test_losses'), test_losses)


# Train
for epoch in tqdm(range(50), desc='Epoch'):
    model.train()
    train_loss = 0
    for data, _ in tqdm(train_loader, desc='Train', leave=False):
        data = data.to(device)

        optimizer.zero_grad()
        predictions = model.forward(data)
        loss = loss_function(predictions, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_loader)
        del data

    train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
    writer.add_scalar('Loss/train', train_loss, epoch)

    if epoch % 2 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data, _ in tqdm(test_loader, desc='Test', leave=False):
                data = data.to(device)

                predictions = model.forward(data)
                loss = loss_function(predictions, data)

                test_loss += loss.item() / len(test_loader)
                del data
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()
            # torch.save(model.state_dict(), os.path.join(weight_dir, f'cp_{epoch}'))
save_metrics()
torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))