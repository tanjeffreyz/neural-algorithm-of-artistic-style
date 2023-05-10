import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.models import vgg19
from config import DEVICE


def get_gram_matrix(features):
    b, n, w, h = features.shape
    assert b == 1, 'Batch size must be 1'

    flattened = features.view(n, w * h)     # Vectorize each feature map
    factor = n * w * h                  # Normalization factor described in paper
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
        self.loss = F.mse_loss(self.content_features, x)
        return x


class StyleLoss(nn.Module):
    def __init__(self, style_features):
        super().__init__()
        self.style_gram = get_gram_matrix(style_features.detach())
        self.loss = None

    def forward(self, x):
        x_gram = get_gram_matrix(x)
        self.loss = F.mse_loss(self.style_gram, x_gram)
        return x


class NeuralStyleTransfer(nn.Module):
    def __init__(self,
                 content_img, style_img,
                 content_layers, style_layers):
        super().__init__()

        assert len(content_img.shape) == 4, 'Content image tensor must be 4-dimensional: BxCxWxH'
        assert len(style_img.shape) == 4, 'Style image tensor must be 4-dimensional: BxCxWxH'

        # Resize style image to be same size as content image
        style_img = resize(style_img, content_img.shape[-2:])
        style_img = style_img.to(DEVICE)
        content_img = content_img.to(DEVICE)

        # Build the model based on PyTorch's pretrained VGG-19
        vgg_model = vgg19(pretrained=True)
        vgg_layers = vgg_model.features.to(DEVICE).eval()
        self.model = nn.Sequential(Normalize().to(DEVICE))

        self.content_loss_layers = []
        self.style_loss_layers = []

        i = 0
        for layer in vgg_layers.children():
            if isinstance(layer, nn.Conv2d):
                i += 1      # Conv layer signals a new block, increment block number
                name = f'conv_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False).to(DEVICE)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'avgpool_{i}'
                layer = nn.AvgPool2d(       # Paper said avg pool produced better results
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding
                ).to(DEVICE)
            else:
                raise NotImplementedError('This case should never have been reached')

            self.model.add_module(name, layer)

            # Add loss layers after specified layers
            if name in content_layers:
                content_features = self.model.forward(content_img)
                content_loss_layer = ContentLoss(content_features).to(DEVICE)
                self.model.add_module(f'content_loss_{i}', content_loss_layer)
                self.content_loss_layers.append(content_loss_layer)
            if name in style_layers:
                style_features = self.model.forward(style_img)
                style_loss_layer = StyleLoss(style_features).to(DEVICE)
                self.model.add_module(f'style_loss_{i}', style_loss_layer)
                self.style_loss_layers.append(style_loss_layer)

        # Trim model until deepest loss layer to avoid unnecessary computation
        for i in reversed(range(len(self.model))):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break
        self.model = self.model[:i+1]

    def forward(self, x):
        return self.model.forward(x)


if __name__ == '__main__':
    import torch
    from config import CONTENT_LAYERS, STYLE_LAYERS

    # Check that loss layer dimensions are correct
    test1 = torch.randn(1, 16, 32, 32)
    test2 = torch.randn(1, 16, 32, 32)

    print(get_gram_matrix(test1).shape)

    cl = ContentLoss(test1)
    cl.forward(test2)
    print(cl.loss)

    sl = StyleLoss(test1)
    sl.forward(test2)
    print(sl.loss)

    # Check that model was built correctly
    content_test = torch.randn(1, 3, 1920, 1080)
    style_test = torch.randn(1, 3, 1234, 1234)
    model = NeuralStyleTransfer(content_test, style_test, CONTENT_LAYERS, STYLE_LAYERS)
    print(model)
