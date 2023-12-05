import torch
import torchvision
from torch import nn
from torch.nn.functional import mse_loss


def get_gram_matrix(x):
    _, c, w, h = x.shape
    flat = x.view(c, w * h)
    return torch.matmul(flat, flat.transpose(0, 1)) / (c * w * h)


class ContentLossProbe(nn.Module):
    def __init__(self, content):
        super().__init__()
        self.content = content.detach()
        self.loss = -1

    def forward(self, x):
        self.loss = mse_loss(self.content, x)
        return x


class StyleLossProbe(nn.Module):
    def __init__(self, style):
        super().__init__()
        self.style = style
        self.style_gram_matrix = get_gram_matrix(self.style.detach())
        self.loss = -1

    def forward(self, x):
        self.loss = mse_loss(
            self.style_gram_matrix,
            get_gram_matrix(x)
        )
        return x


class NormalizeVGG19(nn.Module):
    def forward(self, x):
        """
        Normalize image according to VGG-19's training parameters
        https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
        """

        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
        mean = mean.to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
        std = std.to(x.device)
        return (x - mean) / std


class StyleTransfer(nn.Module):
    def __init__(self, c_img, s_img, c_layers=[], s_layers=[]):
        super().__init__()

        max_depth = max(c_layers + s_layers)
        resize = torchvision.transforms.Resize(c_img.shape[-2:])
        s_img = resize(s_img)

        # Load pre-trained VGG-19 weights
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg19(weights=weights)
        vgg_features = vgg.features.to(c_img.device)
        vgg_features.eval()

        # Name each layer and insert loss probes
        self.seq = nn.Sequential(NormalizeVGG19())
        self.content_losses = []
        self.style_losses = []

        depth = 0
        for layer in vgg_features.children():
            if isinstance(layer, nn.Conv2d):
                depth += 1
                layer_name = 'conv'
            if isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(
                    stride=layer.stride,
                    padding=layer.padding,
                    kernel_size=layer.kernel_size
                ).to(c_img.device)
                layer_name = 'avgpool'
            if isinstance(layer, nn.BatchNorm2d):
                layer_name = 'bnorm'
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False).to(c_img.device)
                layer_name = 'relu'

            self.seq.add_module(f'{layer_name}_{depth}', layer)

            # Insert loss probe after ReLU layer if depth matches
            # The paper mentions that only positive activations must contribute to the loss
            if isinstance(layer, nn.ReLU):
                if depth in c_layers:
                    c_activations = self.seq(c_img)
                    c_probe = ContentLossProbe(c_activations)
                    c_probe = c_probe.to(c_img.device)
                    self.content_losses.append(c_probe)
                    self.seq.add_module(f'closs_{depth}', c_probe)
                if depth in s_layers:
                    s_activations = self.seq(s_img)
                    s_probe = StyleLossProbe(s_activations)
                    s_probe = s_probe.to(c_img.device)
                    self.style_losses.append(s_probe)
                    self.seq.add_module(f'sloss_{depth}', s_probe)

                # No more loss probes needed, discard rest of layers to save compute
                if depth == max_depth:
                    break

        # Freeze the model weights
        self.requires_grad_(False)

    def forward(self, x):
        return self.seq(x)


if __name__ == '__main__':
    content_test = torch.rand(1, 3, 64, 64)
    style_test = torch.rand(1, 3, 64, 512)
    model = StyleTransfer(
        content_test,
        style_test,
        c_layers=[4],
        s_layers=[1, 2, 3, 4]
    )
    print(model)
    print(all(not p.requires_grad for p in model.parameters()))
