import torch


def content_loss(p: torch.Tensor, x: torch.Tensor):
    assert p.shape == x.shape
    return torch.sum((p - x) ** 2) / 2


def _layer_style_loss(a, x):
    a_flat = torch.flatten(a, start_dim=2)
    _, n, m = a_flat.shape

    x_flat = torch.flatten(x, start_dim=2)
    a_gram = torch.matmul(a_flat, torch.transpose(a_flat, 1, 2))
    x_gram = torch.matmul(x_flat, torch.transpose(x_flat, 1, 2))
    return torch.sum((a_gram - x_gram) ** 2) / (4 * (n ** 2) * (m ** 2))


def style_loss(a_list, x_list):
    assert len(a_list) == len(x_list), 'Number of original and generated feature maps must be equal'
    n = len(a_list)


test1 = torch.rand((64, 16, 32, 32))
test2 = torch.rand((64, 16, 32, 32))

print(_layer_style_loss(test1, test2).shape)
