import torch


def content_loss(p: torch.Tensor, x: torch.Tensor):
    assert p.shape == x.shape
    return torch.sum((p - x) ** 2) / 2


def _layer_style_loss(a: torch.Tensor, x: torch.Tensor):
    a_flat = torch.flatten(a, start_dim=2)
    x_flat = torch.flatten(x, start_dim=2)
    b, n, m = a_flat.shape

    a_gram = torch.matmul(a_flat, torch.transpose(a_flat, 1, 2))
    x_gram = torch.matmul(x_flat, torch.transpose(x_flat, 1, 2))
    return torch.sum((a_gram - x_gram) ** 2) / (b * n * m)


def style_loss(a_list, x_list):
    assert len(a_list) == len(x_list), 'Number of original and generated feature maps must be equal'
    w_l = 1 / len(a_list)

    result = torch.zeros([])
    for a, x in zip(a_list, x_list):
        result += _layer_style_loss(a, x) * w_l
    return result


if __name__ == '__main__':
    shape = (1, 16, 32, 32)
    test1 = torch.rand(shape)
    test2 = torch.rand(shape)
    print(content_loss(test1, test2).item())
    print(_layer_style_loss(test1, test2).item())
    print(style_loss([test1, test1, test1], [test2, test2, test2]).item())
