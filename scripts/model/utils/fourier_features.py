import torch
import torch.nn as nn
from math import ceil, log10

#H：x_min=0.01, x_max=16.0
#C：x_min=0.01, x_max=230.0
class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=False, funcs='both', sigma=10, num_freqs=512):

        assert strategy in {'random', 'voronov_et_al', 'lin_float_int', 'log_spaced'}
        assert funcs in {'both', 'sin', 'cos'}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == 'random':
            self.b = torch.randn(num_freqs) * sigma
        if self.strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)],
            )
        elif self.strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, ceil(x_max), 1)],
            )
        elif self.strategy == "log_spaced":
            shifts = torch.logspace(
                log10(x_min),
                log10(x_max),
                steps=num_freqs,
            )
            self.b = shifts.reciprocal()
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('fourier_frequencies', self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]


class RBFExpansion(nn.Module):
    """Expand each scalar into Gaussian radial-basis-function features.

    The centers are fixed and linearly spaced between ``x_min`` and ``x_max``.
    An input with shape ``(...,)`` is returned with shape ``(..., n)``.
    Integer inputs are converted to floating point before computing the
    expansion.
    """

    def __init__(self, x_min, x_max, n, sigma):
        super().__init__()

        if x_min > x_max:
            raise ValueError("x_min must be less than or equal to x_max")
        if n <= 0:
            raise ValueError("n must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.register_buffer(
            "centers",
            torch.linspace(float(x_min), float(x_max), steps=n),
        )
        self.sigma = float(sigma)

    def forward(self, x):
        x = torch.as_tensor(x, device=self.centers.device)
        x = x.to(dtype=self.centers.dtype)
        distances = x.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (distances / self.sigma) ** 2)
