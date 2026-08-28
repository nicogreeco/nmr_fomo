import torch
import torch.nn as nn
from math import log10


class FourierFeatures(nn.Module):
    """Encode scalar values with fixed or trainable sinusoidal frequencies."""

    def __init__(
        self,
        strategy,
        x_min,
        x_max,
        resolution=None,
        trainable=False,
        funcs="both",
        sigma=10,
        num_freqs=512,
    ):
        if strategy not in {
            "random",
            "voronov_et_al",
            "lin_float_int",
            "log_spaced",
        }:
            raise ValueError(f"unknown Fourier strategy: {strategy!r}")
        if funcs not in {"both", "sin", "cos"}:
            raise ValueError(f"unknown Fourier function selection: {funcs!r}")
        if x_min >= x_max:
            raise ValueError("x_min must be smaller than x_max")
        if num_freqs is None or num_freqs < 1:
            raise ValueError("num_freqs must be a positive integer")

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        value_span = float(x_max - x_min)
        if resolution is None:
            resolution = value_span / num_freqs
        resolution = float(resolution)
        if resolution <= 0 or resolution > value_span:
            raise ValueError(
                "resolution must be positive and no larger than x_max-x_min"
            )

        if strategy == "random":
            frequencies = torch.randn(num_freqs) * sigma
        elif strategy == "lin_float_int":
            frequencies = torch.linspace(
                1.0 / value_span,
                1.0 / resolution,
                steps=num_freqs,
            )
        else:
            # Log spacing applies to positive periods, not to the physical input
            # range. Chemical shifts themselves may therefore be negative.
            periods = torch.logspace(
                log10(resolution),
                log10(value_span),
                steps=num_freqs,
            )
            frequencies = periods.reciprocal()

        self.fourier_frequencies = nn.Parameter(
            frequencies.unsqueeze(0),
            requires_grad=trainable,
        )

    def forward(self, x):
        x = 2 * torch.pi * x @ self.fourier_frequencies
        if self.funcs == "both":
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == "cos":
            x = torch.cos(x)
        elif self.funcs == "sin":
            x = torch.sin(x)
        return x

    def num_features(self):
        num_frequencies = self.fourier_frequencies.shape[1]
        return num_frequencies if self.funcs != "both" else 2 * num_frequencies


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
