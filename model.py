import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable

"""
Positional Encoder
"""

class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class FBV_SM(nn.Module):
    def __init__(
            self,
            d_input: int = 5,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4, ),  # 4 mar 29
            output_size: int = 2
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        self.output = nn.Linear(d_filter, output_size)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        x = self.output(x)

        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    d_input = 3
    n_freqs = 10
    log_space = True

    n_layers = 2
    d_filter = 128
    d_viewdirs = None

    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)

    model = FBV_SM(encoder.d_output, n_layers=n_layers, d_filter=d_filter)
    model.to(device)


