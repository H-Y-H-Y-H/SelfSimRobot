import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable


class NeDF(nn.Module):
    def __init__(
            self,
            d_input: int = 3,
            n_layers: int = 6,
            d_filter: int = 256,
            skip: Tuple[int] = ()  # default no skip connection
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip
             else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        self.output = nn.Linear(d_filter, 1)

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
