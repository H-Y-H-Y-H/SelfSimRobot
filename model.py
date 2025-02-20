import torch
from torch import nn

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
    def __init__(self,
                 encoder=None,
                 d_input: int = 5,
                 d_filter: int = 128,
                 output_size: int = 2):
        super(FBV_SM, self).__init__()

        self.d_input = d_input

        self.act = nn.functional.relu
        self.encoder = encoder

        # Initialize layers
        if self.encoder ==None:
            pos_encoder_d = 3
            cmd_encoder_d = d_input-3
            self.feed_forward = nn.Sequential(
                nn.Linear(d_filter * 2+d_input, d_filter),
                nn.ReLU(),
                nn.Linear(d_filter, d_filter // 4)
            )
        else:
            n_freqs = self.encoder.n_freqs
            pos_encoder_d = (n_freqs*2+1)*3
            cmd_encoder_d = (n_freqs*2+1)*(d_input-3)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_filter*2, d_filter),
                nn.ReLU(),
                nn.Linear(d_filter,d_filter//4)
            )

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_encoder_d, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter,d_filter),
        )

        self.cmd_encoder = nn.Sequential(
            nn.Linear(cmd_encoder_d, d_filter),
            nn.ReLU(),
            nn.Linear(d_filter,d_filter),
        )

        self.output = nn.Linear(d_filter//4, output_size)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder !=None:
            x_pos = self.encoder(x[:,:3])
            x_cmd = self.encoder(x[:,3:])
            x_pos = self.pos_encoder(x_pos)
            x_cmd = self.cmd_encoder(x_cmd)
            x = self.feed_forward(torch.cat((x_pos,x_cmd),dim=1))
        else:
            x_pos = self.pos_encoder(x[:,:3])
            x_cmd = self.cmd_encoder(x[:,3:])
            x = self.feed_forward(torch.cat((x_pos, x_cmd,x), dim=1))

        return self.output(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    d_input = 5
    n_freqs = 10
    log_space = True


    encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
    print(encoder.d_output)
    model = FBV_SM(encoder = encoder)
    model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)


