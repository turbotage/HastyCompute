import torch
import torch.nn as nn
import torch.nn.functional as F

from cuequivariance import Irreps
from cuequivariance.nn import EquivariantConv

class EquivariantMixBlock(nn.Module):
    def __init__(self, irreps, scalar_channels, vector_channels):
        super().__init__()

        self.irreps = irreps

        # Equivariant convolution
        self.conv = EquivariantConv(
            irreps,
            irreps,
            kernel_size=3,
            padding=1
        )

        # Scalar MLP (controls mixing)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_channels + vector_channels, scalar_channels),
            nn.SiLU(),
            nn.Linear(scalar_channels, scalar_channels)
        )

        self.gate = nn.Sigmoid()

        self.scalar_channels = scalar_channels
        self.vector_channels = vector_channels

    def forward(self, x):
        """
        x is assumed to be structured tensor compatible with irreps
        """

        # Split irreps (pseudo-API, adjust if needed)
        s, v = x.split_irreps()   # s: (..., ns), v: (..., nv, 3)

        # Compute invariant from vectors
        v_norm = torch.norm(v, dim=-1)  # (..., nv)

        # Concatenate scalar + invariant vector info
        s_input = torch.cat([s, v_norm], dim=-1)

        # Scalar update
        s_update = self.scalar_mlp(s_input)

        # Equivariant conv
        x_conv = self.conv(x)
        s_conv, v_conv = x_conv.split_irreps()

        # Gate vectors using scalars
        gate = self.gate(s_update).unsqueeze(-1)  # (..., ns, 1)

        # Match dimensions if needed
        gate = gate[..., :v_conv.shape[-2], :]  # simple truncation

        v_out = v_conv * gate

        s_out = s_conv + s_update

        return x.combine_irreps(s_out, v_out)

class EquivariantUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode="trilinear"):
        super().__init__()
        self.scale = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale,
            mode=self.mode,
            align_corners=False
        )

class EquivariantSuperResNet(nn.Module):
    def __init__(
        self,
        ns=4,                  # scalar channels
        nv=4,                  # vector channels
        num_blocks_low=3,      # blocks before upsampling
        num_blocks_high=2,     # blocks after upsampling
        scale_factor=2,
        use_skip=True
    ):
        super().__init__()

        # Irreps
        self.irreps_in = Irreps("3x0e + 1x1o")
        self.irreps_hidden = Irreps(f"{ns}x0e + {nv}x1o")
        self.irreps_out = Irreps("1x1o")

        # Lift
        self.lift = EquivariantConv(
            self.irreps_in,
            self.irreps_hidden,
            kernel_size=3,
            padding=1
        )

        # Low-res blocks
        self.blocks_low = nn.ModuleList([
            EquivariantMixBlock(self.irreps_hidden, ns, nv)
            for _ in range(num_blocks_low)
        ])

        # Upsample
        self.upsample = EquivariantUpsample(scale_factor)

        # High-res blocks
        self.blocks_high = nn.ModuleList([
            EquivariantMixBlock(self.irreps_hidden, ns, nv)
            for _ in range(num_blocks_high)
        ])

        # Projection to velocity
        self.proj = EquivariantConv(
            self.irreps_hidden,
            self.irreps_out,
            kernel_size=1
        )

        self.use_skip = use_skip

    def forward(self, x):
        """
        x shape: [B, T, Z, Y, X, channels]
        """

        # Merge batch + time for now (simplest strategy)
        B, T, Z, Y, X, C = x.shape
        x = x.view(B * T, Z, Y, X, C)

        # Lift
        x = self.lift(x)

        # Low-res processing
        for block in self.blocks_low:
            x = block(x)

        skip = x if self.use_skip else None

        # Upsample
        x = self.upsample(x)

        # High-res refinement
        for block in self.blocks_high:
            x = block(x)

        # Optional skip connection (upsampled)
        if self.use_skip:
            skip_up = self.upsample(skip)
            x = x + skip_up

        # Project to velocity
        v = self.proj(x)

        # Reshape back
        _, Z2, Y2, X2, _ = v.shape
        v = v.view(B, T, Z2, Y2, X2, -1)

        return v