"""
PITiDE Model Architecture
=========================

Time-Series Dense Encoder (TiDE) with physics-informed extensions.
Based on: Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Linear -> ReLU -> Linear -> Dropout, added to a (projected) skip connection,
    then optional LayerNorm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        return self.norm(h + self.skip(x))


class ResidualStack(nn.Module):
    """Stack of ResidualBlocks with shared hidden dimension."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], hidden_dim, dims[i + 1], dropout, use_layer_norm)
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PITiDE(nn.Module):
    """
    Physics-Informed Time-Series Dense Encoder (PI-TiDE).

    Args:
        lookback: Length of input target window (L)
        horizon: Forecast length (H)
        n_covariates: Number of time-varying covariates
        temp_idx: Index of temperature covariate in feature vector
        hidden_dim: Width of encoder/decoder residual blocks
        feature_proj_dim: Projected covariate dimension
        encoder_layers: Depth of encoder stack
        decoder_layers: Depth of decoder stack
        decoder_output_dim: Per-horizon-step decoded feature width
        temporal_decoder_hidden: Hidden width in temporal decoder (None -> hidden_dim)
        dropout: Dropout rate
        use_layer_norm: Use LayerNorm after residual additions
        use_revin: Apply Reversible Instance Norm to target window
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        n_covariates: int,
        temp_idx: int,
        hidden_dim: int = 128,
        feature_proj_dim: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        decoder_output_dim: int = 16,
        temporal_decoder_hidden: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_revin: bool = False,
    ):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_covariates = n_covariates
        self.temp_idx = temp_idx
        self.use_revin = use_revin

        if temporal_decoder_hidden is None:
            temporal_decoder_hidden = hidden_dim

        # Feature projection for covariates
        self.feature_projection = ResidualBlock(
            n_covariates, hidden_dim, feature_proj_dim, dropout, use_layer_norm
        )

        # Encoder: processes concatenated past target + projected covariates
        encoder_input_dim = lookback + (lookback + horizon) * feature_proj_dim
        self.encoder = ResidualStack(
            encoder_input_dim, hidden_dim, hidden_dim, encoder_layers, dropout, use_layer_norm
        )

        # Decoder: expands latent to horizon-specific features
        self.decoder = ResidualStack(
            hidden_dim, hidden_dim, horizon * decoder_output_dim, decoder_layers, dropout, use_layer_norm
        )
        self.decoder_output_dim = decoder_output_dim

        # Temporal decoder: per-step refinement with future covariates
        self.temporal_decoder = ResidualBlock(
            decoder_output_dim + feature_proj_dim, temporal_decoder_hidden, 1, dropout, use_layer_norm
        )

        # Global linear residual (past target -> forecast)
        self.global_residual = nn.Linear(lookback, horizon)

    def forward(self, past_target: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_target: (batch, lookback)
            covariates: (batch, lookback + horizon, n_covariates)
        Returns:
            predictions: (batch, horizon)
        """
        batch_size = past_target.shape[0]

        # Optional Reversible Instance Normalization (RevIN)
        if self.use_revin:
            rev_mean = past_target.mean(dim=1, keepdim=True)
            rev_std = past_target.std(dim=1, keepdim=True).clamp_min(1e-5)
            past_target_in = (past_target - rev_mean) / rev_std
        else:
            past_target_in = past_target

        # Project covariates
        proj = self.feature_projection(covariates)  # (B, L+H, d_p)
        proj_flat = proj.reshape(batch_size, -1)    # (B, (L+H)*d_p)

        # Encoder
        enc_input = torch.cat([past_target_in, proj_flat], dim=-1)
        encoded = self.encoder(enc_input)  # (B, d_h)

        # Decoder
        decoded = self.decoder(encoded).reshape(batch_size, self.horizon, self.decoder_output_dim)

        # Temporal decoder with future covariates
        future_proj = proj[:, self.lookback:, :]  # (B, H, d_p)
        temporal_input = torch.cat([decoded, future_proj], dim=-1)
        residual_out = self.temporal_decoder(temporal_input).squeeze(-1)  # (B, H)

        # Global residual
        global_out = self.global_residual(past_target_in)

        # Combine and denormalize if RevIN used
        out = residual_out + global_out
        if self.use_revin:
            out = out * rev_std + rev_mean
        return out