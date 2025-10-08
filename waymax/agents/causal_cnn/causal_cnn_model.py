"""
Causal Convolutional Neural Network for Spatial Risk Prediction
Replaces the deterministic ellipse model with a learned spatial risk field.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CAUSAL CONVOLUTIONAL LAYERS
# ============================================================================

class CausalConv2d(nn.Module):
    """
    2D causal convolution: causal along the temporal axis, normal convolution along spatial axes.
    Input shape: (batch, channels, time, height, width)
    Output shape: (batch, out_channels, time, height_out, width_out)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation=(1,1,1), stride=(1,1,1), bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        # Underlying Conv3d: (C_in, C_out, kT, kH, kW)
        # We treat "time" as the depth dimension
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x):
        # padding along time only (left side), normal symmetric padding along H and W if needed
        pad_time = self.dilation[0] * (self.kernel_size[0] - 1)
        pad_h = self.dilation[1] * (self.kernel_size[1] - 1) // 2
        pad_w = self.dilation[2] * (self.kernel_size[2] - 1) // 2

        # Pad order for F.pad with 5D: (w_left, w_right, h_left, h_right, t_left, t_right)
        pad = (pad_w, pad_w, pad_h, pad_h, pad_time, 0)
        x_padded = F.pad(x, pad)

        return self.conv(x_padded)


class SpatialAttention(nn.Module):
    """
    Spatial attention for spatio-temporal features.
    Input: (B, C, T, H, W)
    Output: (B, C, T, H, W), attention map (B, 1, T, H, W)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 1, 1))

    def forward(self, x):
        B, C, T, H, W = x.shape
        # Collapse channels for attention map, then apply 1x1x1 conv
        pooled = x.mean(dim=1, keepdim=True)  # (B, 1, T, H, W)
        attention = torch.sigmoid(self.conv(pooled))  # (B, 1, T, H, W)
        out = x * attention  # broadcast across channels
        return out, attention


class TemporalAttention(nn.Module):
    """
    Causal temporal attention across frames.
    Input: (B, C, T, H, W) -> flattened to (B, T, F)
    Output: (B, T, F), attention weights (B, T, T)
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        # Flatten spatial dims into features
        F = C * H * W
        x_flat = x.view(B, T, F)

        # Project
        q = self.query(x_flat)   # (B, T, hidden)
        k = self.key(x_flat)     # (B, T, hidden)
        v = self.value(x_flat)   # (B, T, hidden)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # (B, T, T)

        # Causal mask (lower-triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        attended = torch.matmul(attn_weights, v)  # (B, T, hidden)

        return attended, attn_weights


# ============================================================================
# MAIN CAUSAL RISK CNN
# ============================================================================

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple


class CausalConv2D(nn.Module):
    """2D convolution with causal padding (no future leakage)."""
    features: int
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        # Pad only top/left (causal in both dimensions)
        pad_h = self.kernel_size[0] - 1
        pad_w = self.kernel_size[1] - 1
        x = jnp.pad(x, ((0, 0), (pad_h, 0), (pad_w, 0), (0, 0)))
        return nn.Conv(features=self.features, kernel_size=self.kernel_size)(x)


class SpatialAttention(nn.Module):
    """Attention mechanism for spatial risk regions."""
    @nn.compact
    def __call__(self, x):
        # x: (batch, height, width, channels)
        attn = nn.Conv(features=1, kernel_size=(1, 1))(x)
        attn = nn.sigmoid(attn)
        attended = x * attn
        return attended, attn


class TemporalAttention(nn.Module):
    """Attention over observation history (causal)."""
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (batch, time, features)
        batch, time, features = x.shape

        # Linear projections
        query = nn.Dense(self.hidden_dim)(x)
        key = nn.Dense(self.hidden_dim)(x)
        value = nn.Dense(self.hidden_dim)(x)

        # Attention scores
        scores = jnp.matmul(query, key.transpose(0, 2, 1)) / jnp.sqrt(self.hidden_dim)

        # Causal mask: allow only current + past
        mask = jnp.tril(jnp.ones((time, time)))
        scores = jnp.where(mask[None, :, :], scores, -1e9)

        attn_weights = nn.softmax(scores, axis=-1)
        attended = jnp.matmul(attn_weights, value)
        return attended, attn_weights


class CausalRiskCNN(nn.Module):
    """
    Causal CNN with temporal + spatial attention
    to predict a spatial risk grid around ego vehicle.
    """
    grid_size: int = 64
    grid_range: float = 50.0
    history_length: int = 10
    hidden_dims: Tuple[int, ...] = (64, 128, 256, 128, 64)

    @nn.compact
    def __call__(self, observations, training: bool = False):
        """
        Args:
            observations: (batch, history_length, obs_features)
                         Features: [rel_x, rel_y, rel_vx, rel_vy, ego_speed, lead_speed]
        Returns:
            risk_grid: (batch, grid_size, grid_size, 1) with risk values [0, 1]
            attention_maps: dict with 'temporal_attention', 'spatial_attention'
        """
        batch_size = observations.shape[0]

        # --- Temporal attention over history ---
        temporal_features, temporal_attention = TemporalAttention(
            hidden_dim=64
        )(observations)

        # Use most recent attended features
        current_features = temporal_features[:, -1, :]

        # --- Initialize spatial grid (low-res embedding) ---
        spatial_dim = 8  # starting resolution
        reshaped = nn.Dense(spatial_dim * spatial_dim * 32)(current_features)
        reshaped = reshaped.reshape(batch_size, spatial_dim, spatial_dim, 32)

        x = reshaped
        attention_maps = {'temporal_attention': temporal_attention}

        # --- Causal CNN with spatial attention in middle ---
        for i, dim in enumerate(self.hidden_dims):
            x = CausalConv2D(features=dim, kernel_size=(3, 3))(x)

            if i == len(self.hidden_dims) // 2:
                x, spatial_attn = SpatialAttention()(x)
                attention_maps['spatial_attention'] = spatial_attn

            if training:
                x = nn.Dropout(rate=0.1)(x, deterministic=not training)

        # --- Upsample to target grid size ---
        while x.shape[1] < self.grid_size:
            x = jax.image.resize(
                x,
                shape=(batch_size, x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
                method='bilinear'
            )
            x = CausalConv2D(features=32, kernel_size=(3, 3))(x)

        if x.shape[1] != self.grid_size:
            x = jax.image.resize(
                x,
                shape=(batch_size, self.grid_size, self.grid_size, x.shape[3]),
                method='bilinear'
            )

        # --- Final risk prediction ---
        risk_grid = nn.Conv(features=1, kernel_size=(1, 1))(x)
        risk_grid = nn.sigmoid(risk_grid)

        return risk_grid, attention_maps

    def get_risk_at_position(self, risk_grid, x_meters, y_meters):
        """Extract risk value at specific (x, y) position in meters."""
        cell_size = 2 * self.grid_range / self.grid_size
        grid_x = int((x_meters + self.grid_range) / cell_size)
        grid_y = int((y_meters + self.grid_range) / cell_size)

        grid_x = jnp.clip(grid_x, 0, self.grid_size - 1)
        grid_y = jnp.clip(grid_y, 0, self.grid_size - 1)

        return risk_grid[0, grid_x, grid_y, 0]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_observation_history(
    state,
    ego_idx: int,
    lead_idx: int,
    history_length: int = 10
):
    """
    Extract observation history from Waymax state.
    
    Returns:
        observations: (history_length, 6) array with features:
                     [rel_x, rel_y, rel_vx, rel_vy, ego_speed, lead_speed]
    """
    current_timestep = state.timestep
    
    observations = []
    for t in range(max(0, current_timestep - history_length + 1), current_timestep + 1):
        # Get positions
        ego_pos = state.sim_trajectory.xy[ego_idx, t]
        lead_pos = state.sim_trajectory.xy[lead_idx, t]
        
        # Get velocities
        ego_vel = jnp.array([
            state.sim_trajectory.vel_x[ego_idx, t],
            state.sim_trajectory.vel_y[ego_idx, t]
        ])
        lead_vel = jnp.array([
            state.sim_trajectory.vel_x[lead_idx, t],
            state.sim_trajectory.vel_y[lead_idx, t]
        ])
        
        # Compute relative features
        rel_pos = lead_pos - ego_pos
        rel_vel = ego_vel - lead_vel
        ego_speed = jnp.linalg.norm(ego_vel)
        lead_speed = jnp.linalg.norm(lead_vel)
        
        obs = jnp.concatenate([
            rel_pos,
            rel_vel,
            jnp.array([ego_speed, lead_speed])
        ])
        observations.append(obs)
    
    # Pad if needed (for early timesteps)
    while len(observations) < history_length:
        observations.insert(0, observations[0])
    
    return jnp.stack(observations)