"""
Adaptive Gated Fusion Module
Dynamically fuses temporal, spatial, cross-modal, and input features
"""

import torch
import torch.nn as nn


class AdaptiveGatedFusion(nn.Module):
    """
    Adaptive Gated Fusion with learnable gates
    Fuses multiple feature branches with data-dependent weights
    """
    
    def __init__(self, d_model, num_branches=4, dropout=0.1):
        super().__init__()
        self.num_branches = num_branches
        self.d_model = d_model
        
        # Gate network: generates weights for each branch
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * num_branches, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Branch projection layers for feature alignment
        self.branch_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            for _ in range(num_branches)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, X_temporal, X_spatial, X_cross, X_input):
        """
        Args:
            X_temporal: (B, T, H, W, D) temporal features
            X_spatial: (B, T, H, W, D) spatial features
            X_cross: (B, T, H, W, D) cross-modal features
            X_input: (B, T, H, W, D) input embeddings
            
        Returns:
            X_fused: (B, T, H, W, D) fused features
            gate_weights: (B, T, H, W, num_branches) gating weights
        """
        B, T, H, W, D = X_temporal.shape
        
        # Project each branch
        X_temp_proj = self.branch_projs[0](X_temporal)
        X_spat_proj = self.branch_projs[1](X_spatial)
        X_cross_proj = self.branch_projs[2](X_cross)
        X_input_proj = self.branch_projs[3](X_input)
        
        # Concatenate all branches for gate computation
        all_features = torch.cat([
            X_temp_proj, X_spat_proj, X_cross_proj, X_input_proj
        ], dim=-1)  # (B, T, H, W, num_branches * D)
        
        # Compute gate weights
        gate_weights = self.gate_network(all_features)  # (B, T, H, W, num_branches)
        
        # Weighted fusion
        X_fused = (
            gate_weights[..., 0:1] * X_temp_proj +
            gate_weights[..., 1:2] * X_spat_proj +
            gate_weights[..., 2:3] * X_cross_proj +
            gate_weights[..., 3:4] * X_input_proj
        )  # (B, T, H, W, D)
        
        # Final projection
        X_fused = self.output_proj(X_fused)
        
        return X_fused, gate_weights

