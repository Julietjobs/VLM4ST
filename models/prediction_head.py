"""
Prediction Head Module
Uses horizon query tokens with cross-attention to extract future predictions
Based on the approach: learnable query tokens attend to fused VLM features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonQueryPredictionHead(nn.Module):
    """
    Prediction Head with Horizon Query Tokens
    
    Architecture:
        1. T learnable horizon tokens: Q ∈ (T, d), each representing future step t
        2. Cross-attention: Q attends to fused_embeds (K/V)
        3. Output: Z ∈ (B, T, d) → Linear(d → D_out) → predictions
        4. Residual connection with DiX-Attention features
    
    """
    
    def __init__(self, T, H, W, vlm_hidden_dim=768, dix_feature_dim=256,
                 hidden_dim=512, output_channels=1, num_heads=8, dropout=0.1):
        """
        Args:
            T: Temporal length (number of future time steps to predict)
            H: Spatial height
            W: Spatial width
            vlm_hidden_dim: VLM hidden dimension (e.g., 768 for ALBEF)
            dix_feature_dim: DiX-Attention feature dimension
            hidden_dim: Hidden dimension for processing
            output_channels: Number of output channels (default: 1)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.vlm_hidden_dim = vlm_hidden_dim
        self.dix_feature_dim = dix_feature_dim
        self.output_channels = output_channels
        self.num_heads = num_heads
        
        # === Horizon Query Tokens ===
        # T learnable tokens, each representing a future time step
        # Q ∈ (T, vlm_hidden_dim)
        self.horizon_queries = nn.Parameter(
            torch.randn(1, T, vlm_hidden_dim) * 0.02
        )
        
        # === Cross-Attention Module ===
        # Query: horizon_queries
        # Key/Value: fused_embeds from VLM
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=vlm_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attn_norm = nn.LayerNorm(vlm_hidden_dim)
        
        # === Feedforward Network ===
        self.ffn = nn.Sequential(
            nn.Linear(vlm_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # === DiX Feature Processing (Residual Branch) ===
        # Process DiX features and project to spatial predictions
        self.dix_proj = nn.Sequential(
            nn.Conv2d(dix_feature_dim, dix_feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dix_feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dix_feature_dim // 2, dix_feature_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(dix_feature_dim // 4),
            nn.GELU()
        )
        
        # === Spatial Decoder ===
        # Decode from feature space to spatial predictions
        # Input: (B, T, hidden_dim) → Output: (B, T, H, W, output_channels)
        self.spatial_decode = nn.Sequential(
            nn.Linear(hidden_dim, H * W * hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === Fusion and Refinement ===
        # Combine VLM-decoded features with DiX residual
        fusion_dim = hidden_dim // 4 + dix_feature_dim // 4
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(fusion_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv3d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Conv3d(hidden_dim // 4, output_channels, kernel_size=1)
        )
        
        # === Output Refinement ===
        self.output_refine = nn.Sequential(
            nn.Conv3d(output_channels, output_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels * 2),
            nn.GELU(),
            nn.Conv3d(output_channels * 2, output_channels, kernel_size=1)
        )
    
    def forward(self, vlm_output, dix_features):
        """
        Args:
            vlm_output: Dict containing VLM outputs
                - 'fused_embeds': (B, seq_len, vlm_hidden_dim)
                - 'text_cls': (B, vlm_hidden_dim)
            dix_features: (B, T, H, W, dix_feature_dim) DiX-Attention features
            
        Returns:
            prediction: (B, T, H, W, output_channels) spatiotemporal predictions
        """
        B, T, H, W, D_dix = dix_features.shape
        
        # === Step 1: Extract VLM fused embeddings ===
        fused_embeds = vlm_output['fused_embeds']  # (B, seq_len, vlm_hidden_dim)
        
        # === Step 2: Cross-Attention with Horizon Queries ===
        # Expand horizon queries for batch
        queries = self.horizon_queries.expand(B, -1, -1)  # (B, T, vlm_hidden_dim)
        
        # Cross-attention: queries attend to fused_embeds
        attended, attn_weights = self.cross_attention(
            query=queries,           # (B, T, vlm_hidden_dim)
            key=fused_embeds,        # (B, seq_len, vlm_hidden_dim)
            value=fused_embeds,      # (B, seq_len, vlm_hidden_dim)
            need_weights=True
        )
        
        # Residual connection and norm
        queries_out = self.attn_norm(queries + attended)  # (B, T, vlm_hidden_dim)
        
        # === Step 3: Feedforward Network ===
        features = self.ffn(queries_out)  # (B, T, hidden_dim)
        features = self.ffn_norm(features)  # (B, T, hidden_dim)
        
        # === Step 4: Spatial Decoding ===
        # Decode to spatial dimensions
        spatial_features = self.spatial_decode(features)  # (B, T, H*W*hidden_dim//4)
        spatial_features = spatial_features.reshape(B, T, H, W, -1)  # (B, T, H, W, hidden_dim//4)
        
        # === Step 5: Process DiX Features (Residual) ===
        # Reshape for 2D convolution: (B*T, D_dix, H, W)
        dix_reshaped = dix_features.permute(0, 1, 4, 2, 3).reshape(B * T, D_dix, H, W)
        dix_processed = self.dix_proj(dix_reshaped)  # (B*T, D_dix//4, H, W)
        dix_processed = dix_processed.reshape(B, T, D_dix // 4, H, W)
        dix_processed = dix_processed.permute(0, 1, 3, 4, 2)  # (B, T, H, W, D_dix//4)
        
        # === Step 6: Fusion ===
        # Concatenate VLM features and DiX residual
        fused = torch.cat([spatial_features, dix_processed], dim=-1)  # (B, T, H, W, fusion_dim)
        
        # Reshape for 3D convolution: (B, C, T, H, W)
        fused = fused.permute(0, 4, 1, 2, 3)  # (B, fusion_dim, T, H, W)
        
        # === Step 7: 3D Convolution for Spatiotemporal Refinement ===
        prediction = self.fusion_conv(fused)  # (B, output_channels, T, H, W)
        
        # === Step 8: Output Refinement ===
        prediction = self.output_refine(prediction)  # (B, output_channels, T, H, W)
        
        # Reshape to (B, T, H, W, output_channels)
        prediction = prediction.permute(0, 2, 3, 4, 1)
        
        return prediction


class PredictionHead(nn.Module):
    """
    Standard Prediction Head
    Wrapper for HorizonQueryPredictionHead
    For cases where input and output temporal lengths are the same (T_in = T_out = T)
    """
    
    def __init__(self, T, H, W, vlm_hidden_dim=768, dix_feature_dim=256,
                 hidden_dim=512, output_channels=1, dropout=0.1):
        super().__init__()
        
        self.head = HorizonQueryPredictionHead(
            T=T, H=H, W=W,
            vlm_hidden_dim=vlm_hidden_dim,
            dix_feature_dim=dix_feature_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_heads=8,
            dropout=dropout
        )
    
    def forward(self, vlm_output, dix_features):
        """
        Args:
            vlm_output: Dict from VLMModule
            dix_features: (B, T, H, W, dix_feature_dim) from gate fusion
            
        Returns:
            prediction: (B, T, H, W, output_channels)
        """
        return self.head(vlm_output, dix_features)


class MultiStepPredictionHead(nn.Module):
    """
    Variable Horizon Prediction Head
    For cases where input and output temporal lengths differ (T_in ≠ T_out)
    Predicts T_out future steps from T_in input steps
    """
    
    def __init__(self, T_in, T_out, H, W, vlm_hidden_dim=768, dix_feature_dim=256,
                 hidden_dim=512, output_channels=1, dropout=0.1):
        """
        Args:
            T_in: Input temporal length
            T_out: Output temporal length (number of future steps to predict)
            H: Spatial height
            W: Spatial width
            vlm_hidden_dim: VLM hidden dimension
            dix_feature_dim: DiX feature dimension
            hidden_dim: Hidden dimension
            output_channels: Number of output channels
            dropout: Dropout rate
        """
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W
        
        # Use HorizonQueryPredictionHead with T_out queries
        self.head = HorizonQueryPredictionHead(
            T=T_out,  # Predict T_out future steps
            H=H, W=W,
            vlm_hidden_dim=vlm_hidden_dim,
            dix_feature_dim=dix_feature_dim,
            hidden_dim=hidden_dim,
            output_channels=output_channels,
            num_heads=8,
            dropout=dropout
        )
    
    def forward(self, vlm_output, dix_features):
        """
        Args:
            vlm_output: Dict from VLMModule
            dix_features: (B, T_in, H, W, dix_feature_dim)
            
        Returns:
            prediction: (B, T_out, H, W, output_channels)
        """
        # Note: dix_features may have T_in steps, but we predict T_out steps
        # We pool dix_features along time dimension to get a general representation
        B, T_in, H, W, D = dix_features.shape
        
        # Temporal pooling to get a representative feature for all future steps
        dix_pooled = dix_features.mean(dim=1, keepdim=True)  # (B, 1, H, W, D)
        dix_expanded = dix_pooled.expand(B, self.T_out, H, W, D)  # (B, T_out, H, W, D)
        
        # Use the main prediction head
        prediction = self.head(vlm_output, dix_expanded)
        
        return prediction
