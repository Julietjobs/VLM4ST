"""
Disentangled Cross-Attention (DiX-Attention) Module
Captures temporal, spatial, and cross-modal spatiotemporal dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """
    Temporal Self-Attention: Models temporal dependencies for each spatial location
    For each spatial region, model the temporal sequence independently
    """
    
    def __init__(self, d_model, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        # CLS token for temporal attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Cache for storing the output cls_token
        self.last_cls_output = None
        
        # Timestamp embeddings (weekday: 0-6, time_of_day: 0-47)
        self.weekday_embed = nn.Embedding(7, d_model // 2)
        self.timeofday_embed = nn.Embedding(48, d_model // 2)
        self.timestamp_proj = nn.Linear(d_model, d_model)
    
    def forward(self, X, timestamps=None):
        """
        Args:
            X: (B, T, H, W, D) spatiotemporal features
            timestamps: (B, T, 2) optional timestamp information
                        [:, :, 0] = weekday (0-6), [:, :, 1] = time_of_day (0-47)
            
        Returns:
            X_temporal: (B, T, H, W, D) temporally enhanced features
        """
        B, T, H, W, D = X.shape
        
        # Add timestamp embeddings if provided
        if timestamps is not None:
            weekday = timestamps[:, :, 0].long()  # (B, T)
            timeofday = timestamps[:, :, 1].long()  # (B, T)
            
            # Get embeddings
            wd_emb = self.weekday_embed(weekday)  # (B, T, D//2)
            tod_emb = self.timeofday_embed(timeofday)  # (B, T, D//2)
            
            # Concatenate and project
            time_emb = torch.cat([wd_emb, tod_emb], dim=-1)  # (B, T, D)
            time_emb = self.timestamp_proj(time_emb)  # (B, T, D)
            
            # Broadcast to spatial dimensions and add to input
            time_emb = time_emb.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, D)
            X = X + time_emb  # Broadcasting: (B, T, H, W, D)
        
        # Reshape: treat each spatial location as a separate sequence
        # (B, T, H, W, D) -> (B*H*W, T, D)
        X_reshaped = X.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)
        
        # Add cls_token to each sequence
        cls_tokens = self.cls_token.expand(B * H * W, -1, -1)  # (B*H*W, 1, D)
        X_with_cls = torch.cat([cls_tokens, X_reshaped], dim=1)  # (B*H*W, T+1, D)
        
        # Apply temporal attention
        X_temporal = self.temporal_encoder(X_with_cls)  # (B*H*W, T+1, D)
        
        # Extract cls_token and sequence
        cls_out = X_temporal[:, 0, :]  # (B*H*W, D)
        seq_out = X_temporal[:, 1:, :]  # (B*H*W, T, D)
        
        # Store cls_token as module attribute (average across spatial locations)
        self.last_cls_output = cls_out.reshape(B, H, W, D).mean(dim=[1, 2])  # (B, D)
        
        # Reshape back
        seq_out = seq_out.reshape(B, H, W, T, D).permute(0, 3, 1, 2, 4)  # (B, T, H, W, D)
        
        # Residual connection (note: X already includes timestamp information if provided)
        X_temporal = self.norm(seq_out + X)
        
        return X_temporal


class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention: Models spatial dependencies for each time step
    For each time step, model all spatial regions as a sequence
    """
    
    def __init__(self, d_model, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Transformer encoder for spatial modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        # CLS token for spatial attention
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Cache for storing the output cls_token
        self.last_cls_output = None
    
    def forward(self, X):
        """
        Args:
            X: (B, T, H, W, D) spatiotemporal features
            
        Returns:
            X_spatial: (B, T, H, W, D) spatially enhanced features
        """
        B, T, H, W, D = X.shape
        
        # Reshape: treat each time step's spatial grid as a sequence
        # (B, T, H, W, D) -> (B*T, H*W, D)
        X_reshaped = X.reshape(B * T, H * W, D)
        
        # Add cls_token to each sequence
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # (B*T, 1, D)
        X_with_cls = torch.cat([cls_tokens, X_reshaped], dim=1)  # (B*T, H*W+1, D)
        
        # Apply spatial attention
        X_spatial = self.spatial_encoder(X_with_cls)  # (B*T, H*W+1, D)
        
        # Extract cls_token and sequence
        cls_out = X_spatial[:, 0, :]  # (B*T, D)
        seq_out = X_spatial[:, 1:, :]  # (B*T, H*W, D)
        
        # Store cls_token as module attribute (average across time)
        self.last_cls_output = cls_out.reshape(B, T, D).mean(dim=1)  # (B, D)
        
        # Reshape back
        seq_out = seq_out.reshape(B, T, H, W, D)
        
        # Residual connection
        X_spatial = self.norm(seq_out + X)
        
        return X_spatial

class BidirectionalSTCrossAttention(nn.Module):
    """
    Bidirectional Spatiotemporal Cross-Attention
    Captures interactions between temporal and spatial dimensions:
    - Temporal as Query, Spatial as Key/Value
    - Spatial as Query, Temporal as Key/Value
    pool_size: the size of the spatial pooling(default: 7), default is 7 because the spatial dimension is 28, 28/7 = 4, 4*4 = 16, which is a good compromise between memory and performance.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1, pool_size=7):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # Temporal -> Spatial cross attention
        self.t2s_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Spatial -> Temporal cross attention  
        self.s2t_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # CLS tokens for t2s and s2t cross attention
        self.t2s_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.s2t_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Cache for storing the output cls_token
        self.last_cls_output = None
    
    def forward(self, X_temporal, X_spatial):
        """
        Args:
            X_temporal: (B, T, H, W, D) temporal features
            X_spatial: (B, T, H, W, D) spatial features
            
        Returns:
            X_cross: (B, T, H, W, D) cross-modal features
        """
        B, T, H, W, D = X_temporal.shape
        
        # ===== Temporal -> Spatial Cross-Attention =====
        # Idea: For each time step, let temporal info attend to spatial patterns
        # Reshape temporal: (B*T, 1, D) as query
        # Reshape spatial: (B*T, H*W, D) or (B*T, pool_size^2, D) as key/value
        
        temp_query = X_temporal.mean(dim=[2, 3])  # (B, T, D) - temporal global features
        temp_query = temp_query.reshape(B * T, 1, D)  # (B*T, 1, D)
        
        # Add t2s cls_token to query
        t2s_cls = self.t2s_cls_token.expand(B * T, -1, -1)  # (B*T, 1, D)
        temp_query = torch.cat([t2s_cls, temp_query], dim=1)  # (B*T, 2, D)
        
        # Pool spatial dimensions to reduce memory
        spat_kv = X_spatial.permute(0, 1, 4, 2, 3)  # (B, T, D, H, W)
        spat_kv = spat_kv.reshape(B * T, D, H, W)
        spat_kv = self.spatial_pool(spat_kv)  # (B*T, D, pool_size, pool_size)
        spat_kv = spat_kv.flatten(2).permute(0, 2, 1)  # (B*T, pool_size^2, D)
        
        t2s_out, _ = self.t2s_cross_attn(
            query=temp_query,     # (B*T, 2, D)
            key=spat_kv,          # (B*T, pool_size^2 or H*W, D)
            value=spat_kv
        )  # (B*T, 2, D)
        
        t2s_cls_out = t2s_out[:, 0, :]  # (B*T, D) - extract cls_token
        t2s_feat_out = t2s_out[:, 1, :]  # (B*T, D) - extract feature
        t2s_feat_out = t2s_feat_out.reshape(B, T, 1, 1, D).expand(B, T, H, W, D)
        
        # ===== Spatial -> Temporal Cross-Attention =====
        # Idea: For each spatial location, let spatial info attend to temporal evolution
        # Reshape spatial: (B*H*W, 1, D) as query
        # Reshape temporal: (B*H*W, T, D) as key/value
        
        spat_query = X_spatial.mean(dim=1)  # (B, H, W, D) - spatial snapshot
        spat_query = spat_query.reshape(B * H * W, 1, D)  # (B*H*W, 1, D)
        
        # Add s2t cls_token to query
        s2t_cls = self.s2t_cls_token.expand(B * H * W, -1, -1)  # (B*H*W, 1, D)
        spat_query = torch.cat([s2t_cls, spat_query], dim=1)  # (B*H*W, 2, D)
        
        temp_kv = X_temporal.permute(0, 2, 3, 1, 4)  # (B, H, W, T, D)
        temp_kv = temp_kv.reshape(B * H * W, T, D)  # (B*H*W, T, D)
        
        s2t_out, _ = self.s2t_cross_attn(
            query=spat_query,     # (B*H*W, 2, D)
            key=temp_kv,          # (B*H*W, T, D)
            value=temp_kv
        )  # (B*H*W, 2, D)
        
        s2t_cls_out = s2t_out[:, 0, :]  # (B*H*W, D) - extract cls_token
        s2t_feat_out = s2t_out[:, 1, :]  # (B*H*W, D) - extract feature
        s2t_feat_out = s2t_feat_out.reshape(B, H, W, 1, D).permute(0, 3, 1, 2, 4).expand(B, T, H, W, D)
        
        # ===== Fusion =====
        combined = torch.cat([t2s_feat_out, s2t_feat_out], dim=-1)  # (B, T, H, W, 2*D)
        X_cross = self.fusion(combined)  # (B, T, H, W, D)
        
        # Residual connection
        X_cross = self.norm(X_cross + X_temporal + X_spatial)
        
        # Fuse the cls_tokens from t2s and s2t
        t2s_cls_global = t2s_cls_out.reshape(B, T, D).mean(dim=1)  # (B, D)
        s2t_cls_global = s2t_cls_out.reshape(B, H, W, D).mean(dim=[1, 2])  # (B, D)
        
        # Store fused cls_token as module attribute
        self.last_cls_output = (t2s_cls_global + s2t_cls_global) / 2.0  # (B, D)
        
        return X_cross


class DiXAttention(nn.Module):
    """
    Disentangled Cross-Attention Module
    Combines Temporal, Spatial, and Bidirectional Cross-Attention
    """
    
    def __init__(self, d_model, num_heads, num_layers=2, dropout=0.1, pool_size=7):
        super().__init__()
        
        # Three attention modules
        self.temporal_encoder = TemporalSelfAttention(
            d_model, num_heads, num_layers, dropout
        )
        
        self.spatial_encoder = SpatialSelfAttention(
            d_model, num_heads, num_layers, dropout
        )
        
        self.cross_encoder = BidirectionalSTCrossAttention(
            d_model, num_heads, dropout, pool_size=pool_size
        )
        
        # Input embedding projection
        self.input_proj = nn.Linear(1, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, X, timestamps=None):
        """
        Args:
            X: (B, T, H, W) input spatiotemporal data
            timestamps: (B, T, 2) optional timestamp information
                        [:, :, 0] = weekday (0-6), [:, :, 1] = time_of_day (0-47)
            
        Returns:
            X_temporal: (B, T, H, W, D) temporal features
            X_spatial: (B, T, H, W, D) spatial features
            X_cross: (B, T, H, W, D) cross-modal features
            X_input: (B, T, H, W, D) input embeddings
        """
        B, T, H, W = X.shape
        
        # Project input to feature space
        X_input = self.input_proj(X.unsqueeze(-1))  # (B, T, H, W, D)
        X_input = self.norm(X_input)
        
        # Stage 1: Independent temporal and spatial encoding
        X_temporal = self.temporal_encoder(X_input, timestamps=timestamps)
        X_spatial = self.spatial_encoder(X_input)
        
        # Stage 2: Cross-modal interaction
        X_cross = self.cross_encoder(X_temporal, X_spatial)
        
        return X_temporal, X_spatial, X_cross, X_input
    
    def get_cls_tokens(self):
        """
        Extract CLS tokens from the three attention modules
        Call this after forward() to get the cls_tokens
        
        Returns:
            temporal_cls: (B, D) - CLS token from temporal attention
            spatial_cls: (B, D) - CLS token from spatial attention
            cross_cls: (B, D) - Fused CLS token from cross attention (t2s + s2t)
        """
        return (
            self.temporal_encoder.last_cls_output,
            self.spatial_encoder.last_cls_output,
            self.cross_encoder.last_cls_output
        )

