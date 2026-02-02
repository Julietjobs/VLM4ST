"""
Dual-Path Format Transformer
Converts spatiotemporal data to both image and sequence formats for VLM input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalToImageProjector(nn.Module):
    """
    Temporal to Image Projector - Image Path
    Converts (B, T, H, W, D) -> (B, 3, H, W) by intelligently compressing time
    Strategy: Temporal attention + statistical features + projection
    """
    
    def __init__(self, T, H, W, d_model):
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.d_model = d_model
        
        # Temporal position encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, T, 1, 1) * 0.02
        )
        
        # First reduce feature dimension to 1
        self.feature_reduction = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, kernel_size=1),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Conv2d(d_model // 2, 1, kernel_size=1)
        )
        
        # Temporal attention module
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(T, T, kernel_size=3, padding=1, groups=T),
            nn.BatchNorm2d(T),
            nn.GELU(),
            nn.Conv2d(T, T * 3, kernel_size=1)
        )
        
        # Main projection layer
        self.channel_projection = nn.Conv2d(T, 3, kernel_size=1, bias=True)
        
        # Statistical feature fusion
        self.stat_fusion = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 3, kernel_size=1)
        )
        
        # Output normalization
        self.output_norm = nn.BatchNorm2d(3)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights uniformly"""
        nn.init.constant_(self.channel_projection.weight, 1.0 / self.T)
        if self.channel_projection.bias is not None:
            nn.init.constant_(self.channel_projection.bias, 0)
    
    def forward(self, X):
        """
        Args:
            X: (B, T, H, W, D) spatiotemporal features
            
        Returns:
            X_image: (B, 3, H, W) pseudo-RGB image
            attn_weights: (B, 3, T, H, W) attention weights
        """
        B, T, H, W, D = X.shape
        
        # Reduce feature dimension
        X_reduced = X.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W)
        X_reduced = self.feature_reduction(X_reduced)  # (B*T, 1, H, W)
        X_reduced = X_reduced.reshape(B, T, H, W)
        
        # Add temporal position encoding
        X_reduced = X_reduced + self.temporal_pos_encoding
        
        # Compute temporal attention weights
        attn_logits = self.temporal_attention(X_reduced)  # (B, T*3, H, W)
        attn_logits = attn_logits.view(B, 3, T, H, W)
        attn_weights = F.softmax(attn_logits, dim=2)  # Softmax over time
        
        # Apply attention
        X_expanded = X_reduced.unsqueeze(1)  # (B, 1, T, H, W)
        X_attended = (X_expanded * attn_weights).sum(dim=2)  # (B, 3, H, W)
        
        # Basic channel projection
        X_projected = self.channel_projection(X_reduced)  # (B, 3, H, W)
        
        # Temporal statistics
        temporal_mean = X_reduced.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        temporal_std = X_reduced.std(dim=1, keepdim=True)  # (B, 1, H, W)
        temporal_trend = X_reduced[:, -1:, :, :] - X_reduced[:, 0:1, :, :]  # (B, 1, H, W)
        
        temporal_stats = torch.cat([temporal_mean, temporal_std, temporal_trend], dim=1)
        temporal_stats_enhanced = self.stat_fusion(temporal_stats)
        
        # Fusion
        X_image = (
            0.5 * X_attended +
            0.3 * X_projected +
            0.2 * temporal_stats_enhanced
        )
        
        # Normalize
        X_image = self.output_norm(X_image)
        
        return X_image, attn_weights


class PerceiverCompressor(nn.Module):
    """
    Perceiver-style compressor using learnable latent queries
    Compresses long sequences via cross-attention
    """
    
    def __init__(self, d_model, num_latents=300, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents
        
        # Learnable latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, d_model) * 0.02
        )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, X_long):
        """
        Args:
            X_long: (B, seq_len, D) long input sequence
            
        Returns:
            latents: (B, num_latents, D) compressed representation
        """
        B = X_long.size(0)
        
        # Initialize latents
        latents = self.latent_queries.expand(B, -1, -1)
        
        # Multi-layer Perceiver processing
        for cross_attn, norm1, norm2, ffn in zip(
            self.cross_attn_layers,
            self.layer_norms1,
            self.layer_norms2,
            self.ffns
        ):
            # Cross-attention: latents as query, long sequence as key/value
            latents_attn, _ = cross_attn(
                query=latents,
                key=X_long,
                value=X_long,
                need_weights=False
            )
            
            # Residual + LayerNorm
            latents = norm1(latents + latents_attn)
            
            # FFN
            latents_ffn = ffn(latents)
            latents = norm2(latents + latents_ffn)
        
        return latents


class SpatioTemporalToSequence(nn.Module):
    """
    Spatiotemporal to Sequence Converter - Sequence Path
    Converts (B, T, H, W, D) -> (B, seq_len, D_vlm)
    Strategy: Spatial pooling + Perceiver compression
    """
    
    def __init__(self, T, H, W, d_model, spatial_pool_size=10, 
                 num_latents=300, vlm_hidden_dim=768, dropout=0.1):
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.d_model = d_model
        self.spatial_pool_size = spatial_pool_size
        self.num_latents = num_latents
        self.vlm_hidden_dim = vlm_hidden_dim
        
        self.intermediate_seq_len = T * spatial_pool_size * spatial_pool_size
        
        # Spatial adaptive pooling
        self.spatial_pooling = nn.AdaptiveAvgPool2d((spatial_pool_size, spatial_pool_size))
        
        # Spatiotemporal position embedding
        self.spatiotemporal_pos_embed = nn.Parameter(
            torch.randn(1, self.intermediate_seq_len, d_model) * 0.02
        )
        
        # Feature enhancement
        self.feature_enhance = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Perceiver compressor
        self.perceiver = PerceiverCompressor(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=2,
            dropout=dropout
        )
        
        # Project to VLM dimension
        self.proj_to_vlm = nn.Sequential(
            nn.Linear(d_model, vlm_hidden_dim),
            nn.LayerNorm(vlm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, vlm_hidden_dim) * 0.02)
    
    def forward(self, X):
        """
        Args:
            X: (B, T, H, W, D) spatiotemporal features
            
        Returns:
            X_sequence: (B, seq_len+1, D_vlm) sequence with CLS token
            seq_length: int - sequence length (without CLS)
        """
        B, T, H, W, D = X.shape
        
        # Spatial pooling (preserve temporal)
        X_spatial = X.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W)
        X_pooled = self.spatial_pooling(X_spatial)  # (B*T, D, pool_size, pool_size)
        
        pool_h, pool_w = X_pooled.shape[2], X_pooled.shape[3]
        X_pooled = X_pooled.view(B, T, D, pool_h, pool_w)
        X_pooled = X_pooled.permute(0, 1, 3, 4, 2)  # (B, T, pool_h, pool_w, D)
        
        # Flatten to sequence
        X_seq = X_pooled.reshape(B, T * pool_h * pool_w, D)
        
        # Add position encoding
        X_seq = X_seq + self.spatiotemporal_pos_embed
        
        # Feature enhancement
        X_seq = self.feature_enhance(X_seq)
        
        # Perceiver compression
        X_compressed = self.perceiver(X_seq)  # (B, num_latents, D)
        
        # Project to VLM dimension
        X_vlm = self.proj_to_vlm(X_compressed)  # (B, num_latents, vlm_hidden_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        X_sequence = torch.cat([cls_tokens, X_vlm], dim=1)  # (B, num_latents+1, vlm_hidden_dim)
        
        return X_sequence, self.num_latents


class DualPathFormatTransformer(nn.Module):
    """
    Dual-Path Format Transformer
    Generates both image and sequence formats simultaneously
    """
    
    def __init__(self, T, H, W, d_model, d_prompt, vlm_hidden_dim=768,
                 spatial_pool_size=10, num_latents=300, vlm_image_size=224):
        super().__init__()
        
        self.H = H
        self.W = W
        self.vlm_image_size = vlm_image_size
        
        # Image path projector
        self.image_projector = TemporalToImageProjector(T, H, W, d_model)
        
        # Sequence path projector
        self.sequence_projector = SpatioTemporalToSequence(
            T, H, W, d_model,
            spatial_pool_size=spatial_pool_size,
            num_latents=num_latents,
            vlm_hidden_dim=vlm_hidden_dim
        )
        
        # Prompt projection
        self.prompt_proj = nn.Linear(d_prompt, vlm_hidden_dim)
    
    def forward(self, X_fused, B_prompt):
        """
        Args:
            X_fused: (B, T, H, W, D) fused features
            B_prompt: (B, d_prompt) pattern prompt
            
        Returns:
            outputs: Dict containing:
                - 'image': (B, 3, vlm_image_size, vlm_image_size) - resized for VLM
                - 'sequence': (B, seq_len+2, vlm_dim) with CLS and PROMPT
                - 'seq_mask': (B, seq_len+2)
                - 'image_attn_weights': attention weights
                - 'seq_length': sequence length
        """
        B = X_fused.size(0)
        
        # Image path
        D_image, attn_weights = self.image_projector(X_fused)  # (B, 3, H, W)
        
        # Resize image to VLM expected size (e.g., 224x224 for ALBEF)
        if D_image.shape[2] != self.vlm_image_size or D_image.shape[3] != self.vlm_image_size:
            D_image = F.interpolate(
                D_image,
                size=(self.vlm_image_size, self.vlm_image_size),
                mode='bicubic',
                align_corners=False
            )
        
        # Sequence path
        C_sequence, seq_len = self.sequence_projector(X_fused)
        
        # Project prompt
        B_prompt_projected = self.prompt_proj(B_prompt)  # (B, vlm_hidden_dim)
        B_prompt_token = B_prompt_projected.unsqueeze(1)  # (B, 1, vlm_hidden_dim)
        
        # Insert prompt: [CLS] + [PROMPT] + [SEQ]
        C_sequence_with_prompt = torch.cat([
            C_sequence[:, :1, :],    # CLS
            B_prompt_token,          # PROMPT
            C_sequence[:, 1:, :]     # Rest of sequence
        ], dim=1)  # (B, seq_len+2, vlm_hidden_dim)
        
        # Attention mask (all ones - no padding)
        seq_mask = torch.ones(
            B, C_sequence_with_prompt.size(1),
            dtype=torch.long,
            device=X_fused.device
        )
        
        return {
            'image': D_image,
            'sequence': C_sequence_with_prompt,
            'seq_mask': seq_mask,
            'image_attn_weights': attn_weights,
            'seq_length': seq_len + 2
        }

