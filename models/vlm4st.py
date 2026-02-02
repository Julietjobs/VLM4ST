"""
VLM4ST: Vision-Language Model for Spatiotemporal Prediction
Main model integrating all components
"""

import torch
import torch.nn as nn

from .dix_attention import DiXAttention
from .prompt_forge import PromptForge
from .gate_fusion import AdaptiveGatedFusion
from .format_transformer import DualPathFormatTransformer
from .vlm_module import VLMModule
from .prediction_head import PredictionHead, MultiStepPredictionHead


class VLM4ST(nn.Module):
    """
    VLM4ST: Complete model for spatiotemporal prediction
    
    Architecture:
        1. DiX-Attention: Captures temporal, spatial, and cross-modal patterns
        2. PromptForge: Generates pattern-aware prompts
        3. Gate Fusion: Adaptively fuses multi-scale features
        4. Format Transformer: Converts to image and sequence formats
        5. VLM Module: Encodes with vision-language model
        6. Prediction Head: Outputs spatiotemporal predictions
    
    Ablation modes:
        - None: Full model (default)
        - 'no_dix': Without DiX-Attention (only input projection)
        - 'no_prompt': Without PromptForge (zero prompts)
        - 'no_seq': Without Sequence path (visual only)
        - 'no_visual': Without Visual path (sequence only)
    """
    
    def __init__(self, config, ablation_mode=None):
        super().__init__()
        
        # Store ablation mode
        self.ablation_mode = ablation_mode
        
        # Extract configuration
        model_config = config.get('model', {})
        
        self.T = model_config.get('T', 6)  # Input temporal length (T_in)
        self.T_out = model_config.get('T_out', self.T)  # Output temporal length (T_out), defaults to T
        self.H = model_config.get('H', 32)
        self.W = model_config.get('W', 32)
        self.d_model = model_config.get('d_model', 256)
        
        dix_config = model_config.get('dix_attention', {})
        prompt_config = model_config.get('prompt_forge', {})
        fusion_config = model_config.get('gate_fusion', {})
        format_config = model_config.get('format_transformer', {})
        head_config = model_config.get('prediction_head', {})
        
        # Module 1: DiX-Attention
        self.dix_attention = DiXAttention(
            d_model=self.d_model,
            num_heads=dix_config.get('num_heads', 8),
            num_layers=dix_config.get('num_layers', 2),
            dropout=dix_config.get('dropout', 0.1)
        )
        
        # Module 2: PromptForge
        d_prompt = prompt_config.get('d_prompt', 128)
        self.prompt_forge = PromptForge(
            d_model=self.d_model,
            num_coarse=prompt_config.get('num_coarse_prototypes', 5),
            num_fine_per_coarse=prompt_config.get('num_fine_per_coarse', 2),
            d_prompt=d_prompt
        )
        
        # Module 3: Gate Fusion
        self.gate_fusion = AdaptiveGatedFusion(
            d_model=self.d_model,
            num_branches=fusion_config.get('num_branches', 4),
            dropout=dix_config.get('dropout', 0.1)
        )
        
        # Module 4: Format Transformer
        vlm_hidden_dim = format_config.get('vlm_hidden_dim', 768)
        vlm_config = config.get('vlm', {})
        vlm_image_size = vlm_config.get('image_res', 224)
        self.format_transformer = DualPathFormatTransformer(
            T=self.T,
            H=self.H,
            W=self.W,
            d_model=self.d_model,
            d_prompt=d_prompt * 3,  # 3 prompts concatenated
            vlm_hidden_dim=vlm_hidden_dim,
            spatial_pool_size=format_config.get('spatial_pool_size', 10),
            num_latents=format_config.get('num_latents', 300),
            vlm_image_size=vlm_image_size
        )
        
        # Module 5: VLM Module (pass ablation_mode for visual-only or seq-only)
        self.vlm_module = VLMModule(config, ablation_mode=ablation_mode)
        
        # Module 6: Prediction Head
        # Use MultiStepPredictionHead if T_in != T_out, otherwise use PredictionHead
        if self.T != self.T_out:
            self.prediction_head = MultiStepPredictionHead(
                T_in=self.T,
                T_out=self.T_out,
                H=self.H,
                W=self.W,
                vlm_hidden_dim=vlm_hidden_dim,
                dix_feature_dim=self.d_model,
                hidden_dim=head_config.get('hidden_dim', 512),
                output_channels=head_config.get('output_channels', 1),
                dropout=dix_config.get('dropout', 0.1)
            )
        else:
            self.prediction_head = PredictionHead(
                T=self.T,
                H=self.H,
                W=self.W,
                vlm_hidden_dim=vlm_hidden_dim,
                dix_feature_dim=self.d_model,
                hidden_dim=head_config.get('hidden_dim', 512),
                output_channels=head_config.get('output_channels', 1),
                dropout=dix_config.get('dropout', 0.1)
            )
    
    def forward(self, X, timestamps=None, return_intermediate=False):
        """
        Forward pass through the entire VLM4ST model
        
        Args:
            X: (B, T_in, H, W) input spatiotemporal data
            timestamps: (B, T_in, 2) optional timestamp information
                        [:, :, 0] = weekday (0-6), [:, :, 1] = time_of_day (0-47)
            return_intermediate: If True, return intermediate outputs
            
        Returns:
            prediction: (B, T_out, H, W, output_channels) spatiotemporal predictions
            intermediates (optional): Dict of intermediate outputs
        """
        B = X.size(0)
        
        # Stage 1: DiX-Attention (or skip if ablation_mode='no_dix')
        if self.ablation_mode == 'no_dix':
            # Only use input projection, skip temporal/spatial/cross attention
            X_input = self.dix_attention.input_proj(X.unsqueeze(-1))
            X_input = self.dix_attention.norm(X_input)
            X_temporal = X_spatial = X_cross = X_input
            # Create dummy CLS tokens for PromptForge
            dummy_cls = torch.zeros(B, self.d_model, device=X.device)
            temporal_cls = spatial_cls = cross_cls = dummy_cls
        else:
            X_temporal, X_spatial, X_cross, X_input = self.dix_attention(X, timestamps=timestamps)
            temporal_cls, spatial_cls, cross_cls = self.dix_attention.get_cls_tokens()
        
        # Stage 2: PromptForge (or skip if ablation_mode='no_prompt')
        if self.ablation_mode == 'no_prompt':
            # Use zero prompts
            d_prompt_total = self.prompt_forge.d_prompt * 3
            B_prompt = torch.zeros(B, d_prompt_total, device=X.device)
            pattern_info = {
                'temporal_weights': (torch.zeros(B, 4, device=X.device), torch.zeros(B, 4, 2, device=X.device)),
                'spatial_weights': (torch.zeros(B, 4, device=X.device), torch.zeros(B, 4, 2, device=X.device)),
                'cross_weights': (torch.zeros(B, 4, device=X.device), torch.zeros(B, 4, 2, device=X.device))
            }
        else:
            B_prompt, pattern_info = self.prompt_forge(temporal_cls, spatial_cls, cross_cls)
        
        # Stage 3: Gate Fusion
        X_fused, gate_weights = self.gate_fusion(X_temporal, X_spatial, X_cross, X_input)
        
        # Stage 4: Format Transformation
        format_outputs = self.format_transformer(X_fused, B_prompt)
        image = format_outputs['image']
        sequence = format_outputs['sequence']
        seq_mask = format_outputs['seq_mask']
        
        # Stage 5: VLM Encoding (ablation modes handled internally by VLMModule)
        vlm_output = self.vlm_module(image, sequence, seq_mask)
        
        # Stage 6: Prediction
        prediction = self.prediction_head(vlm_output, X_fused)
        
        if return_intermediate:
            intermediates = {
                'X_temporal': X_temporal,
                'X_spatial': X_spatial,
                'X_cross': X_cross,
                'X_input': X_input,
                'B_prompt': B_prompt,
                'pattern_info': pattern_info,
                'X_fused': X_fused,
                'gate_weights': gate_weights,
                'image': image,
                'sequence': sequence,
                'vlm_output': vlm_output,
                'format_outputs': format_outputs
            }
            return prediction, intermediates
        
        return prediction
    
    def get_loss(self, prediction, target, loss_config=None):
        """
        Compute prediction loss
        
        Args:
            prediction: (B, T_out, H, W, C) predicted values
            target: (B, T_out, H, W) target values
            loss_config: Loss configuration dict
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        if loss_config is None:
            loss_config = {'mse_weight': 1.0, 'mae_weight': 0.5}
        
        # Squeeze channel dimension if output_channels=1
        if prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        
        # Prediction losses
        mse_loss = nn.functional.mse_loss(prediction, target)
        mae_loss = nn.functional.l1_loss(prediction, target)
        
        # Total loss
        total_loss = (
            loss_config.get('mse_weight', 1.0) * mse_loss +
            loss_config.get('mae_weight', 0.5) * mae_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'mae_loss': mae_loss.item()
        }
        
        return total_loss, loss_dict
    
    def freeze_vlm_backbone(self):
        """Freeze VLM backbone for LoRA fine-tuning"""
        self.vlm_module.freeze_backbone()
        print("VLM backbone frozen. Only LoRA parameters will be trained.")
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen.")
    
    def get_trainable_parameters(self):
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

