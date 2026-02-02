"""
VLM Module: ALBEF-based Vision-Language Model with LoRA fine-tuning
Uses ALBEF's visual and text encoders for multimodal fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import os

# Import from local models directory
from .vit import VisionTransformer, interpolate_pos_embed
from .xbert import BertConfig, BertModel


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer
    Implements parameter-efficient fine-tuning via low-rank decomposition
    
    For a pretrained weight matrix W ∈ R^(d*k):
        h = Wx + (BA)x
    where B ∈ R^(d*r), A ∈ R^(r*k), and r << min(d, k)
    """
    
    def __init__(self, original_layer, rank=16, alpha=32):
        """
        Args:
            original_layer: Original nn.Linear layer to adapt
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get dimensions
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices: B and A
        # Initialize A with Gaussian, B with zeros (as in paper)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass: h = Wx + scaling * B @ A @ x
        """
        # Original transformation
        result = self.original_layer(x)
        
        # LoRA adaptation
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        result = result + self.scaling * lora_out
        
        return result


class MTLLoRALayer(nn.Module):
    """
    MTL-LoRA (Multi-Task Learning LoRA) Layer
    
    Enhances LoRA with task-adaptive parameters:
        h = Wx + B_t @ Λ_t @ A @ x
    where:
        - A: shared down-projection (all tasks)
        - Λ_t: task-specific diagonal transformation (in low-rank space)
        - B_t: task-specific up-projection (weighted combination of experts)
    """
    
    def __init__(self, original_layer, rank=16, alpha=32, num_tasks=14, num_experts=4):
        """
        Args:
            original_layer: Original nn.Linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            num_tasks: Number of tasks/datasets
            num_experts: Number of expert B matrices
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Shared down-projection A
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Task-specific diagonal transformations Λ_t
        self.task_lambdas = nn.Parameter(torch.ones(num_tasks, rank))
        
        # Expert up-projections B
        self.lora_B_experts = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        
        # Task-to-expert weights
        self.task_expert_weights = nn.Parameter(torch.randn(num_tasks, num_experts) * 0.01)
        
        # Gate network for dynamic task identification
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_tasks)
        )
        
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward: h = Wx + scaling * B_t @ Λ_t @ A @ x
        Task is identified automatically from input features
        """
        result = self.original_layer(x)
        
        # Identify task dynamically (B, seq_len, in_features)
        # Use mean pooling over sequence for task identification
        x_pooled = x.mean(dim=1) if x.dim() == 3 else x  # (B, in_features)
        task_logits = self.gate_net(x_pooled)  # (B, num_tasks)
        task_weights = F.softmax(task_logits, dim=-1)  # (B, num_tasks)
        
        # Compute task-specific lambda (weighted combination)
        # task_weights: (B, num_tasks), task_lambdas: (num_tasks, rank)
        lambda_t = task_weights @ self.task_lambdas  # (B, rank)
        
        # Compute task-specific B (weighted combination of experts)
        # task_weights: (B, num_tasks), task_expert_weights: (num_tasks, num_experts)
        expert_weights = task_weights @ self.task_expert_weights  # (B, num_experts)
        expert_weights = F.softmax(expert_weights, dim=-1)  # (B, num_experts)
        
        # lora_B_experts: (num_experts, out_features, rank)
        # expert_weights: (B, num_experts) -> (B, num_experts, 1, 1)
        B_t = (expert_weights.unsqueeze(-1).unsqueeze(-1) * 
               self.lora_B_experts.unsqueeze(0)).sum(dim=1)  # (B, out_features, rank)
        
        # LoRA forward: A @ x
        lora_out = x @ self.lora_A.T  # (B, seq_len, rank) or (B, rank)
        
        # Apply task-specific lambda
        if lora_out.dim() == 3:  # (B, seq_len, rank)
            lora_out = lora_out * lambda_t.unsqueeze(1)  # (B, seq_len, rank)
            # Apply B_t
            lora_out = torch.bmm(lora_out, B_t.transpose(-2, -1))  # (B, seq_len, out_features)
        else:  # (B, rank)
            lora_out = lora_out * lambda_t  # (B, rank)
            # Apply B_t
            lora_out = torch.bmm(lora_out.unsqueeze(1), B_t.transpose(-2, -1)).squeeze(1)  # (B, out_features)
        
        result = result + self.scaling * lora_out
        return result


class MOELoRALayer(nn.Module):
    """
    MoE-LoRA (Mixture of Experts LoRA) Layer
    
    Uses multiple expert LoRA pairs with gating:
        h = Wx + Σ_i w_i * (B_i @ A_i @ x)
    where w_i are task-dependent expert weights
    """
    
    def __init__(self, original_layer, rank=16, alpha=32, num_experts=4):
        """
        Args:
            original_layer: Original nn.Linear layer
            rank: LoRA rank per expert
            alpha: LoRA scaling factor
            num_experts: Number of expert LoRA pairs
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.num_experts = num_experts
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Expert LoRA pairs
        self.expert_A = nn.Parameter(torch.randn(num_experts, rank, in_features) * 0.01)
        self.expert_B = nn.Parameter(torch.zeros(num_experts, out_features, rank))
        
        # Gate network for expert selection
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
        
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward: h = Wx + Σ_i w_i * (B_i @ A_i @ x)
        """
        result = self.original_layer(x)
        
        # Compute expert weights from input
        x_pooled = x.mean(dim=1) if x.dim() == 3 else x  # (B, in_features)
        expert_weights = F.softmax(self.gate_net(x_pooled), dim=-1)  # (B, num_experts)
        
        # Compute weighted combination of expert outputs
        lora_out = 0
        for i in range(self.num_experts):
            # Expert i: A_i @ x -> B_i
            expert_out = (x @ self.expert_A[i].T) @ self.expert_B[i].T
            # Weight by expert_weights[:, i]
            if expert_out.dim() == 3:  # (B, seq_len, out_features)
                lora_out = lora_out + expert_weights[:, i].unsqueeze(1).unsqueeze(2) * expert_out
            else:  # (B, out_features)
                lora_out = lora_out + expert_weights[:, i].unsqueeze(1) * expert_out
        
        result = result + self.scaling * lora_out
        return result


class TaskAdaptiveLoRALayer(nn.Module):
    """
    Task-Adaptive LoRA Layer (Our Proposed Method)
    
    Key innovation: Task differentiation happens BEFORE the low-rank bottleneck
        A_t = A_shared + Σ_i w_i * A_expert_i
        h = Wx + B @ A_t @ x
    
    This preserves task-specific features in the original high-dimensional space
    before compressing to low-rank representation, avoiding information loss.
    """
    
    def __init__(self, original_layer, rank=16, alpha=32, num_experts=4, task_emb_dim=32):
        """
        Args:
            original_layer: Original nn.Linear layer
            rank: LoRA rank
            alpha: LoRA scaling factor
            num_experts: Number of expert A matrices
            task_emb_dim: Dimension of task embedding for gate
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.num_experts = num_experts
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Shared down-projection (captures common knowledge)
        self.A_shared = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Expert down-projections (capture task-specific knowledge)
        self.A_experts = nn.Parameter(torch.randn(num_experts, rank, in_features) * 0.01)
        
        # Shared up-projection
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Gate network: input -> task representation -> expert weights
        self.gate_net = nn.Sequential(
            nn.Linear(in_features, task_emb_dim),
            nn.ReLU(),
            nn.Linear(task_emb_dim, num_experts)
        )
        
        self.scaling = alpha / rank
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    def forward(self, x):
        """
        Forward: h = Wx + B @ A_t @ x
        where A_t = A_shared + Σ_i w_i * A_expert_i
        """
        result = self.original_layer(x)
        
        # Compute expert weights from input features
        x_pooled = x.mean(dim=1) if x.dim() == 3 else x  # (B, in_features)
        expert_weights = F.softmax(self.gate_net(x_pooled), dim=-1)  # (B, num_experts)
        
        # Compute task-adaptive A matrix
        # A_shared: (rank, in_features)
        # A_experts: (num_experts, rank, in_features)
        # expert_weights: (B, num_experts)
        
        # Start with shared A for all samples in batch
        A_t = self.A_shared.unsqueeze(0).expand(x.size(0), -1, -1)  # (B, rank, in_features)
        
        # Add weighted expert contributions
        # expert_weights: (B, num_experts) -> (B, num_experts, 1, 1)
        weighted_experts = (expert_weights.unsqueeze(-1).unsqueeze(-1) * 
                           self.A_experts.unsqueeze(0)).sum(dim=1)  # (B, rank, in_features)
        A_t = A_t + weighted_experts
        
        # Apply task-adaptive down-projection
        if x.dim() == 3:  # (B, seq_len, in_features)
            # Use batched matrix multiplication
            lora_out = torch.bmm(x, A_t.transpose(-2, -1))  # (B, seq_len, rank)
            # Apply shared up-projection
            lora_out = lora_out @ self.lora_B.T  # (B, seq_len, out_features)
        else:  # (B, in_features)
            lora_out = torch.bmm(x.unsqueeze(1), A_t.transpose(-2, -1)).squeeze(1)  # (B, rank)
            lora_out = lora_out @ self.lora_B.T  # (B, out_features)
        
        result = result + self.scaling * lora_out
        return result


class VLMModule(nn.Module):
    """
    Vision-Language Module based on ALBEF
    
    Architecture:
        1. Visual Encoder: ViT-Base (from ALBEF)
           - Input: (B, 3, 224, 224)
           - Output: (B, num_patches+1, 768) where num_patches = 196
        
        2. Text Encoder: BERT-style (from ALBEF)
           - Input: (B, seq_len, 768) - direct embeddings, skip tokenizer
           - Mode: 'fusion' - enables cross-attention with visual features
           - Output: (B, seq_len, 768) - fused embeddings
        
        3. LoRA Adaptation: Applied to Q, K, V projections in both encoders
    
    Ablation modes:
        - None: Full model (default)
        - 'no_seq': Only use Visual path (skip text encoder)
        - 'no_visual': Only use Sequence path (skip visual encoder)
    """
    
    def __init__(self, config, ablation_mode=None):
        super().__init__()
        
        # Store config and ablation mode for later use
        self.config = config
        self.ablation_mode = ablation_mode
        
        # Extract VLM configuration
        # Handle both full config and model config
        if 'model' in config:
            vlm_config = config.get('model', {}).get('vlm', {})
        else:
            vlm_config = config.get('vlm', {})
        self.image_res = vlm_config.get('image_res', 224)
        self.vision_width = vlm_config.get('vision_width', 768)
        self.embed_dim = vlm_config.get('embed_dim', 256)
        
        # LoRA configuration
        self.use_lora = vlm_config.get('use_lora', True)
        self.lora_rank = vlm_config.get('lora_rank', 16)
        self.lora_alpha = vlm_config.get('lora_alpha', 32)
        
        # BERT config path
        bert_config_path = vlm_config.get('bert_config', './config/config_bert.json')
        self.pretrained_path = vlm_config.get('pretrained_path', None)
        
        # === Visual Encoder (ViT) ===
        self.visual_encoder = VisionTransformer(
            img_size=self.image_res,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        
        # === Text Encoder (BERT) ===
        bert_config = BertConfig.from_json_file(bert_config_path)
        self.text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
        
        # === Projection layers ===
        # These project features to a common embedding space
        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(bert_config.hidden_size, self.embed_dim)
        
        # === Load pretrained weights ===
        self._pretrained_loaded = False  # Track if pretrained weights were loaded
        if self.pretrained_path and os.path.exists(self.pretrained_path):
            self._load_pretrained_weights()
            self._pretrained_loaded = True
        
        # === Apply LoRA ===
        if self.use_lora:
            self._add_lora_adapters()
        else:
            # Only freeze backbone if pretrained weights were loaded
            # (no point freezing random weights when training from scratch)
            if self._pretrained_loaded:
                self._freeze_backbone()
            else:
                print("Training VLM backbone from scratch (no pretrained weights, no LoRA).")
    
    def _load_pretrained_weights(self):
        """Load pretrained ALBEF weights"""
        print(f"Loading pretrained ALBEF weights from: {self.pretrained_path}")
        
        try:
            checkpoint = torch.load(self.pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            print("\n" + "="*60)
            print("Pretrained Weights Loading Summary")
            print("="*60)
            
            # Load visual encoder
            visual_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('visual_encoder.'):
                    new_key = key.replace('visual_encoder.', '')
                    visual_state_dict[new_key] = value
            
            # Interpolate position embeddings if needed
            if 'pos_embed' in visual_state_dict:
                pos_embed_reshaped = interpolate_pos_embed(
                    visual_state_dict['pos_embed'], 
                    self.visual_encoder
                )
                visual_state_dict['pos_embed'] = pos_embed_reshaped
            
            visual_total = sum(p.numel() for p in self.visual_encoder.parameters())
            msg = self.visual_encoder.load_state_dict(visual_state_dict, strict=False)
            # Calculate actually loaded params (exclude unexpected keys)
            visual_loaded = sum(visual_state_dict[k].numel() for k in visual_state_dict.keys() 
                              if k not in msg.unexpected_keys)
            print(f"\n[Visual Encoder]")
            print(f"  Total params: {visual_total:,} | Loaded: {visual_loaded:,} ({100*visual_loaded/visual_total:.1f}%)")
            print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"  Missing: {', '.join(msg.missing_keys[:3])}{'...' if len(msg.missing_keys) > 3 else ''}")
            
            # Load text encoder (BERT)
            # Note: ALBEF pretrain uses BertForMaskedLM with 'bert.' prefix
            # Some checkpoints may have both 'bert.X' and 'X' (duplicates), some only have 'bert.X'
            text_state_dict = {}
            text_state_dict_with_bert_prefix = {}
            
            for key, value in state_dict.items():
                if key.startswith('text_encoder.'):
                    new_key = key.replace('text_encoder.', '')
                    if new_key.startswith('bert.'):
                        # Store bert-prefixed keys separately
                        clean_key = new_key.replace('bert.', '')
                        text_state_dict_with_bert_prefix[clean_key] = value
                    else:
                        # Direct keys (no bert prefix)
                        text_state_dict[new_key] = value
            
            # Merge: prefer keys without bert prefix (they override bert-prefixed ones)
            for key, value in text_state_dict_with_bert_prefix.items():
                if key not in text_state_dict:
                    text_state_dict[key] = value
            
            text_total = sum(p.numel() for p in self.text_encoder.parameters())
            msg = self.text_encoder.load_state_dict(text_state_dict, strict=False)
            # Calculate actually loaded params (exclude unexpected keys from ALBEF's MLM head, cls head, etc.)
            text_loaded = sum(text_state_dict[k].numel() for k in text_state_dict.keys() 
                            if k not in msg.unexpected_keys)
            print(f"\n[Text Encoder]")
            print(f"  Total params: {text_total:,} | Loaded: {text_loaded:,} ({100*text_loaded/text_total:.1f}%)")
            print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected: {', '.join(msg.unexpected_keys[:3])}{'...' if len(msg.unexpected_keys) > 3 else ''}")
            
            # Load projection layers (for potential future use in contrastive learning)
            proj_loaded = 0
            proj_status = []
            if 'vision_proj.weight' in state_dict:
                self.vision_proj.load_state_dict({
                    'weight': state_dict['vision_proj.weight'],
                    'bias': state_dict['vision_proj.bias']
                })
                proj_loaded += state_dict['vision_proj.weight'].numel() + state_dict['vision_proj.bias'].numel()
                proj_status.append('vision_proj')
            if 'text_proj.weight' in state_dict:
                self.text_proj.load_state_dict({
                    'weight': state_dict['text_proj.weight'],
                    'bias': state_dict['text_proj.bias']
                })
                proj_loaded += state_dict['text_proj.weight'].numel() + state_dict['text_proj.bias'].numel()
                proj_status.append('text_proj')
            
            proj_total = sum(p.numel() for p in self.vision_proj.parameters()) + sum(p.numel() for p in self.text_proj.parameters())
            print(f"\n[Projection Layers] (loaded but not used in current forward)")
            print(f"  Total params: {proj_total:,} | Loaded: {proj_loaded:,} ({100*proj_loaded/proj_total:.1f}%)")
            print(f"  Loaded layers: {', '.join(proj_status) if proj_status else 'None'}")
            
            # Overall summary
            total_params = visual_total + text_total + proj_total
            loaded_params = visual_loaded + text_loaded + proj_loaded
            print(f"\n" + "-"*60)
            print(f"[Overall] Total: {total_params:,} | Loaded: {loaded_params:,} ({100*loaded_params/total_params:.1f}%)")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights.")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters (when not using LoRA)"""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        print("Backbone frozen (all parameters).")
    
    def _add_lora_adapters(self):
        """
        Add LoRA adapters to key attention layers
        Apply to Q, K, V projections in both visual and text encoders
        Supports multiple LoRA variants: standard, MTL, MoE, task-adaptive
        """
        # Get LoRA configuration
        if 'model' in self.config:
            vlm_config = self.config.get('model', {}).get('vlm', {})
        else:
            vlm_config = self.config.get('vlm', {})
        
        lora_type = vlm_config.get('lora_type', 'standard')
        num_tasks = vlm_config.get('num_tasks', 14)
        num_experts = vlm_config.get('num_experts', 4)
        task_emb_dim = vlm_config.get('task_emb_dim', 32)
        
        print(f"Adding {lora_type.upper()} LoRA adapters (rank={self.lora_rank}, alpha={self.lora_alpha})...")
        if lora_type in ['mtl', 'moe', 'task_adaptive']:
            print(f"  Multi-task config: num_tasks={num_tasks}, num_experts={num_experts}")
        
        # Select LoRA class based on type
        if lora_type == 'mtl':
            lora_class = MTLLoRALayer
            lora_kwargs = {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha,
                'num_tasks': num_tasks,
                'num_experts': num_experts
            }
        elif lora_type == 'moe':
            lora_class = MOELoRALayer
            lora_kwargs = {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha,
                'num_experts': num_experts
            }
        elif lora_type == 'task_adaptive':
            lora_class = TaskAdaptiveLoRALayer
            lora_kwargs = {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha,
                'num_experts': num_experts,
                'task_emb_dim': task_emb_dim
            }
        else:  # standard
            lora_class = LoRALayer
            lora_kwargs = {
                'rank': self.lora_rank,
                'alpha': self.lora_alpha
            }
        
        # Freeze all parameters first
        for param in self.parameters():
            param.requires_grad = False
        
        lora_count = 0
        
        # === Add LoRA to Visual Encoder (ViT) ===
        for block_idx, block in enumerate(self.visual_encoder.blocks):
            attn = block.attn
            
            # Replace Q, K, V projections with LoRA versions
            # Note: ViT uses a single QKV projection
            if hasattr(attn, 'qkv'):
                # Split QKV into separate components for LoRA
                # Keep the original qkv for initialization but add separate LoRA for each
                original_qkv = attn.qkv
                dim = original_qkv.in_features
                
                # Create separate Q, K, V with LoRA
                attn.query_lora = lora_class(
                    nn.Linear(dim, dim, bias=True),
                    **lora_kwargs
                )
                attn.key_lora = lora_class(
                    nn.Linear(dim, dim, bias=True),
                    **lora_kwargs
                )
                attn.value_lora = lora_class(
                    nn.Linear(dim, dim, bias=True),
                    **lora_kwargs
                )
                
                # Initialize with split QKV weights
                with torch.no_grad():
                    qkv_weight = original_qkv.weight
                    qkv_bias = original_qkv.bias if original_qkv.bias is not None else None
                    
                    attn.query_lora.original_layer.weight.copy_(qkv_weight[:dim, :])
                    attn.key_lora.original_layer.weight.copy_(qkv_weight[dim:2*dim, :])
                    attn.value_lora.original_layer.weight.copy_(qkv_weight[2*dim:, :])
                    
                    if qkv_bias is not None:
                        attn.query_lora.original_layer.bias.copy_(qkv_bias[:dim])
                        attn.key_lora.original_layer.bias.copy_(qkv_bias[dim:2*dim])
                        attn.value_lora.original_layer.bias.copy_(qkv_bias[2*dim:])
                
                attn.use_lora = True
                lora_count += 3
        
        # === Add LoRA to Text Encoder (BERT) ===
        for layer_idx, layer in enumerate(self.text_encoder.encoder.layer):
            # Self-attention
            if hasattr(layer.attention.self, 'query'):
                layer.attention.self.query = lora_class(
                    layer.attention.self.query,
                    **lora_kwargs
                )
                layer.attention.self.key = lora_class(
                    layer.attention.self.key,
                    **lora_kwargs
                )
                layer.attention.self.value = lora_class(
                    layer.attention.self.value,
                    **lora_kwargs
                )
                lora_count += 3
            
            # Cross-attention (if exists, fusion layers)
            if hasattr(layer, 'crossattention'):
                layer.crossattention.self.query = lora_class(
                    layer.crossattention.self.query,
                    **lora_kwargs
                )
                layer.crossattention.self.key = lora_class(
                    layer.crossattention.self.key,
                    **lora_kwargs
                )
                layer.crossattention.self.value = lora_class(
                    layer.crossattention.self.value,
                    **lora_kwargs
                )
                lora_count += 3
        
        print(f"Added {lora_count} LoRA adapters successfully!")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} " 
              f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(self, image, sequence, seq_mask):
        """
        Forward pass through VLM
        
        Args:
            image: (B, 3, H, W) - Pseudo-RGB image from format transformer
            sequence: (B, seq_len, 768) - Sequence embeddings (with CLS and PROMPT)
            seq_mask: (B, seq_len) - Attention mask (1 for valid, 0 for padding)
        
        Returns:
            dict with:
                - 'fused_embeds': (B, seq_len, 768) - Multimodal fused embeddings
                - 'visual_embeds': (B, num_patches+1, 768) - Visual features
                - 'text_embeds': (B, seq_len, 768) - Text features (before fusion)
        """
        B = image.size(0)
        device = image.device
        seq_len = sequence.size(1)
        
        # === Ablation: no_visual (only sequence/text path) ===
        if self.ablation_mode == 'no_visual':
            # Skip visual encoder, use text-only mode
            text_output_only = self.text_encoder(
                encoder_embeds=sequence,
                attention_mask=seq_mask,
                return_dict=True,
                mode='text'  # Text-only mode (no cross-attention)
            )
            text_embeds = text_output_only.last_hidden_state  # (B, seq_len, 768)
            
            # Create dummy visual embeds
            visual_embeds = torch.zeros(B, 197, self.vision_width, device=device)
            
            return {
                'fused_embeds': text_embeds,  # Use text-only as fused
                'visual_embeds': visual_embeds,
                'text_embeds': text_embeds
            }
        
        # === Visual Encoding ===
        visual_embeds = self._encode_vision(image)  # (B, 197, 768) = (B, 196+1, 768)
        
        # === Ablation: no_seq (only visual path) ===
        if self.ablation_mode == 'no_seq':
            # Skip text encoder, use visual features directly
            # Pool visual features to match sequence length for prediction head
            # Use CLS token expanded or simple projection
            visual_cls = visual_embeds[:, 0:1, :]  # (B, 1, 768)
            visual_mean = visual_embeds.mean(dim=1, keepdim=True)  # (B, 1, 768)
            
            # Create pseudo fused_embeds by repeating visual representation
            fused_embeds = visual_mean.expand(B, seq_len, -1)  # (B, seq_len, 768)
            
            # Create dummy text embeds
            text_embeds = torch.zeros(B, seq_len, self.vision_width, device=device)
            
            return {
                'fused_embeds': fused_embeds,
                'visual_embeds': visual_embeds,
                'text_embeds': text_embeds
            }
        
        # === Standard: Full dual-path encoding ===
        # Create visual attention mask (all ones - no padding in images)
        visual_mask = torch.ones(
            visual_embeds.size()[:-1], 
            dtype=torch.long, 
            device=device
        )  # (B, 197)
        
        # === Text Encoding with Visual Cross-Attention (Fusion) ===
        # Use BERT's fusion mode: first 6 layers are text-only, last 6 layers do cross-attention
        text_output = self.text_encoder(
            encoder_embeds=sequence,  # Direct embedding input, skip tokenizer
            attention_mask=seq_mask,
            encoder_hidden_states=visual_embeds,  # Visual features for cross-attention
            encoder_attention_mask=visual_mask,
            return_dict=True,
            mode='fusion'  # Use fusion mode (last 6 layers do cross-attention)
        )
        
        fused_embeds = text_output.last_hidden_state  # (B, seq_len, 768)
        
        # === Also get text-only features (for potential auxiliary tasks) ===
        text_output_only = self.text_encoder(
            encoder_embeds=sequence,
            attention_mask=seq_mask,
            return_dict=True,
            mode='text'  # Text-only mode (first 6 layers only)
        )
        text_embeds = text_output_only.last_hidden_state  # (B, seq_len, 768)
        
        return {
            'fused_embeds': fused_embeds,
            'visual_embeds': visual_embeds,
            'text_embeds': text_embeds
        }
    
    def _encode_vision(self, image):
        """
        Encode image using ViT with LoRA adaptation
        
        Args:
            image: (B, 3, H, W)
        
        Returns:
            visual_embeds: (B, num_patches+1, 768)
        """
        if not self.use_lora:
            # Standard ViT forward
            return self.visual_encoder(image)
        
        # ViT forward with LoRA
        B = image.shape[0]
        x = self.visual_encoder.patch_embed(image)
        
        # Add CLS token
        cls_tokens = self.visual_encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.visual_encoder.pos_embed[:, :x.size(1), :]
        x = self.visual_encoder.pos_drop(x)
        
        # Process through blocks with LoRA
        for block in self.visual_encoder.blocks:
            x = self._vit_block_forward_with_lora(block, x)
        
        x = self.visual_encoder.norm(x)
        
        return x
    
    def _vit_block_forward_with_lora(self, block, x):
        """
        ViT block forward with LoRA adaptation
        """
        # Attention with LoRA
        if hasattr(block.attn, 'use_lora') and block.attn.use_lora:
            x = x + block.drop_path(self._vit_attn_with_lora(block.attn, block.norm1(x)))
        else:
            x = x + block.drop_path(block.attn(block.norm1(x)))
        
        # MLP
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        
        return x
    
    def _vit_attn_with_lora(self, attn, x):
        """
        ViT attention forward with LoRA on Q, K, V
        """
        B, N, C = x.shape
        
        # Apply LoRA to Q, K, V
        q = attn.query_lora(x).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        k = attn.key_lora(x).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        v = attn.value_lora(x).reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        
        # Attention computation
        attn_scores = (q @ k.transpose(-2, -1)) * attn.scale
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = attn.attn_drop(attn_probs)
        
        # Output
        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = attn.proj(x)
        x = attn.proj_drop(x)
        
        return x
    
    def freeze_backbone(self):
        """
        Freeze backbone parameters (for LoRA-only training)
        This is already done in _add_lora_adapters, but provided for explicit control
        """
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters (for full fine-tuning)"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get statistics on trainable parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable,
            'total': total,
            'frozen': total - trainable,
            'trainable_ratio': trainable / total if total > 0 else 0
        }


# === Standalone testing function ===
def test_vlm_module():
    """Test VLM module with dummy inputs"""
    print("=" * 60)
    print("Testing VLM Module")
    print("=" * 60)
    
    # Mock config
    config = {
        'vlm': {
            'image_res': 224,
            'vision_width': 768,
            'embed_dim': 256,
            'bert_config': '../../ALBEF/configs/config_bert.json',
            'pretrained_path': None,  # Set to None for testing
            'use_lora': True,
            'lora_rank': 16,
            'lora_alpha': 32
        }
    }
    
    # Create module
    vlm = VLMModule(config)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 302  # 1 CLS + 1 PROMPT + 300 latents
    
    image = torch.randn(batch_size, 3, 224, 224)
    sequence = torch.randn(batch_size, seq_len, 768)
    seq_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    print(f"\nInput shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Sequence: {sequence.shape}")
    print(f"  Sequence mask: {seq_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = vlm(image, sequence, seq_mask)
    
    print(f"\nOutput shapes:")
    print(f"  Fused embeddings: {output['fused_embeds'].shape}")
    print(f"  Visual embeddings: {output['visual_embeds'].shape}")
    print(f"  Text embeddings: {output['text_embeds'].shape}")
    
    # Check parameter statistics
    param_stats = vlm.get_trainable_params()
    print(f"\nParameter statistics:")
    print(f"  Total: {param_stats['total']:,}")
    print(f"  Trainable: {param_stats['trainable']:,}")
    print(f"  Frozen: {param_stats['frozen']:,}")
    print(f"  Trainable ratio: {param_stats['trainable_ratio']:.2%}")
    
    print("\n" + "=" * 60)
    print("VLM Module test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_vlm_module()

