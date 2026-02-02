"""
PromptForge: Hierarchical Pattern Memory Module
Generates dynamic prompts from temporal, spatial, and cross-modal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalPatternMemory(nn.Module):
    """
    Hierarchical Pattern Memory Pool with coarse and fine-grained prototypes
    Example: Temporal patterns -> Periodicity -> Daily/Weekly cycles
    """
    
    def __init__(self, d_model, num_coarse=5, num_fine_per_coarse=2, d_prompt=128):
        super().__init__()
        self.d_model = d_model
        self.num_coarse = num_coarse
        self.num_fine = num_fine_per_coarse
        self.d_prompt = d_prompt
        
        # Coarse-grained prototypes (main pattern categories)
        # Use orthogonal initialization to maximize initial distinctiveness
        self.coarse_prototypes = nn.Parameter(torch.empty(num_coarse, d_model))
        nn.init.orthogonal_(self.coarse_prototypes)
        
        # Fine-grained prototypes (sub-patterns for each category)
        # Use orthogonal initialization for each coarse category independently
        self.fine_prototypes = nn.Parameter(
            torch.empty(num_coarse, num_fine_per_coarse, d_model)
        )
        for k in range(num_coarse):
            nn.init.orthogonal_(self.fine_prototypes[k])
        
        # Query projection
        self.query_proj = nn.Linear(d_model, d_model)
        
        # Prompt projection
        self.prompt_proj = nn.Linear(d_model, d_prompt)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, cls_token):
        """
        Args:
            cls_token: (B, D) CLS token from attention module
            
        Returns:
            prompt: (B, d_prompt) pattern-aware prompt
            weights: (coarse_weights, fine_weights) for visualization
        """
        # Project query
        query = self.query_proj(cls_token)  # (B, D)
        query = self.norm(query)
        
        # Normalize query for cosine similarity
        query_norm = F.normalize(query, p=2, dim=-1)  # (B, D)
        
        # Step 1: Match coarse-grained prototypes using cosine similarity
        coarse_prototypes_norm = F.normalize(self.coarse_prototypes, p=2, dim=-1)  # (num_coarse, D)
        coarse_sim = torch.matmul(query_norm, coarse_prototypes_norm.T)  # (B, num_coarse)
        coarse_weights = F.softmax(coarse_sim / 0.5, dim=-1)  # Temperature scaling
        
        # Step 2: For each batch, find top-k coarse categories (cumulative weight >= 90%)
        # and match fine-grained prototypes within them
        B = query.size(0)
        fine_weights_all = torch.zeros(B, self.num_coarse, self.num_fine, device=query.device)
        
        # Match fine-grained prototypes only within top-k coarse categories
        for b in range(B):
            # Sort coarse weights and find top-k that cumsum to >= 0.9
            sorted_weights, sorted_indices = torch.sort(coarse_weights[b], descending=True)
            cumsum_weights = torch.cumsum(sorted_weights, dim=0)
            top_k_mask = cumsum_weights <= 0.9
            if top_k_mask.sum() == 0:  # At least select top-1
                top_k_mask[0] = True
            else:
                # Include one more to exceed 90%
                num_selected = top_k_mask.sum().item()
                if num_selected < len(top_k_mask):
                    top_k_mask[num_selected] = True
            
            selected_coarse_indices = sorted_indices[top_k_mask]
            
            # Match fine prototypes for each selected coarse category
            for k in selected_coarse_indices:
                fine_prototypes_k = self.fine_prototypes[k]  # (num_fine, D)
                fine_prototypes_k_norm = F.normalize(fine_prototypes_k, p=2, dim=-1)  # (num_fine, D)
                
                # Cosine similarity with fine prototypes in selected coarse category
                fine_sim_k = torch.matmul(query_norm[b:b+1], fine_prototypes_k_norm.T)  # (1, num_fine)
                fine_weights_k = F.softmax(fine_sim_k / 0.5, dim=-1)  # (1, num_fine)
                fine_weights_all[b, k, :] = fine_weights_k[0]
        
        # Step 3: Combine weights hierarchically
        combined_weights = coarse_weights.unsqueeze(-1) * fine_weights_all  # (B, num_coarse, num_fine)
        
        # Step 4: Generate prompt by weighted combination
        # Project fine prototypes to prompt space
        fine_prototypes_flat = self.fine_prototypes.reshape(
            self.num_coarse * self.num_fine, self.d_model
        )  # (num_coarse * num_fine, D)
        fine_prompts = self.prompt_proj(fine_prototypes_flat)  # (num_coarse * num_fine, d_prompt)
        fine_prompts = fine_prompts.reshape(
            self.num_coarse, self.num_fine, self.d_prompt
        )  # (num_coarse, num_fine, d_prompt)
        
        # Weighted sum
        prompt = torch.einsum('bkm,kmd->bd', combined_weights, fine_prompts)  # (B, d_prompt)
        
        return prompt, (coarse_weights, fine_weights_all)


class PromptForge(nn.Module):
    """
    Multi-modal Pattern Memory Module
    Generates prompts from temporal, spatial, and cross-modal patterns
    """
    
    def __init__(self, d_model, num_coarse=5, num_fine_per_coarse=2, d_prompt=128):
        super().__init__()
        self.d_prompt = d_prompt
        
        # Three independent memory pools
        self.temporal_memory = HierarchicalPatternMemory(
            d_model, num_coarse, num_fine_per_coarse, d_prompt
        )
        
        self.spatial_memory = HierarchicalPatternMemory(
            d_model, num_coarse, num_fine_per_coarse, d_prompt
        )
        
        self.cross_memory = HierarchicalPatternMemory(
            d_model, num_coarse, num_fine_per_coarse, d_prompt
        )
        
        # Prompt fusion layer (optional enhancement)
        self.prompt_fusion = nn.Sequential(
            nn.Linear(d_prompt * 3, d_prompt * 2),
            nn.LayerNorm(d_prompt * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_prompt * 2, d_prompt * 3)
        )
        
        self.norm = nn.LayerNorm(d_prompt * 3)
    
    def forward(self, temporal_cls, spatial_cls, cross_cls):
        """
        Args:
            temporal_cls: (B, D) CLS token from temporal attention
            spatial_cls: (B, D) CLS token from spatial attention
            cross_cls: (B, D) CLS token from cross attention
            
        Returns:
            B_prompt: (B, d_prompt * 3) concatenated prompts
            pattern_info: Dict with pattern weights for visualization
        """
        # Generate three prompts from cls_tokens
        P_temp, W_temp = self.temporal_memory(temporal_cls)
        P_spat, W_spat = self.spatial_memory(spatial_cls)
        P_cross, W_cross = self.cross_memory(cross_cls)
        
        # Concatenate prompts
        B_prompt = torch.cat([P_temp, P_spat, P_cross], dim=-1)  # (B, d_prompt * 3)
        
        # Optional: Further fusion
        B_prompt_enhanced = self.prompt_fusion(B_prompt)
        B_prompt = self.norm(B_prompt + B_prompt_enhanced)
        
        # Pattern information for analysis
        pattern_info = {
            'temporal_weights': W_temp,
            'spatial_weights': W_spat,
            'cross_weights': W_cross
        }
        
        return B_prompt, pattern_info

