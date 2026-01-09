"""
GNN Encoder for MARL (Works for both Tier 1 and Tier 2)

This module provides a Graph Neural Network (GNN) encoder and custom PPO policy
that uses the GNN to preprocess observations before feeding them to the standard
actor-critic network.

Architecture:
    Tier 1: 6D → GNN (18D output) → [128, 64] MLP
    Tier 2 (Hierarchical): 
        - 6D base features → GNN (12D) 
        - 8D forecast features → GNN (12D)
        - Cross-attention fusion → 24D → [256, 128] MLP
        - Allows separate processing of base vs forecast features with learned interactions

Compatible with Stable Baselines3 PPO.

REFACTORED: Merged custom_policy.py and gnn_encoder.py into this unified module.
GNN encoder is independent of forecast integration - works on base observations (Tier 1)
or forecast-enhanced observations (Tier 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Optional
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ============================================================================
# GNN Encoder Components (merged from gnn_encoder.py)
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer (GAT)
    
    IMPROVED: Supports multi-head attention to capture different types of relationships:
    - Head 1: Price-forecast interactions
    - Head 2: Position-forecast interactions  
    - Head 3: Risk-forecast interactions
    
    Computes attention-weighted aggregation of neighbor features:
    h_i' = ||_k=1^K σ(Σ_j α_ij^k W^k h_j)
    
    where α_ij^k = softmax(LeakyReLU(a^k^T [W^k h_i || W^k h_j]))
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2, num_heads: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.num_heads = num_heads
        
        # IMPROVED: Multi-head attention - each head learns different relationships
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.head_dim = out_features // num_heads
        
        # Learnable weight matrices for each head
        self.W = nn.Parameter(torch.zeros(size=(num_heads, in_features, self.head_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanisms for each head
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes) - 1 if connected, 0 otherwise
            
        Returns:
            h': Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = h.shape
        
        # IMPROVED: Multi-head attention computation
        head_outputs = []
        for head in range(self.num_heads):
            # Linear transformation for this head
            Wh = torch.matmul(h, self.W[head])  # (batch, num_nodes, head_dim)
            
            # Prepare for attention computation
            Wh1 = Wh.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, -1)
            Wh2 = Wh.repeat(1, num_nodes, 1)
            
            # Concatenate and compute attention scores
            e = torch.cat([Wh1, Wh2], dim=2)  # (batch, num_nodes*num_nodes, 2*head_dim)
            e = self.leakyrelu(torch.matmul(e, self.a[head]).squeeze(2))  # (batch, num_nodes*num_nodes)
            e = e.view(batch_size, num_nodes, num_nodes)  # (batch, num_nodes, num_nodes)
            
            # Mask attention scores with adjacency matrix
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            
            # Softmax to get attention weights
            attention = F.softmax(attention, dim=2)  # (batch, num_nodes, num_nodes)
            attention = F.dropout(attention, self.dropout, training=self.training)
            
            # Apply attention to features
            h_head = torch.matmul(attention, Wh)  # (batch, num_nodes, head_dim)
            head_outputs.append(h_head)
        
        # Concatenate all heads
        h_prime = torch.cat(head_outputs, dim=2)  # (batch, num_nodes, out_features)
        
        return h_prime


class GNNObservationEncoder(nn.Module):
    """
    GNN Encoder for MARL observations
    
    Converts raw observations into graph, applies GAT layers, and outputs encoded features.
    
    Args:
        obs_dim: Input observation dimension (6 for Tier 1, 14 for Tier 2)
        hidden_dim: Hidden dimension for GAT layers (default: 32)
        output_dim: Output encoded feature dimension (default: 16)
        num_layers: Number of GAT layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        graph_type: 'full' (fully-connected) or 'learned' (learnable adjacency) or 'hierarchical' (Tier 2 structured)
        num_heads: Number of attention heads (default: 3)
        use_attention_pooling: Use attention pooling instead of mean (default: True)
        base_feature_dim: For Tier 2, number of base features (6) before forecast features (8)
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        graph_type: str = 'full',
        num_heads: int = 3,
        use_attention_pooling: bool = True,
        base_feature_dim: Optional[int] = None
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.graph_type = graph_type
        self.num_heads = num_heads
        self.use_attention_pooling = use_attention_pooling
        self.base_feature_dim = base_feature_dim if base_feature_dim is not None else 6  # Default for Tier 2
        
        # Determine if this is hierarchical structure
        # Hierarchical GNN works for any observation space with a base/forecast split
        # - Tier 2 Investor: 14D (6 base + 8 forecast)
        # - Tier 3 Investor: 18D (6 base + 8 forecast + 4 bridge) - hierarchical processes first 14D
        # - Battery: 8D (4 base + 4 forecast)
        self.is_hierarchical = (graph_type == 'hierarchical') and (base_feature_dim is not None)
        
        # Track if we have bridge vectors appended (Tier 3)
        self.has_bridge_vectors = False
        self.forecast_feature_dim = None  # Initialize to None (will be set below)
        if self.is_hierarchical:
            # Calculate forecast feature dimension
            # For Tier 3 (18D): first 14D are base+forecast, last 4D are bridge vectors
            # For Tier 2 (14D): all 14D are base+forecast
            if obs_dim > self.base_feature_dim * 2:
                # Likely has bridge vectors (Tier 3): obs_dim = 18, base_feature_dim = 6
                # Forecast features are 8D (between base and bridge)
                self.forecast_feature_dim = obs_dim - self.base_feature_dim - 4  # 18 - 6 - 4 = 8
                self.has_bridge_vectors = True
            else:
                # No bridge vectors (Tier 2): obs_dim = 14, base_feature_dim = 6
                self.forecast_feature_dim = obs_dim - self.base_feature_dim  # 14 - 6 = 8
                self.has_bridge_vectors = False
            
            # IMPROVED: Separate GNN encoders with INCREASED capacity for forecast features
            # Forecast features are critical for Tier 2/3 performance, so give them MORE capacity
            # Base encoder: Standard capacity (base features are well-understood)
            # Forecast encoder: INCREASED capacity (forecast features need more processing to learn alignment)
            
            # ROBUST FIX: Ensure base and forecast output dims sum to output_dim (features_dim)
            # We want: base_output_dim + forecast_output_dim = output_dim
            # But each must be divisible by their respective num_heads
            
            forecast_num_heads = num_heads + 1
            
            # Try all possible splits of output_dim that satisfy divisibility constraints
            # We need: base_output_dim % num_heads == 0 AND forecast_output_dim % forecast_num_heads == 0
            # AND base_output_dim + forecast_output_dim == output_dim
            
            base_output_dim = None
            forecast_output_dim = None
            
            # Try different splits, starting from roughly equal split
            # Search from both ends to find the most balanced split
            candidates = list(range(num_heads, output_dim - forecast_num_heads + 1, num_heads))
            if not candidates:
                raise RuntimeError(f"No valid base candidates for output_dim={output_dim}, num_heads={num_heads}, forecast_num_heads={forecast_num_heads}")
            
            # Sort by distance from output_dim // 2 to prefer balanced splits
            candidates.sort(key=lambda x: abs(x - output_dim // 2))
            
            base_output_dim = None
            forecast_output_dim = None
            
            # Try different splits, starting from roughly equal split
            for base_candidate in candidates:
                forecast_candidate = output_dim - base_candidate
                if forecast_candidate >= forecast_num_heads and forecast_candidate % forecast_num_heads == 0:
                    base_output_dim = base_candidate
                    forecast_output_dim = forecast_candidate
                    break
            
            # If no valid split found, try the reverse (prioritize forecast)
            if base_output_dim is None:
                for forecast_candidate in range(forecast_num_heads, output_dim - num_heads + 1, forecast_num_heads):
                    base_candidate = output_dim - forecast_candidate
                    if base_candidate >= num_heads and base_candidate % num_heads == 0:
                        base_output_dim = base_candidate
                        forecast_output_dim = forecast_candidate
                        break
            
            # If still no valid split, we cannot proceed - this is a configuration error
            # The user must choose a features_dim that allows a valid split
            if base_output_dim is None:
                # Try to find the closest valid features_dim that would work
                # Calculate LCM of num_heads and forecast_num_heads to find valid multiples
                import math
                lcm = math.lcm(num_heads, forecast_num_heads)
                # Find multiples of lcm that are close to output_dim
                lower_multiple = (output_dim // lcm) * lcm
                upper_multiple = ((output_dim // lcm) + 1) * lcm
                
                raise RuntimeError(
                    f"Cannot find valid dimension split for output_dim={output_dim}, "
                    f"num_heads={num_heads}, forecast_num_heads={forecast_num_heads}.\n"
                    f"  Required: base_output_dim divisible by {num_heads} AND forecast_output_dim divisible by {forecast_num_heads} "
                    f"AND base_output_dim + forecast_output_dim = {output_dim}.\n"
                    f"  Suggested features_dim values: {lower_multiple} or {upper_multiple} "
                    f"(multiples of LCM({num_heads}, {forecast_num_heads})={lcm})"
                )
            
            # Final validation
            if base_output_dim is None or forecast_output_dim is None:
                raise RuntimeError(
                    f"Cannot find valid dimension split for output_dim={output_dim}, "
                    f"num_heads={num_heads}, forecast_num_heads={forecast_num_heads}. "
                    f"Try adjusting features_dim to be divisible by both {num_heads} and {forecast_num_heads}."
                )
            
            if base_output_dim + forecast_output_dim != output_dim:
                raise RuntimeError(
                    f"Dimension split failed: base_output_dim ({base_output_dim}) + "
                    f"forecast_output_dim ({forecast_output_dim}) != output_dim ({output_dim})"
                )
            
            if base_output_dim % num_heads != 0:
                raise RuntimeError(f"base_output_dim ({base_output_dim}) not divisible by num_heads ({num_heads})")
            
            if forecast_output_dim % forecast_num_heads != 0:
                raise RuntimeError(f"forecast_output_dim ({forecast_output_dim}) not divisible by forecast_num_heads ({forecast_num_heads})")
            
            # ROBUST FIX: Ensure base hidden_dim is divisible by num_heads
            base_hidden_dim = hidden_dim // 2
            if base_hidden_dim % num_heads != 0:
                base_hidden_dim = (base_hidden_dim // num_heads) * num_heads
                if base_hidden_dim == 0:
                    base_hidden_dim = num_heads
            
            self.base_encoder = GNNObservationEncoder(
                obs_dim=self.base_feature_dim,
                hidden_dim=base_hidden_dim,  # Standard capacity for base features (adjusted)
                output_dim=base_output_dim,
                num_layers=num_layers,  # Use full depth for better feature extraction
                dropout=dropout,
                graph_type='full',  # Base features can use full connectivity
                num_heads=num_heads,
                use_attention_pooling=use_attention_pooling,
                base_feature_dim=None  # Explicitly None to prevent recursive hierarchical mode
            )
            
            # ROBUST FIX: Ensure forecast hidden_dim is divisible by forecast_num_heads
            forecast_hidden_dim = int(hidden_dim * 0.75)
            if forecast_hidden_dim % forecast_num_heads != 0:
                forecast_hidden_dim = (forecast_hidden_dim // forecast_num_heads) * forecast_num_heads
                if forecast_hidden_dim == 0:
                    forecast_hidden_dim = forecast_num_heads
            
            self.forecast_encoder = GNNObservationEncoder(
                obs_dim=self.forecast_feature_dim,
                hidden_dim=forecast_hidden_dim,  # INCREASED capacity (adjusted)
                output_dim=forecast_output_dim,
                num_layers=num_layers + 1,  # INCREASED depth by 1 layer for better forecast processing
                dropout=dropout * 0.8,  # Slightly less dropout for forecast features (they're important)
                graph_type='learned',  # Forecast features benefit from learned relationships
                num_heads=forecast_num_heads,  # INCREASED heads for better relationship learning
                use_attention_pooling=use_attention_pooling,
                base_feature_dim=None  # Explicitly None to prevent recursive hierarchical mode
            )
            
            # ROBUST FIX: Verify nested encoders didn't modify their output dimensions
            # The nested encoders should output exactly what we specified
            actual_base_output = getattr(self.base_encoder, 'output_dim', base_output_dim)
            actual_forecast_output = getattr(self.forecast_encoder, 'output_dim', forecast_output_dim)
            
            if actual_base_output != base_output_dim:
                raise RuntimeError(
                    f"Base encoder modified output_dim: expected {base_output_dim}, got {actual_base_output}. "
                    f"This should not happen - nested encoders should use the exact dimensions specified."
                )
            
            if actual_forecast_output != forecast_output_dim:
                raise RuntimeError(
                    f"Forecast encoder modified output_dim: expected {forecast_output_dim}, got {actual_forecast_output}. "
                    f"This should not happen - nested encoders should use the exact dimensions specified."
                )
            
            # ROBUST FIX: Final output_dim should equal input output_dim (features_dim)
            # After concatenation: base_output_dim + forecast_output_dim = output_dim
            # Verify this is correct
            if base_output_dim + forecast_output_dim != output_dim:
                # This should never happen with the logic above, but fail fast if it does
                raise RuntimeError(
                    f"GNN encoder dimension mismatch: base_output_dim ({base_output_dim}) + "
                    f"forecast_output_dim ({forecast_output_dim}) = {base_output_dim + forecast_output_dim} != output_dim ({output_dim}). "
                    f"This indicates a bug in the dimension split logic."
                )
            self.output_dim = output_dim  # Final output is the concatenation, which equals output_dim
            actual_output_dim = output_dim  # Set actual_output_dim for use in fusion layers
            
            # TIER 3: Bridge vector processor (if bridge vectors present)
            if self.has_bridge_vectors:
                bridge_dim = 4  # overlay_bridge_dim = 4
                # Small MLP to project bridge vectors to match output_dim for fusion
                # Bridge vectors are coordination signals, should be fused into encoded features
                self.bridge_processor = nn.Sequential(
                    nn.Linear(bridge_dim, output_dim // 4),  # Project to smaller dim
                    nn.LayerNorm(output_dim // 4),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(output_dim // 4, output_dim)  # Project to output_dim for fusion
                )
            else:
                self.bridge_processor = None
            
            # IMPROVED: Cross-attention fusion layer with MULTI-HEAD for better capacity
            # Use 2 heads for cross-attention to capture different types of base-forecast interactions
            # Head 1: Price-forecast interactions
            # Head 2: Position-forecast interactions
            # This allows the model to learn richer relationships between base and forecast features
            cross_attention_heads = 2  # Increased from 1 to 2 for better capacity
            
            # ROBUST FIX: Use actual encoder output dimensions (may have been adjusted)
            base_enc_output_dim = base_output_dim
            forecast_enc_output_dim = forecast_output_dim
            
            # Cross-attention uses the smaller of the two encoder outputs
            cross_embed_dim = min(base_enc_output_dim, forecast_enc_output_dim)
            if cross_embed_dim % cross_attention_heads != 0:
                # Round down to nearest divisible value
                cross_embed_dim = (cross_embed_dim // cross_attention_heads) * cross_attention_heads
                if cross_embed_dim == 0:
                    cross_embed_dim = cross_attention_heads  # At least cross_attention_heads
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=cross_embed_dim,
                num_heads=cross_attention_heads,
                dropout=dropout,
                batch_first=False
            )
            
            # Projection layers to ensure dimensions match for fusion
            # Project base encoder output to cross_embed_dim if needed
            if base_enc_output_dim != cross_embed_dim:
                self.base_cross_proj = nn.Linear(base_enc_output_dim, cross_embed_dim)
            else:
                self.base_cross_proj = nn.Identity()
            
            # Project forecast encoder output to cross_embed_dim if needed
            if forecast_enc_output_dim != cross_embed_dim:
                self.forecast_cross_proj = nn.Linear(forecast_enc_output_dim, cross_embed_dim)
            else:
                self.forecast_cross_proj = nn.Identity()
            
            # Final projection to output_dim (use max of base/forecast for fusion)
            fusion_input_dim = cross_embed_dim * 2  # Concatenated base + forecast after cross-attention
            if fusion_input_dim != actual_output_dim:
                self.cross_attn_proj = nn.Linear(fusion_input_dim, actual_output_dim)
            else:
                self.cross_attn_proj = nn.Identity()
            
            # IMPROVED: Layer normalization before cross-attention for stability
            # Use actual encoder output dimensions
            self.base_norm = nn.LayerNorm(base_enc_output_dim)
            self.forecast_norm = nn.LayerNorm(forecast_enc_output_dim)
            
            # IMPROVED: Gated fusion mechanism to learn optimal base-forecast combination
            # Gate learns when to trust forecast-enhanced features vs original encodings
            # Outputs gate values for each dimension
            # Input is concatenated enhanced features (actual_output_dim) + original (base_enc_output_dim + forecast_enc_output_dim)
            fusion_gate_input_dim = actual_output_dim + base_enc_output_dim + forecast_enc_output_dim
            self.fusion_gate = nn.Sequential(
                nn.Linear(fusion_gate_input_dim, actual_output_dim),  # Input: concat of enhanced + original
                nn.Sigmoid()  # Outputs [0, 1] gate values per dimension
            )
            
            # IMPROVED: Final fusion layer with residual connection and better capacity
            self.fusion_layer = nn.Sequential(
                nn.Linear(actual_output_dim, actual_output_dim * 2),  # Expand capacity
                nn.LayerNorm(actual_output_dim * 2),
                nn.GELU(),  # GELU for smoother gradients
                nn.Dropout(dropout * 0.5),  # Lighter dropout
                nn.Linear(actual_output_dim * 2, actual_output_dim),
                nn.Dropout(dropout * 0.5)
            )
            
            # Residual connection projection (identity since dimensions match)
            self.residual_proj = nn.Identity()  # No projection needed since input and output dimensions are the same
        else:
            # Standard GNN encoder (Tier 1 or non-hierarchical Tier 2)
            # Build GAT layers with multi-head attention
            # Each observation dimension is a node with 1 feature (its value)
            # So input to first GAT layer is 1 feature per node
            
            # ROBUST FIX: Ensure hidden_dim is divisible by num_heads for intermediate layers
            adjusted_hidden_dim = hidden_dim
            if adjusted_hidden_dim % num_heads != 0:
                # Round down to nearest value divisible by num_heads
                adjusted_hidden_dim = (adjusted_hidden_dim // num_heads) * num_heads
                if adjusted_hidden_dim == 0:
                    adjusted_hidden_dim = num_heads  # At least num_heads
                # Update self.hidden_dim to reflect the actual hidden dimension
                self.hidden_dim = adjusted_hidden_dim
            
            self.gat_layers = nn.ModuleList()

            # First layer: 1 feature per node → adjusted_hidden_dim (with multi-head)
            self.gat_layers.append(GraphAttentionLayer(1, adjusted_hidden_dim, dropout, num_heads=num_heads))

            # Middle layers: adjusted_hidden_dim → adjusted_hidden_dim (with multi-head)
            for _ in range(num_layers - 2):
                self.gat_layers.append(GraphAttentionLayer(adjusted_hidden_dim, adjusted_hidden_dim, dropout, num_heads=num_heads))

            # Last layer: hidden_dim → output_dim (with multi-head)
            # ROBUST FIX: Ensure output_dim is divisible by num_heads
            final_output_dim = output_dim
            if final_output_dim % num_heads != 0:
                # Round down to nearest value divisible by num_heads
                final_output_dim = (final_output_dim // num_heads) * num_heads
                if final_output_dim == 0:
                    final_output_dim = num_heads  # At least num_heads
                # Update self.output_dim to reflect the actual output dimension
                self.output_dim = final_output_dim
            
            if num_layers > 1:
                # Use adjusted_hidden_dim (may have been adjusted above)
                actual_hidden_dim = getattr(self, 'hidden_dim', adjusted_hidden_dim)
                self.gat_layers.append(GraphAttentionLayer(actual_hidden_dim, final_output_dim, dropout, num_heads=num_heads))
            else:
                # Single layer case
                self.gat_layers[0] = GraphAttentionLayer(1, final_output_dim, dropout, num_heads=num_heads)
            
            # Adjacency matrix (fully-connected or learnable)
            if graph_type == 'full':
                # Fully-connected graph (all nodes connected)
                self.register_buffer('adj', torch.ones(obs_dim, obs_dim))
            elif graph_type == 'learned':
                # Learnable adjacency matrix with structured initialization for Tier 2
                if obs_dim == 14:  # Tier 2: encourage base-base and forecast-forecast connections
                    adj_init = self._initialize_structured_adjacency()
                    self.adj = nn.Parameter(adj_init)
                else:
                    self.adj = nn.Parameter(torch.ones(obs_dim, obs_dim))
            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")
            
            # IMPROVED: Attention pooling instead of mean pooling
            # This learns which features are most important for the task
            # ROBUST FIX: Use actual output_dim (may have been adjusted)
            actual_output_dim = getattr(self, 'output_dim', output_dim)
            if use_attention_pooling:
                self.attention_pool = nn.Sequential(
                    nn.Linear(actual_output_dim, actual_output_dim // 2),
                    nn.Tanh(),
                    nn.Linear(actual_output_dim // 2, 1)
                )
            else:
                # Fallback to mean pooling
                self.pool = lambda x: torch.mean(x, dim=1)  # (batch, num_nodes, features) → (batch, features)
    
    def _initialize_structured_adjacency(self):
        """
        Initialize adjacency matrix for hierarchical structure with:
        - Strong connections within base features
        - Strong connections within forecast features
        - Weaker but learnable connections between base and forecast
        """
        adj = torch.ones(self.obs_dim, self.obs_dim)
        base_dim = self.base_feature_dim  # 6 for investor, 4 for battery
        forecast_dim = self.obs_dim - base_dim  # 8 for investor, 4 for battery
        
        # Base-to-base: strong connections (2.0)
        adj[:base_dim, :base_dim] = 2.0
        
        # Forecast-to-forecast: strong connections (2.0)
        adj[base_dim:, base_dim:] = 2.0
        
        # Base-to-forecast: moderate connections (0.5) - learnable
        adj[:base_dim, base_dim:] = 0.5
        adj[base_dim:, :base_dim] = 0.5
        
        return adj
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Raw observations (batch, obs_dim)

        Returns:
            encoded: Encoded features (batch, output_dim)
        """
        if self.is_hierarchical:
            # TIER 2/3 IMPROVED: Enhanced hierarchical processing with residual connections and gated fusion
            # Split base and forecast features
            base_obs = obs[:, :self.base_feature_dim]  # (batch, base_feature_dim)
            
            # Handle bridge vectors for Tier 3 (18D observations)
            if self.has_bridge_vectors:
                # Tier 3: obs_dim = 18, base_feature_dim = 6
                # Structure: [6 base, 8 forecast, 4 bridge]
                # Extract only forecast features (skip bridge vectors)
                forecast_end_idx = self.base_feature_dim + self.forecast_feature_dim  # 6 + 8 = 14
                forecast_obs = obs[:, self.base_feature_dim:forecast_end_idx]  # (batch, 8) - only forecast
                bridge_obs = obs[:, forecast_end_idx:]  # (batch, 4) - bridge vectors for later fusion
            else:
                # Tier 2: obs_dim = 14, base_feature_dim = 6
                # Structure: [6 base, 8 forecast]
                forecast_obs = obs[:, self.base_feature_dim:]  # (batch, 8) - all remaining are forecast
                bridge_obs = None  # No bridge vectors
            
            # Encode separately
            base_encoded = self.base_encoder(base_obs)  # (batch, base_enc_output_dim)
            forecast_encoded = self.forecast_encoder(forecast_obs)  # (batch, forecast_enc_output_dim)
            
            # IMPROVED: Layer normalization before cross-attention for stability
            base_encoded_norm = self.base_norm(base_encoded)
            forecast_encoded_norm = self.forecast_norm(forecast_encoded)
            
            # ROBUST FIX: Project to cross-attention dimension if needed
            base_for_cross = self.base_cross_proj(base_encoded_norm)  # (batch, cross_embed_dim)
            forecast_for_cross = self.forecast_cross_proj(forecast_encoded_norm)  # (batch, cross_embed_dim)
            
            # Cross-attention: base attends to forecast and vice versa
            # Reshape for multihead attention: (seq_len, batch, embed_dim)
            base_seq = base_for_cross.unsqueeze(0)  # (1, batch, cross_embed_dim)
            forecast_seq = forecast_for_cross.unsqueeze(0)  # (1, batch, cross_embed_dim)
            
            # Cross-attention: base queries forecast and vice versa
            # Base queries forecast (base learns from forecast)
            base_enhanced, _ = self.cross_attention(base_seq, forecast_seq, forecast_seq)  # (1, batch, cross_embed_dim)
            base_enhanced = base_enhanced.squeeze(0)  # (batch, cross_embed_dim)
            
            # Forecast queries base (forecast learns from base)
            forecast_enhanced, _ = self.cross_attention(forecast_seq, base_seq, base_seq)  # (1, batch, cross_embed_dim)
            forecast_enhanced = forecast_enhanced.squeeze(0)  # (batch, cross_embed_dim)
            
            # Concatenate enhanced features after cross-attention
            combined_cross = torch.cat([base_enhanced, forecast_enhanced], dim=1)  # (batch, cross_embed_dim * 2)
            
            # Project to output_dim (features_dim)
            combined_enhanced = self.cross_attn_proj(combined_cross)  # (batch, output_dim)
            
            # IMPROVED: Residual connections - preserve original encodings
            # Concatenate original encodings for residual
            combined_original = torch.cat([base_encoded, forecast_encoded], dim=1)  # (batch, base_output_dim + forecast_output_dim = output_dim)
            
            # Verify dimensions match (they should after our fix)
            if combined_original.shape[1] != self.output_dim:
                # This should never happen, but project if needed
                if not hasattr(self, '_original_proj'):
                    self._original_proj = nn.Linear(combined_original.shape[1], self.output_dim).to(combined_original.device)
                combined_original = self._original_proj(combined_original)
            
            # IMPROVED: Gated fusion - learn optimal combination of original vs enhanced
            # Compute gate values (learns when to trust enhanced vs original)
            gate_input = torch.cat([combined_enhanced, combined_original], dim=1)  # (batch, output_dim * 2)
            gate_values = self.fusion_gate(gate_input)  # (batch, output_dim) - one gate value per dimension
            
            # Apply gating: weighted combination of enhanced and original
            # Gate learns per-dimension how much to trust enhanced features
            gated_combined = gate_values * combined_enhanced + (1.0 - gate_values) * combined_original
            
            # Final fusion with residual connection
            fused = self.fusion_layer(gated_combined)  # (batch, output_dim)
            encoded = fused + self.residual_proj(gated_combined)  # Residual connection
            
            # ROBUST VERIFICATION: Ensure output dimension matches expected
            if encoded.shape[1] != self.output_dim:
                raise RuntimeError(
                    f"GNN encoder output dimension mismatch: expected {self.output_dim}D, got {encoded.shape[1]}D. "
                    f"This indicates a bug in the forward pass."
                )
            
            # TIER 3: Fuse bridge vectors if present (maintains output_dim)
            # Bridge vectors are coordination signals from DL overlay
            if self.has_bridge_vectors and bridge_obs is not None and self.bridge_processor is not None:
                # Process bridge vectors through small MLP to output_dim
                bridge_encoded = self.bridge_processor(bridge_obs)  # (batch, 4) → (batch, output_dim)
                # Fuse bridge into encoded features via gated addition
                # Bridge provides coordination context, add to encoded features
                encoded = encoded + 0.3 * bridge_encoded  # Weighted fusion (bridge adds context)
            
            return encoded
        else:
            # Standard GNN processing
            # Reshape observations as graph nodes
            # Each observation dimension becomes a node with 1 feature (its value)
            # (batch, obs_dim) → (batch, obs_dim, 1)
            batch_size = obs.shape[0]
            h = obs.unsqueeze(2)  # (batch, obs_dim, 1)

            # Apply GAT layers with multi-head attention
            for i, gat in enumerate(self.gat_layers):
                h = gat(h, self.adj)
                if i < len(self.gat_layers) - 1:
                    h = F.elu(h)  # Activation between layers

            # IMPROVED: Attention pooling instead of mean pooling
            # This learns which features are most important for the task
            if self.use_attention_pooling:
                # Compute attention weights for each node
                attention_scores = self.attention_pool(h)  # (batch, obs_dim, 1)
                attention_weights = F.softmax(attention_scores, dim=1)  # (batch, obs_dim, 1)
                
                # Weighted sum of node features
                encoded = torch.sum(attention_weights * h, dim=1)  # (batch, output_dim)
            else:
                # Fallback to mean pooling
                encoded = self.pool(h)  # (batch, output_dim)

            return encoded


# ============================================================================
# Stable Baselines3 Integration
# ============================================================================

class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that uses GNN to encode observations.
    
    This replaces the standard MLP feature extractor in Stable Baselines3.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        graph_type: str = 'full',
        num_heads: int = 3,
        use_attention_pooling: bool = True,
        base_feature_dim: Optional[int] = None
    ):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        # IMPROVED: GNN encoder with multi-head attention and attention pooling
        # For Tier 2 (14D), pass base_feature_dim to enable hierarchical processing
        self.gnn_encoder = GNNObservationEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=features_dim,
            num_layers=num_layers,
            dropout=dropout,
            graph_type=graph_type,
            num_heads=num_heads,
            use_attention_pooling=use_attention_pooling,
            base_feature_dim=base_feature_dim
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, obs_dim)
            
        Returns:
            features: (batch, features_dim)
        """
        return self.gnn_encoder(observations)


class GNNActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy with GNN feature extraction.
    
    This policy uses a GNN encoder to preprocess observations before
    feeding them to the actor and critic networks.
    
    Usage:
        from stable_baselines3 import PPO
        from gnn_encoder import GNNActorCriticPolicy
        
        model = PPO(
            GNNActorCriticPolicy,
            env,
            policy_kwargs={
                "features_extractor_class": GNNFeaturesExtractor,
                "features_extractor_kwargs": {
                    "features_dim": 16,
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "dropout": 0.1,
                    "graph_type": "full"
                },
                "net_arch": [128, 64]  # Smaller MLP since GNN does feature extraction
            }
        )
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs
    ):
        # Default to smaller network since GNN does feature extraction
        if net_arch is None:
            net_arch = [128, 64]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )


# ============================================================================
# Policy Configuration Helper
# ============================================================================

def get_gnn_policy_kwargs(
    features_dim: int = 18,
    hidden_dim: int = 32,
    num_layers: int = 2,
    dropout: float = 0.1,
    graph_type: str = 'full',
    num_heads: int = 3,
    use_attention_pooling: bool = True,
    net_arch: Optional[List[int]] = None,
    base_feature_dim: Optional[int] = None
) -> Dict:
    """
    Helper function to get policy_kwargs for GNN policy.
    
    Args:
        features_dim: Output dimension of GNN encoder (default: 18, must be divisible by num_heads)
        hidden_dim: Hidden dimension of GAT layers (default: 32)
        num_layers: Number of GAT layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        graph_type: 'full', 'learned', or 'hierarchical' (default: 'full')
        num_heads: Number of attention heads (default: 3)
        use_attention_pooling: Use attention pooling instead of mean (default: True)
        net_arch: MLP architecture after GNN (default: [128, 64])
        base_feature_dim: For Tier 2 hierarchical mode, number of base features (6) before forecast features (8)
        
    Returns:
        policy_kwargs: Dict to pass to PPO(..., policy_kwargs=...)
    """
    # Validate that features_dim is divisible by num_heads
    if features_dim % num_heads != 0:
        raise ValueError(f"features_dim ({features_dim}) must be divisible by num_heads ({num_heads}). "
                        f"Suggested values: {num_heads * (features_dim // num_heads + 1)} or {num_heads * (features_dim // num_heads)}")
    if net_arch is None:
        net_arch = [128, 64]
    
    return {
        "features_extractor_class": GNNFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "graph_type": graph_type,
            "num_heads": num_heads,
            "use_attention_pooling": use_attention_pooling,
            "base_feature_dim": base_feature_dim
        },
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
        "normalize_images": False,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-8, "weight_decay": 0.0}
    }

