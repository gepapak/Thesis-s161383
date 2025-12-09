"""
GNN Encoder for Tier 2 Forecast Integration

This module provides a Graph Neural Network (GNN) encoder and custom PPO policy
that uses the GNN to preprocess observations before feeding them to the standard
actor-critic network.

Architecture:
    Observations (9D) → GNN Encoder (16D) → MLP Actor/Critic → Actions/Values

Compatible with Stable Baselines3 PPO.

REFACTORED: Merged custom_policy.py and gnn_encoder.py into this unified module.
All GNN-related functionality is now in one place for Tier 2 with GNN encoder.
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
        obs_dim: Input observation dimension (9 for investor, 8 for battery)
        hidden_dim: Hidden dimension for GAT layers (default: 32)
        output_dim: Output encoded feature dimension (default: 16)
        num_layers: Number of GAT layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        graph_type: 'full' (fully-connected) or 'learned' (learnable adjacency)
        num_heads: Number of attention heads (default: 3)
        use_attention_pooling: Use attention pooling instead of mean (default: True)
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
        use_attention_pooling: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.graph_type = graph_type
        self.num_heads = num_heads
        self.use_attention_pooling = use_attention_pooling
        
        # Build GAT layers with multi-head attention
        # Each observation dimension is a node with 1 feature (its value)
        # So input to first GAT layer is 1 feature per node
        self.gat_layers = nn.ModuleList()

        # First layer: 1 feature per node → hidden_dim (with multi-head)
        self.gat_layers.append(GraphAttentionLayer(1, hidden_dim, dropout, num_heads=num_heads))

        # Middle layers: hidden_dim → hidden_dim (with multi-head)
        for _ in range(num_layers - 2):
            self.gat_layers.append(GraphAttentionLayer(hidden_dim, hidden_dim, dropout, num_heads=num_heads))

        # Last layer: hidden_dim → output_dim (with multi-head)
        if num_layers > 1:
            self.gat_layers.append(GraphAttentionLayer(hidden_dim, output_dim, dropout, num_heads=num_heads))
        else:
            # Single layer case
            self.gat_layers[0] = GraphAttentionLayer(1, output_dim, dropout, num_heads=num_heads)
        
        # Adjacency matrix (fully-connected or learnable)
        if graph_type == 'full':
            # Fully-connected graph (all nodes connected)
            self.register_buffer('adj', torch.ones(obs_dim, obs_dim))
        elif graph_type == 'learned':
            # Learnable adjacency matrix
            self.adj = nn.Parameter(torch.ones(obs_dim, obs_dim))
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # IMPROVED: Attention pooling instead of mean pooling
        # This learns which features are most important for the task
        if use_attention_pooling:
            self.attention_pool = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.Tanh(),
                nn.Linear(output_dim // 2, 1)
            )
        else:
            # Fallback to mean pooling
            self.pool = lambda x: torch.mean(x, dim=1)  # (batch, num_nodes, features) → (batch, features)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Raw observations (batch, obs_dim)

        Returns:
            encoded: Encoded features (batch, output_dim)
        """
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
        use_attention_pooling: bool = True
    ):
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        # IMPROVED: GNN encoder with multi-head attention and attention pooling
        self.gnn_encoder = GNNObservationEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=features_dim,
            num_layers=num_layers,
            dropout=dropout,
            graph_type=graph_type,
            num_heads=num_heads,
            use_attention_pooling=use_attention_pooling
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
    net_arch: Optional[List[int]] = None
) -> Dict:
    """
    Helper function to get policy_kwargs for GNN policy.
    
    Args:
        features_dim: Output dimension of GNN encoder (default: 18, must be divisible by num_heads)
        hidden_dim: Hidden dimension of GAT layers (default: 32)
        num_layers: Number of GAT layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        graph_type: 'full' or 'learned' (default: 'full')
        num_heads: Number of attention heads (default: 3)
        use_attention_pooling: Use attention pooling instead of mean (default: True)
        net_arch: MLP architecture after GNN (default: [128, 64])
        
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
            "use_attention_pooling": use_attention_pooling
        },
        "net_arch": net_arch,
        "activation_fn": nn.ReLU,
        "normalize_images": False,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-8, "weight_decay": 0.0}
    }

