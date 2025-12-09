#!/usr/bin/env python3
"""
Single Agent Reinforcement Learning (SARL) DQN Optimizer
========================================================

Implements a unified Deep Q-Network agent for renewable energy portfolio optimization.
This serves as a baseline comparison to the multi-agent reinforcement learning approach.

Key Features:
- Unified decision making across investment, battery, risk, and meta control
- Deep Q-Network with experience replay and target networks
- Combined action space handling all portfolio decisions
- Direct comparison baseline to MARL approach

Academic Foundation:
- Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Van Hasselt et al. (2016) "Deep reinforcement learning with double q-learning"
- Wang et al. (2016) "Dueling network architectures for deep reinforcement learning"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class SARLConfig:
    """Configuration for SARL-DQN optimizer."""
    
    # Network architecture
    state_dim: int = 50
    action_dim: int = 12
    hidden_dims: List[int] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100000
    target_update_freq: int = 1000
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 50000
    
    # RL parameters
    gamma: float = 0.99
    tau: float = 0.005  # Soft target updates
    n_step: int = 3     # Multi-step returns
    
    # Training settings
    min_buffer_size: int = 1000
    train_freq: int = 4
    grad_clip: float = 10.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class ActorNetwork(nn.Module):
    """
    ADDED: Actor network for continuous action selection.
    Maps states directly to actions in [-1, 1] range.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(ActorNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        return self.network(state)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.

    This architecture helps with learning by separating the estimation of
    state values from action advantages, leading to better performance.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(DuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )

        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through dueling architecture.
        
        Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        """
        features = self.feature_layers(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage streams
        # Subtract mean advantage for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences based on their TD error magnitude,
    allowing the agent to learn more from surprising experiences.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, experience: Experience):
        """Add experience with maximum priority."""
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class SARLDQNOptimizer:
    """
    Single Agent Reinforcement Learning DQN Optimizer.
    
    Unified agent that handles all aspects of renewable energy portfolio optimization:
    - Investment decisions (wind, solar, hydro)
    - Battery operations (charge/discharge)
    - Risk management (hedging, cash reserves)
    - Meta control (frequencies, allocations)
    """
    
    def __init__(self, config: SARLConfig, device: str = 'cpu', state_dim: int = None):
        self.config = config
        self.device = torch.device(device)

        # Use provided state_dim or fall back to config
        actual_state_dim = state_dim if state_dim is not None else config.state_dim

        # ADDED: Initialize actor network for continuous action selection
        self.actor_network = ActorNetwork(
            actual_state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)

        self.target_actor = ActorNetwork(
            actual_state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)

        # Initialize Q-networks (critic)
        self.q_network = DuelingDQN(
            actual_state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)

        self.target_network = DuelingDQN(
            actual_state_dim,
            config.action_dim,
            config.hidden_dims
        ).to(self.device)

        # Copy weights to target networks
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_actor.load_state_dict(self.actor_network.state_dict())
        self.target_actor.eval()

        # Initialize optimizers (separate for actor and critic)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.epsilon = config.epsilon_start
        
        # Performance tracking
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_losses': [],
            'epsilon_values': [],
            'buffer_size': []
        }
        
        actor_params = sum(p.numel() for p in self.actor_network.parameters())
        critic_params = sum(p.numel() for p in self.q_network.parameters())
        logger.info(f"Initialized SARL Actor-Critic: Actor={actor_params} params, Critic={critic_params} params")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        FIXED: Select action using actor network (proper continuous action selection).

        Uses a dedicated actor network that maps states to actions in [-1, 1] range.
        This is the correct approach for continuous action spaces.

        Args:
            state: Current environment state
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action vector
        """
        if training and random.random() < self.epsilon:
            # Random exploration
            action = np.random.uniform(-1, 1, self.config.action_dim)
        else:
            # FIXED: Use actor network for deterministic action selection
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.actor_network(state_tensor).cpu().numpy().flatten()

        # Add small noise during training for exploration
        if training and self.epsilon > 0.01:
            noise = np.random.normal(0, 0.1 * self.epsilon, size=action.shape)
            action = action + noise

        return np.clip(action, -1, 1)
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step if buffer has enough experiences.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {}
        
        if self.steps % self.config.train_freq != 0:
            return {}
        
        # Sample batch from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        if not experiences:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # FIXED: Actor-Critic training
        # Step 1: Train Critic (Q-network)
        # Current Q-values for the actions that were actually taken
        current_q_values = self.q_network(states)
        # Use mean of Q-values as state value estimate (simplified)
        current_q = current_q_values.mean(dim=1)

        # Target Q-values using target networks
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.target_actor(next_states)

            # Get Q-values for next state-action pairs
            target_next_q_values = self.target_network(next_states)
            target_q = target_next_q_values.mean(dim=1)  # Use mean as state value

            target_q_values = rewards + (self.config.gamma * target_q * ~dones)

        # Calculate TD errors for priority updates
        td_errors = (current_q - target_q_values).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Calculate weighted critic loss
        critic_loss = F.mse_loss(current_q, target_q_values, reduction='none')
        weighted_critic_loss = (critic_loss * weights_tensor).mean()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        weighted_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.critic_optimizer.step()

        # Step 2: Train Actor (policy network)
        # Actor loss: maximize Q-value of actions produced by actor
        actor_actions = self.actor_network(states)
        actor_q_values = self.q_network(states)
        actor_loss = -actor_q_values.mean()  # Negative because we want to maximize Q-values

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.config.grad_clip)
        self.actor_optimizer.step()
        
        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            self._soft_update_target_network()
        
        # Update epsilon
        self._update_epsilon()
        
        # Update step counter
        self.steps += 1
        
        # Record metrics
        metrics = {
            'critic_loss': weighted_critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'mean_q_value': current_q.mean().item(),
            'td_error_mean': np.abs(td_errors).mean()
        }

        self.training_metrics['q_losses'].append(metrics['critic_loss'])
        self.training_metrics['epsilon_values'].append(metrics['epsilon'])
        self.training_metrics['buffer_size'].append(metrics['buffer_size'])

        return metrics

    def _soft_update_target_network(self):
        """FIXED: Soft update of both target networks (actor and critic)."""
        # Update critic target
        for target_param, local_param in zip(self.target_network.parameters(),
                                           self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data +
                (1.0 - self.config.tau) * target_param.data
            )

        # Update actor target
        for target_param, local_param in zip(self.target_actor.parameters(),
                                           self.actor_network.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data +
                (1.0 - self.config.tau) * target_param.data
            )
    
    def _update_epsilon(self):
        """Update epsilon for exploration decay."""
        if self.steps < self.config.epsilon_decay:
            self.epsilon = (
                self.config.epsilon_start - 
                (self.config.epsilon_start - self.config.epsilon_end) * 
                (self.steps / self.config.epsilon_decay)
            )
        else:
            self.epsilon = self.config.epsilon_end
    
    def save_model(self, filepath: str):
        """Save model and training state."""
        torch.save({
            'actor_network_state_dict': self.actor_network.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'training_metrics': self.training_metrics
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor_network.load_state_dict(checkpoint['actor_network_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        self.training_metrics = checkpoint['training_metrics']

        logger.info(f"Model loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        return {
            'total_steps': self.steps,
            'total_episodes': self.episodes,
            'current_epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'network_parameters': sum(p.numel() for p in self.q_network.parameters()),
            'recent_metrics': {
                'avg_q_loss': np.mean(self.training_metrics['q_losses'][-100:]) if self.training_metrics['q_losses'] else 0,
                'avg_episode_reward': np.mean(self.training_metrics['episode_rewards'][-100:]) if self.training_metrics['episode_rewards'] else 0,
                'avg_episode_length': np.mean(self.training_metrics['episode_lengths'][-100:]) if self.training_metrics['episode_lengths'] else 0,
            }
        }


class ActionSpaceConverter:
    """
    Converts between SARL unified actions and MARL distributed actions.

    This class handles the mapping between the single agent's unified action space
    and the original multi-agent action spaces for compatibility with the existing environment.
    """

    def __init__(self):
        # Define action space mappings
        self.action_mappings = {
            'investor': slice(0, 3),      # Wind, solar, hydro investments
            'battery': slice(3, 5),       # Battery operations
            'risk': slice(5, 8),          # Risk management
            'meta': slice(8, 12)          # Meta control
        }

        # Action space bounds for each agent
        self.action_bounds = {
            'investor': {'low': -1.0, 'high': 1.0},
            'battery': {'low': -1.0, 'high': 1.0},
            'risk': {'low': 0.0, 'high': 1.0},
            'meta': {'low': 0.0, 'high': 1.0}
        }

    def unified_to_distributed(self, unified_action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert unified SARL action to distributed MARL actions.

        Args:
            unified_action: Single action vector from SARL agent

        Returns:
            Dictionary of actions for each agent type
        """
        distributed_actions = {}

        for agent_type, action_slice in self.action_mappings.items():
            agent_action = unified_action[action_slice]

            # Apply bounds clipping
            bounds = self.action_bounds[agent_type]
            agent_action = np.clip(agent_action, bounds['low'], bounds['high'])

            distributed_actions[agent_type] = agent_action

        return distributed_actions

    def distributed_to_unified(self, distributed_actions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert distributed MARL actions to unified SARL action.

        Args:
            distributed_actions: Dictionary of actions from each agent

        Returns:
            Single unified action vector
        """
        unified_action = np.zeros(12)

        for agent_type, action_slice in self.action_mappings.items():
            if agent_type in distributed_actions:
                agent_action = distributed_actions[agent_type]
                unified_action[action_slice] = agent_action

        return unified_action
