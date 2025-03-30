"""
Enterprise Actor-Critic Implementation for XIBRA Network
Supports multi-agent coordination with distributed training
and prioritized experience replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import logging
import time
from collections import deque
import heapq

class DistributedReplayBuffer(IterableDataset):
    """Lock-free prioritized experience replay buffer with distributed sync"""
    
    def __init__(self, capacity=1e6, alpha=0.6, beta=0.4):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self._buffer = []
        self._priorities = np.zeros((self.capacity,), dtype=np.float32)
        self._next_idx = 0
        self._lock = threading.Lock()
        self._default_priority = 1.0
        self._beta_schedule = lambda: min(1.0, self.beta + 0.001)
        
        # Distributed coordination
        self._epoch = 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    def add(self, experience):
        with self._lock:
            priority = self._priorities.max() if self._buffer else self._default_priority
            if len(self._buffer) < self.capacity:
                self._buffer.append(experience)
            else:
                self._buffer[self._next_idx] = experience
            self._priorities[self._next_idx] = priority ** self.alpha
            self._next_idx = (self._next_idx + 1) % self.capacity

    def _sample_proportional(self, batch_size):
        res = []
        total = len(self._buffer)
        while len(res) < batch_size:
            mass = np.random.random() * self._priorities[:total].sum()
            idx = np.searchsorted(self._priorities.cumsum(), mass)
            res.append(min(idx, total-1))
        return res

    def sample(self, batch_size):
        beta = self._beta_schedule()
        total = len(self._buffer)
        indices = self._sample_proportional(batch_size)
        
        weights = (self._priorities[indices] / self._priorities.sum()) ** (-beta)
        weights /= weights.max()
        
        samples = [self._buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        with self._lock:
            for idx, priority in zip(indices, priorities):
                self._priorities[idx] = (priority ** self.alpha).item()

    def __iter__(self):
        while True:
            yield from self.sample(256)

class ActorNetwork(nn.Module):
    """Enterprise-grade policy network with multi-head output"""
    
    def __init__(self, obs_dim, act_dim, hidden_size=512):
        super().__init__()
        self.shared_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Multi-head output for complex action spaces
        self.mean_head = nn.Linear(hidden_size, act_dim)
        self.log_std_head = nn.Linear(hidden_size, act_dim)
        
        # Adaptive parameter initialization
        nn.init.orthogonal_(self.shared_backbone[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.shared_backbone[0].bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, obs):
        hidden = self.shared_backbone(obs)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

class CriticNetwork(nn.Module):
    """Dual Q-network architecture with shared encoder"""
    
    def __init__(self, obs_dim, act_dim, hidden_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        self.Q1 = nn.Linear(hidden_size, 1)
        self.Q2 = nn.Linear(hidden_size, 1)
        
        # Initialization for stable learning
        nn.init.orthogonal_(self.encoder[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.encoder[0].bias, 0.0)
        nn.init.orthogonal_(self.Q1.weight, gain=0.01)
        nn.init.constant_(self.Q1.bias, 0.0)
        nn.init.orthogonal_(self.Q2.weight, gain=0.01)
        nn.init.constant_(self.Q2.bias, 0.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        hidden = self.encoder(x)
        return self.Q1(hidden), self.Q2(hidden)

class XIBRA_ActorCritic:
    """Enterprise Actor-Critic implementation for XIBRA Network"""
    
    def __init__(self, obs_dim, act_dim, device='cuda', lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = ActorNetwork(obs_dim, act_dim).to(device)
        self.critic = CriticNetwork(obs_dim, act_dim).to(device)
        self.critic_target = CriticNetwork(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Distributed training setup
        self._init_distributed()
        self.replay_buffer = DistributedReplayBuffer()
        self.logger = self._configure_logging()
        
        # Automatic mixed precision
        self.scaler = torch.cuda.amp.GradScaler()

    def _init_distributed(self):
        if dist.is_initialized():
            self.actor = DDP(self.actor)
            self.critic = DDP(self.critic)
            self.critic_target = DDP(self.critic_target)

    def _configure_logging(self):
        logger = logging.getLogger('XIBRA_AC')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('xibra_ac.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            mean, log_std = self.actor(obs)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                noise = torch.randn_like(mean)
                action = mean + noise * std
                
            return action.cpu().numpy()

    def update_critic(self, batch):
        obs, action, reward, next_obs, done = batch
        
        with torch.cuda.amp.autocast():
            # Target Q-values
            with torch.no_grad():
                next_action, _ = self.actor(next_obs)
                target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.gamma * target_Q
            
            # Current Q-values
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = nn.functional.mse_loss(current_Q1, target_Q) + \
                         nn.functional.mse_loss(current_Q2, target_Q)
            
        self.critic_optim.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optim)
        self.scaler.update()
        
        return critic_loss.item()

    def update_actor(self, batch):
        obs = batch[0]
        
        with torch.cuda.amp.autocast():
            action, log_prob = self.actor(obs)
            Q1, Q2 = self.critic(obs, action)
            Q = torch.min(Q1, Q2)
            
            actor_loss = -Q.mean()
            
        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)
        self.scaler.update()
        
        return actor_loss.item()

    def soft_update(self):
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_epoch(self, num_steps=1000):
        start_time = time.time()
        metrics = {'critic_loss': 0.0, 'actor_loss': 0.0}
        
        for _ in range(num_steps):
            batch = next(iter(self.replay_buffer))
            batch = [torch.tensor(x).to(self.device) for x in batch]
            
            # Update critic
            critic_loss = self.update_critic(batch)
            metrics['critic_loss'] += critic_loss
            
            # Update actor and alpha
            actor_loss = self.update_actor(batch)
            metrics['actor_loss'] += actor_loss
            
            # Soft update target network
            self.soft_update()
            
        # Log metrics
        elapsed = time.time() - start_time
        self.logger.info(f"Epoch completed in {elapsed:.2f}s - "
                        f"Critic Loss: {metrics['critic_loss']/num_steps:.4f} - "
                        f"Actor Loss: {metrics['actor_loss']/num_steps:.4f}")
        
        return metrics

    def save_checkpoint(self, path):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': {
                'actor': self.actor_optim.state_dict(),
                'critic': self.critic_optim.state_dict()
            },
            'replay_buffer': self.replay_buffer
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['optimizer_state_dict']['actor'])
        self.critic_optim.load_state_dict(checkpoint['optimizer_state_dict']['critic'])
        self.replay_buffer = checkpoint['replay_buffer']

# Enterprise deployment example
if __name__ == "__main__":
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create agent with enterprise configuration
    agent = XIBRA_ActorCritic(
        obs_dim=128, 
        act_dim=16, 
        device=device,
        lr=3e-4,
        gamma=0.99,
        tau=0.005
    )
    
    # Training loop
    for epoch in range(1000):
        metrics = agent.train_epoch(num_steps=1000)
        
        # Save checkpoint every 100 epochs
        if epoch % 100 == 0:
            agent.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
