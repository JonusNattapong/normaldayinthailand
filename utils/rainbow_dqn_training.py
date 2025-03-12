import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from collections import deque, namedtuple

logger = logging.getLogger(__name__)

# Define experience tuple structure with priority
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, base_model_path, vocab_size, atom_size=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        hidden_size = self.base_model.config.hidden_size
        
        self.vocab_size = vocab_size  # Number of possible actions (tokens)
        self.atom_size = atom_size  # Number of atoms in value distribution
        self.v_min = v_min  # Minimum value for distributional RL
        self.v_max = v_max  # Maximum value for distributional RL
        self.supports = torch.linspace(v_min, v_max, atom_size).to(next(self.base_model.parameters()).device)
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        
        # Dueling network architecture
        self.advantage_hidden = NoisyLinear(hidden_size, hidden_size // 2)
        self.value_hidden = NoisyLinear(hidden_size, hidden_size // 2)
        
        self.advantage = NoisyLinear(hidden_size // 2, vocab_size * atom_size)
        self.value = NoisyLinear(hidden_size // 2, atom_size)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        advantage_hidden = F.relu(self.advantage_hidden(last_token_hidden))
        value_hidden = F.relu(self.value_hidden(last_token_hidden))
        
        advantage = self.advantage(advantage_hidden).view(-1, self.vocab_size, self.atom_size)
        value = self.value(value_hidden).view(-1, 1, self.atom_size)
        
        # Combine value and advantage using dueling network formula
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Get probabilities with softmax
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # Avoid NaNs
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers"""
        self.advantage_hidden.reset_noise()
        self.value_hidden.reset_noise()
        self.advantage.reset_noise()
        self.value.reset_noise()
    
    def act(self, input_ids, attention_mask=None):
        """Select action using the policy network with the highest expected value"""
        dist = self.forward(input_ids, attention_mask)
        expected_value = (dist * self.supports).sum(-1)
        action = expected_value.argmax(dim=-1).unsqueeze(-1)
        return action

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for storing experience samples"""
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization is used (0 = no prioritization)
        self.beta = beta    # Importance-sampling, corrects bias in updates (0 = no correction)
        self.beta_increment = beta_increment  # Controls beta schedule
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        """Store experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of experiences with priorities"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Prioritized sampling
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights
        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
    def __len__(self):
        return len(self.buffer)

def train_rainbow_dqn(
    model_path,
    dataset,
    output_dir,
    reward_model_path=None,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    gamma=0.99,            # Discount factor
    target_update=10,      # How often to update target network
    buffer_size=10000,
    atom_size=51,          # Number of atoms in distribution
    v_min=-10,            # Minimum value for distribution
    v_max=10,             # Maximum value for distribution
    n_step=3,              # Multi-step learning
    priority_alpha=0.6,    # Prioritized experience replay alpha
    priority_beta=0.4      # Prioritized experience replay initial beta
):
    """
    Train a Rainbow DQN model for language generation
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        reward_model_path: Path to a pre-trained reward model (optional)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
        gamma: Discount factor
        target_update: How many steps between target network updates
        buffer_size: Size of replay buffer
        atom_size: Number of atoms in categorical distribution
        v_min: Minimum value in value distribution
        v_max: Maximum value in value distribution
        n_step: Number of steps for multi-step learning
        priority_alpha: Alpha parameter for prioritized replay
        priority_beta: Initial beta parameter for prioritized replay
    
    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and set vocabulary size
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    # Load reward model if provided
    reward_model = None
    if reward_model_path:
        try:
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path).to(device)
            logger.info(f"Loaded reward model from {reward_model_path}")
        except Exception as e:
            logger.warning(f"Could not load reward model: {e}")
    
    # Initialize networks
    online_net = RainbowDQN(model_path, vocab_size, atom_size, v_min, v_max).to(device)
    target_net = RainbowDQN(model_path, vocab_size, atom_size, v_min, v_max).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()  # Target network is not trained directly
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    
    # Initialize replay buffer with prioritized experience replay
    replay_buffer = PrioritizedReplayBuffer(
        capacity=buffer_size, 
        alpha=priority_alpha, 
        beta=priority_beta
    )
    
    # Initialize n-step buffer for multi-step learning
    n_step_buffer = deque(maxlen=n_step)
    
    logger.info("Starting Rainbow DQN training...")
    
    steps_done = 0
    gamma_n = gamma ** n_step
    
    # Training loop
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            # Process batch data
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Reset noise for exploration
            online_net.reset_noise()
            target_net.reset_noise()
            
            # State = current input sequence
            state = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }
            
            # Select action using online network
            with torch.no_grad():
                action = online_net.act(inputs.input_ids, inputs.attention_mask)
            
            # Get reward (from reward model or simple heuristic)
            with torch.no_grad():
                if reward_model:
                    # Add generated token to input and get reward from reward model
                    extended_inputs = torch.cat([inputs.input_ids, action], dim=1)
                    extended_attention_mask = torch.cat([inputs.attention_mask, torch.ones_like(action)], dim=1)
                    reward_output = reward_model(extended_inputs, attention_mask=extended_attention_mask)
                    reward = reward_output.logits[:, -1].unsqueeze(-1)
                else:
                    # Simple heuristic reward based on token distribution
                    dist = online_net(inputs.input_ids, inputs.attention_mask)
                    expected_values = (dist * online_net.supports).sum(-1)
                    reward = expected_values.gather(1, action).detach() / 10  # Scale rewards
            
            # Next state = state with action appended
            next_state_ids = torch.cat([inputs.input_ids, action], dim=1)
            next_attention_mask = torch.cat([
                inputs.attention_mask, 
                torch.ones_like(action)
            ], dim=1)
            
            next_state = {
                'input_ids': next_state_ids,
                'attention_mask': next_attention_mask
            }
            
            # Done flag (assuming sequence isn't done until max length)
            done = torch.zeros(batch_size, 1).to(device)
            
            # N-step returns
            n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(n_step_buffer) < n_step:
                continue
            
            # Get n-step transition
            state_n, action_n, reward_n, next_state_n, done_n = n_step_buffer[0]
            
            # Calculate n-step reward
            for j in range(1, n_step):
                reward_j = n_step_buffer[j][2]
                done_j = n_step_buffer[j][4]
                reward_n += (gamma ** j) * reward_j * (1 - done_j)
                if done_j:
                    break
            
            # Store experience in replay buffer
            for j in range(batch_size):
                replay_buffer.push(
                    {k: v[j:j+1] for k, v in state_n.items()},
                    action_n[j:j+1],
                    reward_n[j:j+1],
                    {k: v[j:j+1] for k, v in next_state_n.items()},
                    done_n[j:j+1]
                )
            
            total_reward += reward.mean().item()
            steps_done += 1
            
                        # Training step if enough experiences are accumulated
            if len(replay_buffer) > batch_size:
                # Sample a batch from replay buffer with priorities
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_batch = {
                    'input_ids': torch.cat([s['input_ids'] for s in states], dim=0).to(device),
                    'attention_mask': torch.cat([s['attention_mask'] for s in states], dim=0).to(device)
                }
                actions_batch = torch.cat([a for a in actions], dim=0).to(device)
                rewards_batch = torch.cat([r for r in rewards], dim=0).to(device)
                next_states_batch = {
                    'input_ids': torch.cat([ns['input_ids'] for ns in next_states], dim=0).to(device),
                    'attention_mask': torch.cat([ns['attention_mask'] for ns in next_states], dim=0).to(device)
                }
                dones_batch = torch.cat([d for d in dones], dim=0).to(device)
                weights = torch.FloatTensor(weights).to(device)
                
                # Compute current distribution
                current_dist = online_net(states_batch['input_ids'], states_batch['attention_mask'])
                
                # Compute log probability of actions taken
                log_probs = torch.log(current_dist[range(batch_size), actions_batch.squeeze()])
                
                # N-step Distributional RL update
                with torch.no_grad():
                    # Get next state distribution from target network
                    next_dist = target_net(next_states_batch['input_ids'], next_states_batch['attention_mask'])
                    
                    # Get action with highest expected value from online network
                    next_action = online_net(
                        next_states_batch['input_ids'], 
                        next_states_batch['attention_mask']
                    ).mean(-1).argmax(1, keepdim=True)
                    
                    # Get distribution for chosen actions
                    next_dist = next_dist[range(batch_size), next_action.squeeze()]
                    
                    # Compute target distribution (Categorical projection)
                    Tz = rewards_batch + (1 - dones_batch) * gamma_n * online_net.supports
                    Tz = Tz.clamp(online_net.v_min, online_net.v_max)
                    
                    # Compute projection
                    b = (Tz - online_net.v_min) / online_net.delta_z
                    l = b.floor().long()
                    u = b.ceil().long()
                    
                    # Distribute probability of Tz
                    target_dist = torch.zeros_like(next_dist)
                    
                    for j in range(batch_size):
                        for atom in range(online_net.atom_size):
                            target_dist[j, l[j, atom]] += next_dist[j, atom] * (u[j, atom] - b[j, atom])
                            target_dist[j, u[j, atom]] += next_dist[j, atom] * (b[j, atom] - l[j, atom])
                
                # Compute KL divergence loss
                loss = -(target_dist * torch.log(current_dist[range(batch_size), actions_batch.squeeze()])).sum(-1)
                
                # Apply importance sampling weights from PER
                loss = (loss * weights).mean()
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to stabilize training
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
                optimizer.step()
                
                # Update priorities in buffer
                priorities = loss.detach().cpu().numpy() + 1e-5  # Add small constant to avoid zero priorities
                replay_buffer.update_priorities(indices, priorities)
                
                total_loss += loss.item()
                
                # Update target network
                if steps_done % target_update == 0:
                    target_net.load_state_dict(online_net.state_dict())
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    # Save the fine-tuned model
    online_net.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Rainbow DQN training complete. Model saved to {output_dir}")
    return online_net.base_model, tokenizer