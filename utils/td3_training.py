import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    def __init__(self, base_model_path, vocab_size):
        super(Actor, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        # Actor head for action selection
        self.actor_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size // 2, vocab_size),
            nn.Tanh()  # Output scaled to [-1, 1] for TD3
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        # Output action (continuous representation in vocabulary space)
        action = self.actor_head(last_token_hidden)
        return action

class Critic(nn.Module):
    def __init__(self, base_model_path, vocab_size):
        super(Critic, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        # TD3 uses two critics for learning
        self.q1 = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size + vocab_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size // 2, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size + vocab_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, action, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        # Concatenate state representation with action
        state_action = torch.cat([last_token_hidden, action], dim=1)
        
        # Output Q-values from both critics
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        
        return q1, q2
    
    def q1_forward(self, input_ids, action, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        # Concatenate state representation with action
        state_action = torch.cat([last_token_hidden, action], dim=1)
        
        # Output Q-value from first critic only
        q1 = self.q1(state_action)
        return q1

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)

def train_td3(
    model_path,
    dataset,
    output_dir,
    reward_model_path=None,
    batch_size=4,
    epochs=1,
    actor_lr=1e-5,
    critic_lr=1e-4,
    gamma=0.99,
    tau=0.005,  # Target network update rate
    policy_noise=0.2,  # Noise added to target policy
    noise_clip=0.5,  # Limit for noise
    policy_freq=2,  # Frequency of delayed policy updates
    buffer_size=10000,
    noise_std=0.1  # Exploration noise
):
    """
    Train a TD3 (Twin Delayed Deep Deterministic Policy Gradient) model for language generation
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        reward_model_path: Path to a pre-trained reward model (optional)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        actor_lr: Learning rate for actor network
        critic_lr: Learning rate for critic network
        gamma: Discount factor
        tau: Target network update rate
        policy_noise: Noise added to target policy
        noise_clip: Limit for noise
        policy_freq: Frequency of delayed policy updates
        buffer_size: Size of replay buffer
        noise_std: Standard deviation of exploration noise
    
    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load reward model if provided
    reward_model = None
    if reward_model_path:
        try:
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path).to(device)
            logger.info(f"Loaded reward model from {reward_model_path}")
        except Exception as e:
            logger.warning(f"Could not load reward model: {e}")
    
    # Initialize actor, critic and their target networks
    actor = Actor(model_path, tokenizer.vocab_size).to(device)
    critic = Critic(model_path, tokenizer.vocab_size).to(device)
    
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    
    # Initialize optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    logger.info("Starting TD3 training...")
    
    # Training loop
    total_steps = 0
    for epoch in range(epochs):
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            # Process batch data
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # State = current input sequence
            state = inputs
            
            # Get action from actor network with exploration noise
            with torch.no_grad():
                action = actor(inputs.input_ids, inputs.attention_mask)
                action = action + torch.normal(0, noise_std, size=action.size()).to(device)
                action = torch.clamp(action, -1, 1)  # Clamp to action space
            
            # Convert continuous action to token ID (e.g., by discretizing or nearest neighbor)
            token_distribution = F.softmax(action, dim=-1)
            next_token = torch.multinomial(token_distribution, 1)
            
            # Get reward (from reward model or simple heuristic)
            with torch.no_grad():
                if reward_model:
                    # Add generated token to input and get reward from reward model
                    extended_inputs = torch.cat([inputs.input_ids, next_token], dim=1)
                    reward_output = reward_model(extended_inputs)
                    reward = reward_output.logits[:, -1].unsqueeze(-1)
                else:
                    # Simple heuristic reward based on token likelihood
                    reward = token_distribution.gather(1, next_token)
            
            # Next state = state with action appended
            next_state_ids = torch.cat([inputs.input_ids, next_token], dim=1)
            next_attention_mask = torch.cat([
                inputs.attention_mask, 
                torch.ones_like(next_token)
            ], dim=1)
            
            next_state = {
                'input_ids': next_state_ids,
                'attention_mask': next_attention_mask
            }
            
            # Done flag (assuming sequence isn't done until max length)
            done = torch.zeros(batch_size, 1).to(device)
            
            # Store experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            
            total_reward += reward.mean().item()
            total_steps += 1
            
            # Training step if enough experiences are accumulated
            if len(replay_buffer) > batch_size:
                # Sample a batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_batch = {
                    'input_ids': torch.cat([s['input_ids'] for s in states], dim=0),
                    'attention_mask': torch.cat([s['attention_mask'] for s in states], dim=0)
                }
                actions_batch = torch.cat([a for a in actions], dim=0)
                rewards_batch = torch.cat([r for r in rewards], dim=0)
                next_states_batch = {
                    'input_ids': torch.cat([ns['input_ids'] for ns in next_states], dim=0),
                    'attention_mask': torch.cat([ns['attention_mask'] for ns in next_states], dim=0)
                }
                dones_batch = torch.cat([d for d in dones], dim=0)
                
                # Update critic (both Q-networks)
                with torch.no_grad():
                    # Select next action according to target policy with noise
                    noise = torch.normal(0, policy_noise, size=actions_batch.size()).to(device)
                    noise = torch.clamp(noise, -noise_clip, noise_clip)
                    
                    next_actions = target_actor(next_states_batch['input_ids'], next_states_batch['attention_mask'])
                    next_actions = torch.clamp(next_actions + noise, -1, 1)
                    
                    # Get minimum Q-value of both critics
                    target_q1, target_q2 = target_critic(
                        next_states_batch['input_ids'],
                        next_actions,
                        next_states_batch['attention_mask']
                    )
                    target_q = torch.min(target_q1, target_q2)
                    
                    # Compute target Q-value
                    target_q = rewards_batch + gamma * target_q * (1 - dones_batch)
                
                # Get current Q-values
                current_q1, current_q2 = critic(
                    states_batch['input_ids'],
                    actions_batch,
                    states_batch['attention_mask']
                )
                
                # Compute critic loss (MSE)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                
                # Update critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                total_critic_loss += critic_loss.item()
                
                # Delayed policy updates
                if total_steps % policy_freq == 0:
                    # Compute actor loss
                    actor_actions = actor(states_batch['input_ids'], states_batch['attention_mask'])
                    actor_loss = -critic.q1_forward(
                        states_batch['input_ids'],
                        actor_actions,
                        states_batch['attention_mask']
                    ).mean()
                    
                    # Update actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    
                    total_actor_loss += actor_loss.item()
                    
                    # Soft update target networks
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Actor Loss: {total_actor_loss/(i+1):.4f}, "
                           f"Critic Loss: {total_critic_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    # Save the fine-tuned model
    actor.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"TD3 training complete. Model saved to {output_dir}")
    return actor.base_model, tokenizer