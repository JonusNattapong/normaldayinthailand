import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from collections import deque

logger = logging.getLogger(__name__)

class DQNetwork(nn.Module):
    def __init__(self, base_model_path, vocab_size, hidden_dim=768):
        super(DQNetwork, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        # Q-network head
        self.q_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        # Output Q-values for each action (token)
        q_values = self.q_head(last_token_hidden)
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.memory)

def train_dqn(
    model_path,
    dataset,
    output_dir,
    reward_model_path=None,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    gamma=0.99,  # Discount factor
    epsilon_start=1.0,  # Exploration rate start value
    epsilon_end=0.1,  # Exploration rate end value
    epsilon_decay=0.995,  # Exploration rate decay
    target_update=10,  # How often to update target network
    buffer_size=10000
):
    """
    Train a DQN (Deep Q-Network) model for language generation
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        reward_model_path: Path to a pre-trained reward model (optional)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
        gamma: Discount factor
        epsilon_start: Starting value of epsilon for epsilon-greedy exploration
        epsilon_end: Minimum value of epsilon
        epsilon_decay: Rate at which epsilon decays
        target_update: How many steps between target network updates
        buffer_size: Size of replay buffer
    
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
    policy_net = DQNetwork(model_path, vocab_size).to(device)
    target_net = DQNetwork(model_path, vocab_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is not trained directly
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)
    
    # Initialize epsilon (for exploration)
    epsilon = epsilon_start
    
    logger.info("Starting DQN training...")
    
    steps_done = 0
    
    # Training loop
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            # Process batch data
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # State = current input sequence
            state = {
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }
            
            # Select action using epsilon-greedy policy
            with torch.no_grad():
                # Get Q-values for all possible actions
                q_values = policy_net(inputs.input_ids, inputs.attention_mask)
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    # Explore: choose random action
                    next_token = torch.randint(0, vocab_size, (inputs.input_ids.shape[0], 1)).to(device)
                else:
                    # Exploit: choose best action
                    next_token = q_values.max(1)[1].unsqueeze(1)
            
            # Get reward (from reward model or simple heuristic)
            with torch.no_grad():
                if reward_model:
                    # Add generated token to input and get reward from reward model
                    extended_inputs = torch.cat([inputs.input_ids, next_token], dim=1)
                    extended_attention_mask = torch.cat([inputs.attention_mask, torch.ones_like(next_token)], dim=1)
                    reward_output = reward_model(extended_inputs, attention_mask=extended_attention_mask)
                    reward = reward_output.logits[:, -1].unsqueeze(-1)
                else:
                    # Simple heuristic reward - use logits from policy network
                    selected_q_values = q_values.gather(1, next_token)
                    reward = torch.tanh(selected_q_values / 10)  # Scale and bound rewards
            
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
            for j in range(batch_size):
                replay_buffer.push(
                    {k: v[j:j+1] for k, v in state.items()},
                    next_token[j:j+1],
                    reward[j:j+1],
                    {k: v[j:j+1] for k, v in next_state.items()},
                    done[j:j+1]
                )
            
            total_reward += reward.mean().item()
            steps_done += 1
            
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
                
                # Compute current Q values
                current_q = policy_net(states_batch['input_ids'], states_batch['attention_mask'])
                current_q_values = current_q.gather(1, actions_batch)
                
                # Compute target Q values
                with torch.no_grad():
                    next_q_values = target_net(next_states_batch['input_ids'], next_states_batch['attention_mask'])
                    # DQN uses max Q-value for next state
                    max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                    target_q_values = rewards_batch + gamma * max_next_q_values * (1 - dones_batch)
                
                # Compute loss
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to stabilize training
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update target network
                if steps_done % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}, "
                           f"Epsilon: {epsilon:.4f}")
    
    # Save the fine-tuned model
    policy_net.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"DQN training complete. Model saved to {output_dir}")
    return policy_net.base_model, tokenizer