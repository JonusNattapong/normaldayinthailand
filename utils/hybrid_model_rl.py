import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class WorldModel(nn.Module):
    """Model-based component that predicts next token probabilities and rewards"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Prediction head
        self.next_token_predictor = nn.Linear(self.hidden_size, base_model.config.vocab_size)
        self.reward_predictor = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Predict next tokens and rewards
        token_logits = self.next_token_predictor(hidden_states)
        reward_estimates = self.reward_predictor(hidden_states)
        
        return {
            'token_logits': token_logits,
            'reward_estimates': reward_estimates,
            'hidden_states': hidden_states
        }

class PolicyNetwork(nn.Module):
    """Model-free component that directly outputs policy"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Policy head
        self.policy_head = nn.Linear(self.hidden_size, base_model.config.vocab_size)
        
    def forward(self, input_ids, attention_mask=None, hidden_states=None):
        if hidden_states is None:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
        policy_logits = self.policy_head(hidden_states)
        return policy_logits

def imagine_trajectories(world_model, input_ids, attention_mask, tokenizer, n_steps=5, n_trajectories=10):
    """Use the world model to imagine possible future trajectories"""
    batch_size = input_ids.size(0)
    device = input_ids.device
    
    # Get initial hidden states
    with torch.no_grad():
        outputs = world_model.forward(input_ids, attention_mask)
        hidden_states = outputs['hidden_states'][:, -1, :].unsqueeze(1)  # Last token
    
    # Expand for multiple trajectories
    hidden_states = hidden_states.expand(batch_size, n_trajectories, -1).contiguous()
    hidden_states = hidden_states.view(batch_size * n_trajectories, 1, -1)
    
    # Generate tokens and track rewards
    all_rewards = torch.zeros(batch_size * n_trajectories, n_steps).to(device)
    current_input_ids = input_ids[:, -1:].expand(batch_size * n_trajectories, 1)
    
    for step in range(n_steps):
        # Predict next token distribution and sample
        with torch.no_grad():
            outputs = world_model(current_input_ids)
            token_logits = outputs['token_logits'][:, -1, :]
            reward_estimates = outputs['reward_estimates'][:, -1, 0]
            
            # Sample next token
            next_token_probs = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1)
            
            # Record predicted reward
            all_rewards[:, step] = reward_estimates
            
            # Update for next step
            current_input_ids = next_token
            hidden_states = outputs['hidden_states']
    
    # Calculate trajectory values
    trajectory_values = all_rewards.sum(dim=1)
    trajectory_values = trajectory_values.view(batch_size, n_trajectories)
    
    # Find best trajectory for each batch item
    best_trajectory_indices = trajectory_values.argmax(dim=1)
    
    return {
        'trajectory_values': trajectory_values,
        'best_indices': best_trajectory_indices
    }

def train_hybrid_model_rl(base_model_path, train_dataset, output_dir, reward_model_path=None,
                          batch_size=4, epochs=1, lr=1e-5, imagination_steps=3):
    """Train a hybrid model-based and model-free RL system."""
    logger.info("Initializing Hybrid Model-based/Model-free RL Training")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create world model and policy components
    world_model = WorldModel(base_model)
    policy_network = PolicyNetwork(base_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use world model for rewards")
        reward_model = None
        
    # Setup optimizers
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    world_model.train()
    policy_network.train()
    
    for epoch in range(epochs):
        total_world_loss = 0
        total_policy_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Phase 1: Train World Model
            world_outputs = world_model(inputs['input_ids'], inputs['attention_mask'])
            token_logits = world_outputs['token_logits'][:, :-1]  # Shift to predict next tokens
            
            # Calculate next token prediction loss
            targets = inputs['input_ids'][:, 1:]  # Target is next token
            token_prediction_loss = F.cross_entropy(
                token_logits.reshape(-1, tokenizer.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )
            
            # Calculate reward prediction loss if reward model available
            if reward_model:
                with torch.no_grad():
                    reward_targets = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
                    reward_targets = reward_targets.mean(dim=-1).unsqueeze(-1)
            else:
                # Simplified reward: prefer non-pad tokens
                reward_targets = (inputs['input_ids'] != tokenizer.pad_token_id).float().unsqueeze(-1)
            
            reward_prediction_loss = F.mse_loss(world_outputs['reward_estimates'], reward_targets)
            
            # Combined world model loss
            world_loss = token_prediction_loss + reward_prediction_loss
            
            # Update world model
            world_optimizer.zero_grad()
            world_loss.backward()
            world_optimizer.step()
            
            # Phase 2: Train Policy using world model simulations
            trajectory_data = imagine_trajectories(
                world_model,
                inputs['input_ids'],
                inputs['attention_mask'],
                tokenizer,
                n_steps=imagination_steps
            )
            
            # Get policy outputs
            policy_logits = policy_network(inputs['input_ids'], inputs['attention_mask'])
            
            # Use trajectory values to guide policy updates
            trajectory_values = trajectory_data['trajectory_values']
            advantages = trajectory_values - trajectory_values.mean(dim=1, keepdim=True)
            
            # Policy gradient loss
            policy_loss = 0
            for b in range(batch_size):
                best_idx = trajectory_data['best_indices'][b]
                if best_idx.item() > 0:  # Ensure we have a valid best trajectory
                    # Encourage policy to output distribution that would lead to high-value trajectory
                    target_logits = world_model(inputs['input_ids'][b:b+1]).token_logits
                    policy_loss += -advantages[b, best_idx] * F.kl_div(
                        F.log_softmax(policy_logits[b], dim=-1),
                        F.softmax(target_logits[b], dim=-1),
                        reduction='sum'
                    )
            
            if batch_size > 0:
                policy_loss /= batch_size
            
            # Update policy network
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            total_world_loss += world_loss.item()
            total_policy_loss += policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"World Loss: {world_loss.item():.4f}, Policy Loss: {policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss:.4f}")
    
        avg_world_loss = total_world_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average World Loss: {avg_world_loss:.4f}, "
                   f"Average Policy Loss: {avg_policy_loss:.4f}")
    
    # Save the models
    logger.info(f"Training completed. Saving models to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save world model
    world_model.base_model.save_pretrained(os.path.join(output_dir, "world_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "world_model"))
    
    # Save policy model (main output model)
    policy_network.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Return final model and tokenizer
    return policy_network.base_model, tokenizer
