import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

class AWRModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Value function head
        self.value_head = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = outputs.logits
        values = self.value_head(hidden_states)
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states
        }

def compute_advantages(values, rewards, gamma=0.99, lam=0.95):
    """Compute generalized advantage estimates"""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Reverse iterate through time
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # For last timestep, next value is 0
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * last_gae
        last_gae = advantages[t]
        
    return advantages

def train_awr(base_model_path, train_dataset, output_dir, reward_model_path=None,
            batch_size=4, epochs=1, lr=1e-5, beta=1.0, max_weight=20.0):
    """Train a policy using Advantage-Weighted Regression."""
    logger.info("Initializing Advantage Weighted Regression (AWR) Training")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create AWR model
    model = AWRModel(base_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use simple rewards")
        reward_model = None
        
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Generate samples with behavior policy (using base model)
            with torch.no_grad():
                outputs = base_model.generate(
                    inputs['input_ids'],
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    output_scores=True,
                    return_dict_in_generate=True,
                    attention_mask=inputs['attention_mask']
                )
                
                # Get generated sequences
                sequences = outputs.sequences
                
                # Get scores (log probs) from behavior policy
                behavior_log_probs = []
                for step_scores in outputs.scores:
                    behavior_log_probs.append(F.log_softmax(step_scores, dim=-1))
                behavior_log_probs = torch.stack(behavior_log_probs, dim=1)
            
            # Get rewards for generated sequences
            if reward_model:
                with torch.no_grad():
                    reward_outputs = reward_model(sequences)
                    rewards = reward_outputs.logits.mean(dim=-1)
            else:
                # Simple reward function: prefer diversity and avoid repetition
                rewards = torch.zeros(sequences.size(0), device=sequences.device)
                
                for b in range(sequences.size(0)):
                    # Count unique tokens as a diversity measure
                    unique_tokens = torch.unique(sequences[b]).size(0)
                    rewards[b] = unique_tokens / sequences.size(1)
            
            # Get values and target policy log probs
            model_outputs = model(sequences)
            values = model_outputs['values'].squeeze(-1)
            target_logits = model_outputs['logits']
            
            # Compute advantages
            advantages = compute_advantages(values.detach(), rewards.unsqueeze(1).expand_as(values).detach())
            
            # Compute exponential advantage weights, clipped to prevent extremely large weights
            weights = torch.exp(advantages / beta)
            weights = torch.clamp(weights, 0, max_weight)
            
            # AWR update: weighted supervised learning
            # For each position in the sequence with a generated token:
            policy_loss = 0
            for t in range(sequences.size(1) - 1):
                # Target is the next token
                target_tokens = sequences[:, t+1]
                
                # Predicted distribution for current token
                logits = target_logits[:, t, :]
                
                # Compute cross-entropy loss weighted by advantages
                token_loss = F.cross_entropy(
                    logits, 
                    target_tokens,
                    reduction='none'
                )
                
                # Apply weights from advantages
                weighted_loss = (token_loss * weights[:, t]).mean()
                policy_loss += weighted_loss
            
            # Value function loss
            value_loss = F.mse_loss(values, rewards.unsqueeze(1).expand_as(values))
            
            # Combined loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Policy Loss: {policy_loss.item():.4f}, "
                           f"Value Loss: {value_loss.item():.4f}")
    
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Policy Loss: {avg_policy_loss:.4f}, "
                   f"Average Value Loss: {avg_value_loss:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model.base_model, tokenizer
