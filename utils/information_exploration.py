import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class InformationExplorationModel(nn.Module):
    """Model that uses information-theoretic measures for exploration"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Value head
        self.value_head = nn.Linear(self.hidden_size, 1)
        
        # Uncertainty estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = outputs.logits
        
        # Compute values
        values = self.value_head(hidden_states)
        
        # Compute uncertainty estimates
        uncertainty = self.uncertainty_head(hidden_states)
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states,
            'uncertainty': uncertainty
        }
    
    def compute_entropy_bonus(self, logits):
        """Compute entropy of token distribution as exploration bonus"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

def compute_information_gain(model, tokenizer, input_ids, num_samples=10):
    """Estimate information gain of different actions"""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    vocab_size = model.base_model.config.vocab_size
    
    # Get current hidden states
    with torch.no_grad():
        outputs = model.base_model(input_ids)
        current_hidden = outputs.last_hidden_state[:, -1, :]  # Last token
    
    # Sample possible next tokens
    with torch.no_grad():
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # Sample top-k tokens for efficiency
        top_k_values, top_k_indices = torch.topk(probs, k=min(num_samples, vocab_size), dim=-1)
    
    # Compute expected information gain for each token
    info_gains = []
    
    for batch_idx in range(batch_size):
        token_gains = []
        
        for i in range(top_k_indices.size(1)):
            token = top_k_indices[batch_idx, i].unsqueeze(0).unsqueeze(0)
            
            # Create next input with this token
            next_input = torch.cat([input_ids[batch_idx:batch_idx+1], token], dim=1)
            
            # Get prediction for next step
            with torch.no_grad():
                next_outputs = model.base_model(next_input)
                next_hidden = next_outputs.last_hidden_state[:, -1, :]
            
            # Compute KL divergence as information gain
            # Simplified: using L2 distance between hidden states as proxy
            info_gain = torch.sum((next_hidden - current_hidden[batch_idx:batch_idx+1]) ** 2)
            
            # Weight by probability
            weighted_gain = top_k_values[batch_idx, i].item() * info_gain.item()
            token_gains.append((token.item(), weighted_gain))
        
        # Sort by information gain
        token_gains.sort(key=lambda x: x[1], reverse=True)
        info_gains.append(token_gains)
    
    return info_gains

def train_information_exploration(base_model_path, train_dataset, output_dir, reward_model_path=None,
                                batch_size=4, epochs=1, lr=1e-5, explore_coef=0.1):
    """Train a model with information-theoretic exploration."""
    logger.info("Initializing Information-theoretic Exploration")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create model
    model = InformationExplorationModel(base_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use entropy-based rewards")
        reward_model = None
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_bonus = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            logits = outputs['logits']
            values = outputs['values']
            uncertainty = outputs['uncertainty']
            
            # Generate next tokens based on information gain
            info_gains = compute_information_gain(model, tokenizer, inputs['input_ids'])
            
            # Create targets based on information gain
            # We'll use the tokens with highest information gain as targets
            targets = []
            for batch_idx in range(len(info_gains)):
                if info_gains[batch_idx]:
                    targets.append(info_gains[batch_idx][0][0])  # First token in sorted list
                else:
                    # Fallback if no info gain computed
                    targets.append(tokenizer.eos_token_id)
            
            targets = torch.tensor(targets, device=base_model.device).unsqueeze(1)
            
            # Compute entropy bonus
            entropy_bonus = model.compute_entropy_bonus(logits[:, -1, :])
            
            # Compute intrinsic rewards based on information gain and uncertainty
            with torch.no_grad():
                intrinsic_rewards = uncertainty[:, -1, 0] + explore_coef * entropy_bonus
            
            # Get extrinsic rewards if reward model available
            if reward_model:
                with torch.no_grad():
                    reward_outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    extrinsic_rewards = reward_outputs.logits.mean(dim=-1)
                    
                    # Combine rewards
                    combined_rewards = extrinsic_rewards + explore_coef * intrinsic_rewards
            else:
                # Use only intrinsic rewards
                combined_rewards = intrinsic_rewards
            
            # Policy loss: maximize reward by picking high information gain tokens
            policy_logits = logits[:, -1, :]
            policy_loss = F.cross_entropy(policy_logits, targets.squeeze())
            
            # Value loss: predict combined rewards
            value_loss = F.mse_loss(values[:, -1, 0], combined_rewards)
            
            # Combined loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_bonus += entropy_bonus.mean().item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Policy Loss: {policy_loss.item():.4f}, "
                           f"Value Loss: {value_loss.item():.4f}, "
                           f"Entropy Bonus: {entropy_bonus.mean().item():.4f}")
        
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy_bonus = total_entropy_bonus / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Policy Loss: {avg_policy_loss:.4f}, "
                   f"Average Value Loss: {avg_value_loss:.4f}, "
                   f"Average Entropy Bonus: {avg_entropy_bonus:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also save the exploration-specific components
    torch.save({
        'value_head': model.value_head.state_dict(),
        'uncertainty_head': model.uncertainty_head.state_dict()
    }, f"{output_dir}/exploration_components.pt")
    
    return model.base_model, tokenizer
