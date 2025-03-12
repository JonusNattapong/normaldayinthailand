import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import math

logger = logging.getLogger(__name__)

class TsallisEntropyPolicy(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Value function head
        self.value_head = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Get logits for policy
        logits = outputs.logits
        values = self.value_head(hidden_states)
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states
        }

def compute_tsallis_entropy(probs, q=2.0):
    """Compute Tsallis entropy with entropic index q"""
    if q == 1.0:
        # Special case: Shannon entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    else:
        # General case: Tsallis entropy
        entropy = (1 / (q - 1)) * (1 - torch.sum(probs**q, dim=-1))
    
    return entropy

def compute_tsallis_policy_gradient_loss(logits, values, rewards, q=2.0, gamma=0.99, entropy_coef=0.01):
    """Compute policy gradient loss with Tsallis entropy regularization"""
    # Get probabilities from logits
    probs = F.softmax(logits, dim=-1)
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute Tsallis entropy
    tsallis_entropy = compute_tsallis_entropy(probs, q)
    
    # Compute policy gradient loss
    policy_loss = -torch.mean(log_probs * rewards)
    
    # Apply Tsallis entropy regularization
    entropy_regularized_loss = policy_loss - entropy_coef * tsallis_entropy.mean()
    
    return entropy_regularized_loss, tsallis_entropy.mean()

def train_tsallis_entropy_rl(base_model_path, train_dataset, output_dir, reward_model_path=None,
                           batch_size=4, epochs=1, lr=1e-5, q=2.0, entropy_coef=0.01):
    """Train a policy with Tsallis entropy regularization."""
    logger.info(f"Initializing Tsallis Entropy RL training with q={q}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create policy model
    policy = TsallisEntropyPolicy(base_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use simple rewards")
        reward_model = None
        
    # Setup optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    policy.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_entropy = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Get policy outputs
            policy_outputs = policy(inputs['input_ids'], inputs['attention_mask'])
            logits = policy_outputs['logits']
            values = policy_outputs['values']
            
            # Generate next tokens using current policy
            with torch.no_grad():
                probs = F.softmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.multinomial(probs, 1)
                next_inputs = torch.cat([inputs['input_ids'], next_tokens], dim=1)
                
                # Get rewards
                if reward_model:
                    reward_outputs = reward_model(next_inputs)
                    rewards = reward_outputs.logits.mean(dim=-1)
                else:
                    # Simple language modeling reward - higher probability = better
                    next_token_probs = F.softmax(base_model(next_inputs).logits[:, -2, :], dim=-1)
                    next_token_indices = next_tokens
                    rewards = torch.gather(next_token_probs, 1, next_token_indices).squeeze(1)
                    rewards = rewards.detach()
            
            # Compute advantages (simplified)
            advantages = rewards.unsqueeze(1).expand_as(values) - values.detach()
            
            # Compute Tsallis policy gradient loss
            loss, entropy = compute_tsallis_policy_gradient_loss(
                logits,
                values,
                advantages,
                q=q,
                entropy_coef=entropy_coef
            )
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_entropy += entropy.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Loss: {loss.item():.4f}, Tsallis Entropy: {entropy.item():.4f}")
    
        avg_loss = total_loss / num_batches
        avg_entropy = total_entropy / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Loss: {avg_loss:.4f}, "
                   f"Average Tsallis Entropy: {avg_entropy:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    policy.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return policy.base_model, tokenizer
