import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VTraceModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Value function head
        self.value_head = nn.Linear(self.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Get logits for policy and values
        logits = outputs.logits
        values = self.value_head(hidden_states)
        
        return {
            'logits': logits,
            'values': values,
            'hidden_states': hidden_states
        }

def compute_vtrace_advantages(values, next_values, rewards, behavior_logits, target_logits, 
                            gamma=0.99, lambda_=0.95, rho_clipping=1.0, c_clipping=1.0):
    """Compute V-trace advantages to correct for off-policy data."""
    
    # Compute importance weights
    behavior_probs = F.softmax(behavior_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)
    # Use clipped importance sampling ratio
    rho = torch.clamp(target_probs / (behavior_probs + 1e-10), 0, rho_clipping)
    c = torch.clamp(target_probs / (behavior_probs + 1e-10), 0, c_clipping)
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values - values
    
    # Compute v-trace targets
    vtrace_targets = values + rho * td_errors
    
    # Compute GAE-style advantages
    advantages = torch.zeros_like(vtrace_targets)
    last_gae = 0
    
    # Reverse accumulate advantages
    for t in reversed(range(len(advantages))):
        if t == len(advantages) - 1:
            next_value = next_values[-1]
        else:
            next_value = vtrace_targets[t+1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lambda_ * c[t] * last_gae
        advantages[t] = last_gae
        
    return advantages, vtrace_targets

def train_off_policy_correction(base_model_path, train_dataset, output_dir, behavior_model_path=None,
                              reward_model_path=None, batch_size=4, epochs=1, lr=1e-5):
    """Train a model using off-policy correction with V-trace."""
    logger.info("Initializing Off-Policy Correction with V-trace")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create V-trace model
    model = VTraceModel(base_model)
    
    # Load behavior model if provided, otherwise use a copy of base model
    if behavior_model_path:
        logger.info(f"Loading behavior model from {behavior_model_path}")
        behavior_model = AutoModelForCausalLM.from_pretrained(behavior_model_path)
    else:
        logger.info("No behavior model provided, will use a copy of base model")
        behavior_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
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
    behavior_model.eval()  # Behavior model is fixed
    
    for epoch in range(epochs):
        total_actor_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Get behavior policy logits (frozen)
            with torch.no_grad():
                behavior_outputs = behavior_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                behavior_logits = behavior_outputs.logits
            
            # Get target policy outputs
            target_outputs = model(inputs['input_ids'], inputs['attention_mask'])
            target_logits = target_outputs['logits']
            values = target_outputs['values']
            
            # Compute rewards
            if reward_model:
                with torch.no_grad():
                    reward_outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                    rewards = reward_outputs.logits.mean(dim=-1).unsqueeze(-1)
            else:
                # Simple reward function based on next token prediction
                seq_len = inputs['input_ids'].size(1)
                if seq_len > 1:
                    next_tokens = inputs['input_ids'][:, 1:]
                    pred_logits = target_logits[:, :-1, :]
                    
                    # Get probability of correct next token
                    next_token_probs = F.softmax(pred_logits, dim=-1)
                    next_token_indices = next_tokens.unsqueeze(-1)
                    correct_probs = torch.gather(next_token_probs, 2, next_token_indices).squeeze(-1)
                    
                    # Reward is log probability of correct next token
                    rewards = torch.log(correct_probs + 1e-10)
                    rewards = rewards.unsqueeze(-1)
                else:
                    rewards = torch.zeros((inputs['input_ids'].size(0), 1, 1), device=inputs['input_ids'].device)
            
            # Get next state values
            with torch.no_grad():
                next_inputs = torch.cat([inputs['input_ids'][:, 1:], 
                                       torch.ones((inputs['input_ids'].size(0), 1), 
                                                 dtype=torch.long, 
                                                 device=inputs['input_ids'].device) * tokenizer.eos_token_id], dim=1)
                next_mask = torch.cat([inputs['attention_mask'][:, 1:], 
                                     torch.ones((inputs['attention_mask'].size(0), 1), 
                                               device=inputs['attention_mask'].device)], dim=1)
                next_outputs = model(next_inputs, next_mask)
                next_values = next_outputs['values']
            
            # Compute V-trace advantages and targets
            advantages, vtrace_targets = compute_vtrace_advantages(
                values[:, :-1].detach(),
                next_values[:, :-1].detach(),
                rewards[:, :-1].detach(),
                behavior_logits[:, :-1].detach(),
                target_logits[:, :-1].detach()
            )
            
            # Compute policy loss (actor loss)
            log_probs = F.log_softmax(target_logits[:, :-1], dim=-1)
            action_log_probs = torch.gather(
                log_probs, 
                2, 
                inputs['input_ids'][:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Actor loss uses advantages from V-trace
            actor_loss = -(action_log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values[:, :-1], vtrace_targets.detach())
            
            # Entropy bonus
            entropy = -(F.softmax(target_logits[:, :-1], dim=-1) * F.log_softmax(target_logits[:, :-1], dim=-1)).sum(-1).mean()
            
            # Combined loss
            entropy_coef = 0.01
            value_coef = 0.5
            loss = actor_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Actor Loss: {actor_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                           f"Entropy: {entropy.item():.4f}")
    
        avg_actor_loss = total_actor_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Actor Loss: {avg_actor_loss:.4f}, "
                   f"Average Value Loss: {avg_value_loss:.4f}, "
                   f"Average Entropy: {avg_entropy:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model.base_model, tokenizer
