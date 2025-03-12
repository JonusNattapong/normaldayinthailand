import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EmpowermentModel(nn.Module):
    """Model that computes empowerment-based intrinsic motivation"""
    def __init__(self, base_model, hidden_size=128):
        super().__init__()
        self.base_model = base_model
        self.model_hidden_size = base_model.config.hidden_size
        self.hidden_size = hidden_size
        
        # Forward model: predicts next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(self.model_hidden_size + self.model_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.model_hidden_size)
        )
        
        # Inverse model: predicts action given current and next state
        self.inverse_model = nn.Sequential(
            nn.Linear(self.model_hidden_size + self.model_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, base_model.config.vocab_size)
        )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = outputs.logits
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }
    
    def compute_empowerment(self, current_state, actions, next_states):
        """Compute empowerment as mutual information between actions and next states"""
        batch_size = current_state.size(0)
        num_actions = actions.size(1)
        
        # Expand current state for all actions
        expanded_current = current_state.unsqueeze(1).expand(-1, num_actions, -1)
        expanded_current = expanded_current.reshape(batch_size * num_actions, -1)
        
        # Reshape actions and next states
        flattened_actions = actions.reshape(batch_size * num_actions, -1)
        flattened_next_states = next_states.reshape(batch_size * num_actions, -1)
        
        # Forward prediction: p(next_state | current_state, action)
        predicted_next_states = self.forward_model(
            torch.cat([expanded_current, flattened_actions], dim=1)
        )
        forward_loss = F.mse_loss(predicted_next_states, flattened_next_states)
        
        # Inverse prediction: p(action | current_state, next_state)
        predicted_actions = self.inverse_model(
            torch.cat([expanded_current, flattened_next_states], dim=1)
        )
        inverse_loss = F.cross_entropy(
            predicted_actions,
            flattened_actions.argmax(dim=1) if flattened_actions.dim() > 1 else flattened_actions
        )
        
        # Empowerment is approximated by the negative of the inverse model loss
        # The better the inverse model can predict the action, the higher the empowerment
        empowerment = -inverse_loss
        
        return empowerment, forward_loss, inverse_loss

def train_empowerment(base_model_path, train_dataset, output_dir, reward_model_path=None,
                    batch_size=4, epochs=1, lr=1e-5, empowerment_coef=0.01):
    """Train a model with intrinsic motivation through empowerment."""
    logger.info("Initializing Intrinsic Motivation through Empowerment")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create empowerment model
    model = EmpowermentModel(base_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use only intrinsic rewards")
        reward_model = None
        
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_policy_loss = 0
        total_empowerment = 0
        total_forward_loss = 0
        total_inverse_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            logits = outputs['logits']
            hidden_states = outputs['hidden_states']
            
            # Generate next tokens
            with torch.no_grad():
                next_token_probs = F.softmax(logits[:, -1, :], dim=-1)
                next_tokens = torch.multinomial(next_token_probs, 5)  # Sample 5 possible next tokens
                
                # Get embeddings for actions
                action_embeddings = base_model.get_input_embeddings()(next_tokens)
                
                # Generate next states for each action
                next_states = []
                for j in range(next_tokens.size(1)):
                    next_input = torch.cat([inputs['input_ids'], next_tokens[:, j:j+1]], dim=1)
                    next_output = model(next_input)
                    next_state = next_output['hidden_states'][:, -1, :].unsqueeze(1)
                    next_states.append(next_state)
                
                next_states = torch.cat(next_states, dim=1)
            
            # Compute empowerment and model losses
            empowerment, forward_loss, inverse_loss = model.compute_empowerment(
                hidden_states[:, -1, :],
                action_embeddings,
                next_states
            )
            
            # Get extrinsic rewards if reward model available
            if reward_model:
                with torch.no_grad():
                    next_input = torch.cat([inputs['input_ids'], next_tokens[:, 0:1]], dim=1)  # Use first sampled token
                    reward_outputs = reward_model(next_input)
                    extrinsic_rewards = reward_outputs.logits.mean(dim=-1)
            else:
                extrinsic_rewards = torch.zeros(batch_size, device=base_model.device)
            
            # Combined rewards: extrinsic + empowerment-based intrinsic
            combined_rewards = extrinsic_rewards + empowerment_coef * empowerment.detach()
            
            # Policy loss: encourage actions that maximize combined reward
            policy_logits = logits[:, -1, :]
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # Use the rewards to guide policy improvement
            policy_loss = -(policy_log_probs * combined_rewards.unsqueeze(1)).mean()
            
            # Total loss
            loss = policy_loss + forward_loss + inverse_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_empowerment += empowerment.mean().item()
            total_forward_loss += forward_loss.item()
            total_inverse_loss += inverse_loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Policy Loss: {policy_loss.item():.4f}, "
                           f"Empowerment: {empowerment.mean().item():.4f}, "
                           f"Forward Loss: {forward_loss.item():.4f}, "
                           f"Inverse Loss: {inverse_loss.item():.4f}")
    
        avg_policy_loss = total_policy_loss / num_batches
        avg_empowerment = total_empowerment / num_batches
        avg_forward_loss = total_forward_loss / num_batches
        avg_inverse_loss = total_inverse_loss / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Policy Loss: {avg_policy_loss:.4f}, "
                   f"Average Empowerment: {avg_empowerment:.4f}, "
                   f"Average Forward Loss: {avg_forward_loss:.4f}, "
                   f"Average Inverse Loss: {avg_inverse_loss:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model.base_model, tokenizer
