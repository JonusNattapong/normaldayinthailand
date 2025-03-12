import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class LatentDynamicsModel(nn.Module):
    """Model that predicts transitions in latent space"""
    def __init__(self, latent_dim, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Forward dynamics model: predicts next latent state given current state and action
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim),  # latent_state + action embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Reward predictor based on latent state
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, latent_state, action_embedding):
        """Predict next latent state and expected reward"""
        combined = torch.cat([latent_state, action_embedding], dim=1)
        next_latent = self.dynamics(combined)
        predicted_reward = self.reward_predictor(next_latent)
        
        return next_latent, predicted_reward

class VAEEncoder(nn.Module):
    """Encoder for a VAE to map text to latent space"""
    def __init__(self, base_model, latent_dim=64):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.latent_dim = latent_dim
        
        # Project to latent space
        self.mean = nn.Linear(self.hidden_size, latent_dim)
        self.logvar = nn.Linear(self.hidden_size, latent_dim)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]  # Take last token representation
        
        # Get mean and logvar for latent space
        mean = self.mean(hidden_states)
        logvar = self.logvar(hidden_states)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar

class VAEDecoder(nn.Module):
    """Decoder for a VAE to reconstruct text from latent space"""
    def __init__(self, base_model, latent_dim=64):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.latent_dim = latent_dim
        
        # Project from latent space to hidden state
        self.latent_to_hidden = nn.Linear(latent_dim, self.hidden_size)
    
    def forward(self, z, input_ids=None):
        # Convert latent vector to hidden state
        hidden_state = self.latent_to_hidden(z)
        
        # Feed hidden state to base model's language modeling head
        logits = self.base_model.lm_head(hidden_state.unsqueeze(1))
        
        return logits

class WorldModelController(nn.Module):
    """Controller that uses latent planning to generate text"""
    def __init__(self, latent_dim, vocab_size, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Policy network that maps latent state to action probabilities
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, latent_state):
        action_logits = self.policy(latent_state)
        return action_logits

def train_world_models(base_model_path, train_dataset, output_dir, reward_model_path=None,
                     batch_size=4, epochs=1, lr=1e-4, latent_dim=64, planning_horizon=5):
    """Train a world model for text generation."""
    logger.info("Initializing World Models training for text generation")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create world model components
    encoder = VAEEncoder(base_model, latent_dim)
    decoder = VAEDecoder(base_model, latent_dim)
    dynamics_model = LatentDynamicsModel(latent_dim)
    controller = WorldModelController(latent_dim, tokenizer.vocab_size)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use simple rewards")
        reward_model = None
    
    # Setup optimizers
    vae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    dynamics_optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        total_vae_loss = 0
        total_dynamics_loss = 0
        total_controller_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Phase 1: Train VAE
            # Encode to latent space
            z, mean, logvar = encoder(inputs['input_ids'], inputs['attention_mask'])
            
            # Decode from latent space
            reconstructed_logits = decoder(z)
            
            # Reconstruction loss
            recon_loss = F.cross_entropy(
                reconstructed_logits.reshape(-1, tokenizer.vocab_size),
                inputs['input_ids'].reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
            
            # Total VAE loss
            vae_loss = recon_loss + 0.1 * kl_loss
            
            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()
            
            # Phase 2: Train dynamics model
            # For consecutive sequences in input
            if inputs['input_ids'].size(1) > 1:
                # Get latent states for all tokens
                with torch.no_grad():
                    states = []
                    for t in range(inputs['input_ids'].size(1) - 1):
                        z_t, _, _ = encoder(inputs['input_ids'][:, :t+1], 
                                          inputs['attention_mask'][:, :t+1] if inputs['attention_mask'] is not None else None)
                        states.append(z_t)
                    
                    # Get action embeddings (next token embeddings)
                    action_embeddings = base_model.get_input_embeddings()(inputs['input_ids'][:, 1:])
                    # Average embeddings over hidden dim for simplicity
                    action_embeddings = action_embeddings.mean(dim=2)
                    
                # Predict next latent states
                dynamics_loss = 0
                reward_loss = 0
                
                for t in range(len(states) - 1):
                    # Use dynamics model to predict next latent state
                    pred_next_state, pred_reward = dynamics_model(
                        states[t], 
                        states[t]  # Using latent state as action embedding for simplicity
                    )
                    
                    # Get actual next state
                    actual_next_state = states[t + 1].detach()
                    
                    # Compute prediction loss
                    dynamics_loss += F.mse_loss(pred_next_state, actual_next_state)
                    
                    # Compute reward if reward model available
                    if reward_model:
                        with torch.no_grad():
                            # Get reward for current sequence
                            reward_outputs = reward_model(inputs['input_ids'][:, :t+2])
                            actual_reward = reward_outputs.logits[:, -1, :].mean(dim=-1)
                        
                        reward_loss += F.mse_loss(pred_reward.squeeze(), actual_reward)
                
                # Update dynamics model
                total_dynamics_loss = dynamics_loss + reward_loss
                
                dynamics_optimizer.zero_grad()
                total_dynamics_loss.backward()
                dynamics_optimizer.step()
            
            # Phase 3: Train controller with planning
            # Do latent planning and compute controller loss
            controller_loss = 0
            
            # Encode current state
            with torch.no_grad():
                current_state, _, _ = encoder(inputs['input_ids'], inputs['attention_mask'])
            
            # Rollout trajectories in latent space
            best_actions = []
            best_values = []
            
            for b in range(batch_size):
                # Planning: simulate multiple trajectories and find best one
                best_value = float('-inf')
                best_action_idx = 0
                
                # Try different first actions
                num_actions_to_try = min(10, tokenizer.vocab_size)  # Limit for efficiency
                for a in range(num_actions_to_try):
                    # Create action embedding
                    action_embedding = torch.zeros(latent_dim).to(base_model.device)
                    action_embedding[a % latent_dim] = 1.0  # Simple encoding
                    
                    # Simulate trajectory
                    sim_state = current_state[b].clone()
                    cumulative_reward = 0
                    
                    for step in range(planning_horizon):
                        # Predict next state and reward
                        next_state, reward = dynamics_model(
                            sim_state.unsqueeze(0), 
                            action_embedding.unsqueeze(0)
                        )
                        
                        cumulative_reward += reward.item()
                        sim_state = next_state.squeeze(0)
                    
                    # Update best action if this trajectory is better
                    if cumulative_reward > best_value:
                        best_value = cumulative_reward
                        best_action_idx = a
                
                best_actions.append(best_action_idx)
                best_values.append(best_value)
            
            # Convert to tensor
            best_actions = torch.tensor(best_actions, device=base_model.device)
            
            # Get controller outputs
            controller_logits = controller(current_state)
            
            # Controller loss: make actions with highest value more likely
            controller_loss = F.cross_entropy(controller_logits, best_actions)
            
            controller_optimizer.zero_grad()
            controller_loss.backward()
            controller_optimizer.step()
            
            total_vae_loss += vae_loss.item()
            total_kl_loss += kl_loss.item()
            if isinstance(total_dynamics_loss, torch.Tensor):
                total_dynamics_loss += total_dynamics_loss.item()
            total_controller_loss += controller_loss.item()
            num_batches += 1
            
            if num_batches % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"VAE Loss: {vae_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, "
                           f"Dynamics Loss: {total_dynamics_loss if isinstance(total_dynamics_loss, float) else total_dynamics_loss.item():.4f}, "
                           f"Controller Loss: {controller_loss.item():.4f}")
    
        avg_vae_loss = total_vae_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_dynamics_loss = total_dynamics_loss / num_batches if isinstance(total_dynamics_loss, float) else total_dynamics_loss.item() / num_batches
        avg_controller_loss = total_controller_loss / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average VAE Loss: {avg_vae_loss:.4f}, "
                   f"Average KL Loss: {avg_kl_loss:.4f}, "
                   f"Average Dynamics Loss: {avg_dynamics_loss:.4f}, "
                   f"Average Controller Loss: {avg_controller_loss:.4f}")
    
    # Save all components
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base model (with VAE components)
    encoder.base_model.save_pretrained(os.path.join(output_dir, "encoder"))
    tokenizer.save_pretrained(os.path.join(output_dir, "encoder"))
    
    # Save controller (main model for generation)
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save additional component states
    torch.save({
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'dynamics_model_state': dynamics_model.state_dict(),
        'controller_state': controller.state_dict(),
        'latent_dim': latent_dim
    }, os.path.join(output_dir, "world_model_components.pt"))
    
    return base_model, tokenizer
