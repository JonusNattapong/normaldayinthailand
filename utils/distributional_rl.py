import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class CategoricalDQN(nn.Module):
    def __init__(self, base_model, n_atoms=51, vmin=-10, vmax=10):
        super().__init__()
        self.base_model = base_model
        self.n_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.supports = torch.linspace(vmin, vmax, n_atoms).to(self.base_model.device)
        self.delta = (vmax - vmin) / (n_atoms - 1)
        
        # Distribution head
        hidden_size = self.base_model.config.hidden_size
        self.value_dist = nn.Linear(hidden_size, n_atoms)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, -1, :]
        logits = self.value_dist(hidden_states)
        return F.softmax(logits, dim=-1)  # Return probability distribution
    
    def get_value(self, input_ids, attention_mask=None):
        dist = self(input_ids, attention_mask)
        return torch.sum(dist * self.supports.expand_as(dist), dim=1)

def project_distribution(target_support, target_dist, support, n_atoms, vmin, vmax, delta):
    """Projects the categorical distribution onto a new support."""
    batch_size = target_dist.size(0)
    
    # Clipping projection
    proj_support = torch.clamp(target_support, vmin, vmax)
    
    # Compute projection
    tz_j = (proj_support - vmin) / delta
    tz_j_floor = tz_j.floor().long()
    tz_j_ceil = tz_j.ceil().long()
    
    # Handle corner cases
    tz_j_floor = torch.clamp(tz_j_floor, 0, n_atoms - 1)
    tz_j_ceil = torch.clamp(tz_j_ceil, 0, n_atoms - 1)
    
    # Compute weights
    ceil_weight = tz_j - tz_j_floor.float()
    floor_weight = 1.0 - ceil_weight
    
    # Distribute probability
    proj_dist = torch.zeros_like(target_dist)
    
    for b in range(batch_size):
        for i in range(n_atoms):
            floor_idx, ceil_idx = tz_j_floor[b][i], tz_j_ceil[b][i]
            proj_dist[b][floor_idx] += target_dist[b][i] * floor_weight[b][i]
            proj_dist[b][ceil_idx] += target_dist[b][i] * ceil_weight[b][i]
            
    return proj_dist

def train_distributional_rl(base_model_path, train_dataset, output_dir, reward_model_path=None, 
                           batch_size=4, epochs=1, lr=1e-5, n_atoms=51, vmin=-10, vmax=10):
    """Train a distributional RL model for language generation."""
    logger.info("Initializing Distributional RL training with Categorical DQN")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create distributional RL model
    model = CategoricalDQN(base_model, n_atoms, vmin, vmax)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use base model for rewards")
        reward_model = base_model
        
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    support = torch.linspace(vmin, vmax, n_atoms).to(base_model.device)
    delta = (vmax - vmin) / (n_atoms - 1)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Get current distribution
            current_dist = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Generate next tokens
            with torch.no_grad():
                outputs = base_model.generate(
                    inputs['input_ids'],
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    attention_mask=inputs['attention_mask']
                )
                
            # Get rewards
            with torch.no_grad():
                rewards = reward_model(outputs, attention_mask=torch.ones_like(outputs)).logits.mean(dim=1)
                
            # Calculate target distribution
            target_support = support.unsqueeze(0).expand(batch_size, -1) + rewards.unsqueeze(1)
            target_dist = project_distribution(
                target_support, 
                current_dist.detach(), 
                support,
                n_atoms, 
                vmin, 
                vmax, 
                delta
            )
            
            # Compute KL divergence loss
            log_probs = F.log_softmax(model.value_dist(base_model(inputs['input_ids']).last_hidden_state[:, -1, :]), dim=1)
            loss = -(target_dist * log_probs).sum(dim=1).mean()
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer
