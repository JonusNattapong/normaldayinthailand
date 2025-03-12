import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

class RewardNetwork(nn.Module):
    """Reward network that learns to distinguish expert data from generated data"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Discriminator head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Get reward prediction for each token
        rewards = self.reward_head(hidden_states)
        
        return rewards
    
    def get_sequence_reward(self, input_ids, attention_mask=None):
        """Get the total reward for a sequence"""
        token_rewards = self(input_ids, attention_mask)
        # Sum rewards across sequence
        return token_rewards.sum(dim=1)

class GeneratorNetwork(nn.Module):
    """Generator network that creates text to fool the discriminator"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, input_ids, attention_mask=None):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.base_model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **kwargs
        )

def train_adversarial_irl(base_model_path, train_dataset, output_dir, expert_dataset=None,
                         batch_size=4, epochs=1, lr=1e-5, disc_steps=5, gen_steps=1):
    """Train a model using Adversarial Inverse Reinforcement Learning."""
    logger.info("Initializing Adversarial Inverse Reinforcement Learning")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create reward and generator networks
    reward_network = RewardNetwork(AutoModelForCausalLM.from_pretrained(base_model_path))
    generator = GeneratorNetwork(base_model)
    
    # If no expert dataset provided, use a portion of train dataset
    if expert_dataset is None:
        expert_size = len(train_dataset) // 4
        expert_indices = torch.randperm(len(train_dataset))[:expert_size]
        expert_dataset = torch.utils.data.Subset(train_dataset, expert_indices)
        logger.info(f"Created expert dataset with {len(expert_dataset)} examples")
    
    # Setup optimizers
    reward_optimizer = torch.optim.Adam(reward_network.parameters(), lr=lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        total_reward_loss = 0
        total_generator_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            # Get batch
            train_batch = train_dataset[i:i+batch_size]
            
            # Get expert batch
            expert_idx = torch.randperm(len(expert_dataset))[:batch_size]
            expert_batch = torch.utils.data.Subset(expert_dataset, expert_idx)
            
            # Tokenize inputs
            train_inputs = tokenizer(train_batch['text'], return_tensors="pt", padding=True, truncation=True)
            train_inputs = {k: v.to(base_model.device) for k, v in train_inputs.items()}
            
            expert_inputs = tokenizer([item['text'] for item in expert_batch], return_tensors="pt", padding=True, truncation=True)
            expert_inputs = {k: v.to(base_model.device) for k, v in expert_inputs.items()}
            
            # Step 1: Train discriminator/reward network
            for _ in range(disc_steps):
                # Generate samples from current policy
                with torch.no_grad():
                    generated_outputs = generator.generate(
                        train_inputs['input_ids'],
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        attention_mask=train_inputs['attention_mask']
                    )
                
                # Get discriminator predictions
                generated_rewards = reward_network.get_sequence_reward(generated_outputs)
                expert_rewards = reward_network.get_sequence_reward(expert_inputs['input_ids'], expert_inputs['attention_mask'])
                
                # Discriminator loss: expert should get high reward, generated should get low reward
                # Using binary cross-entropy loss
                expert_labels = torch.ones_like(expert_rewards)
                generated_labels = torch.zeros_like(generated_rewards)
                
                expert_loss = F.binary_cross_entropy_with_logits(expert_rewards, expert_labels)
                generated_loss = F.binary_cross_entropy_with_logits(generated_rewards, generated_labels)
                
                reward_loss = expert_loss + generated_loss
                
                # Update reward network
                reward_optimizer.zero_grad()
                reward_loss.backward()
                reward_optimizer.step()
                
                total_reward_loss += reward_loss.item()
            
            # Step 2: Train generator using rewards
            for _ in range(gen_steps):
                # Generate trajectories with gradient tracking
                generator_outputs = generator(train_inputs['input_ids'], train_inputs['attention_mask'])
                generator_logits = generator_outputs.logits
                
                # Sample from generator
                probs = F.softmax(generator_logits[:, -1, :], dim=-1)
                actions = torch.multinomial(probs, 1)
                
                # Prepare next tokens for generator
                next_tokens = torch.cat([train_inputs['input_ids'], actions], dim=1)
                
                # Get rewards for generated tokens
                with torch.no_grad():
                    rewards = reward_network.get_sequence_reward(next_tokens)
                
                # Compute policy gradient loss
                log_probs = F.log_softmax(generator_logits[:, -1, :], dim=-1)
                selected_log_probs = torch.gather(log_probs, 1, actions)
                
                # Policy gradient: maximize rewards
                generator_loss = -(selected_log_probs * rewards).mean()
                
                # Update generator
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()
                
                total_generator_loss += generator_loss.item()
            
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Reward Loss: {reward_loss.item():.4f}, "
                           f"Generator Loss: {generator_loss.item():.4f}")
        
        avg_reward_loss = total_reward_loss / (num_batches * disc_steps)
        avg_generator_loss = total_generator_loss / (num_batches * gen_steps)
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Reward Loss: {avg_reward_loss:.4f}, "
                   f"Average Generator Loss: {avg_generator_loss:.4f}")
    
    # Save the models
    logger.info(f"Training completed. Saving models to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save generator (main model)
    generator.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save reward network
    reward_network.base_model.save_pretrained(os.path.join(output_dir, "reward_network"))
    
    return generator.base_model, tokenizer
