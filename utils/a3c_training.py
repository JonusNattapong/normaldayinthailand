import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class ActorCriticNetwork(nn.Module):
    def __init__(self, base_model_path, vocab_size):
        super(ActorCriticNetwork, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size // 2, vocab_size)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        last_token_hidden = hidden_states[:, -1, :]
        
        # Actor outputs probability distribution over vocabulary
        actor_output = self.actor(last_token_hidden)
        policy_logits = F.log_softmax(actor_output, dim=-1)
        
        # Critic outputs value estimate
        value = self.critic(last_token_hidden)
        
        return policy_logits, value

def train_a3c(
    model_path,
    dataset,
    output_dir,
    reward_model_path=None,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    gamma=0.99,
    entropy_weight=0.01,
    max_grad_norm=0.5
):
    """
    Train an A3C (Asynchronous Advantage Actor-Critic) model for language generation
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        reward_model_path: Path to a pre-trained reward model (optional)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
        gamma: Discount factor
        entropy_weight: Weight for entropy regularization
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load reward model if provided
    reward_model = None
    if reward_model_path:
        try:
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path).to(device)
            logger.info(f"Loaded reward model from {reward_model_path}")
        except Exception as e:
            logger.warning(f"Could not load reward model: {e}")
    
    # Create Actor-Critic network
    actor_critic = ActorCriticNetwork(model_path, tokenizer.vocab_size).to(device)
    
    # Set up optimizer
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
    
    logger.info("Starting A3C training...")
    
    # Training loop
    actor_critic.train()
    for epoch in range(epochs):
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            # Process batch data
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Forward pass
            policy_logits, values = actor_critic(inputs.input_ids, inputs.attention_mask)
            
            # Generate next token to get reward
            with torch.no_grad():
                next_token_logits = policy_logits.detach()
                next_token = torch.multinomial(torch.exp(next_token_logits), 1)
                
                # Get reward (from reward model or simple heuristic)
                if reward_model:
                    # Add generated token to input and get reward from reward model
                    extended_inputs = torch.cat([inputs.input_ids, next_token], dim=1)
                    reward_output = reward_model(extended_inputs)
                    rewards = reward_output.logits[:, -1].unsqueeze(-1)
                else:
                    # Simple heuristic reward based on token likelihood
                    rewards = F.softmax(policy_logits, dim=-1).gather(1, next_token)
            
            # Calculate advantage = R - V
            advantages = rewards - values
            
            # Actor loss: -log_prob * advantage
            selected_log_probs = policy_logits.gather(1, next_token)
            actor_loss = -selected_log_probs * advantages.detach()
            
            # Critic loss: MSE between value and reward
            critic_loss = F.mse_loss(values, rewards.detach())
            
            # Entropy regularization to encourage exploration
            entropy = -(torch.exp(policy_logits) * policy_logits).sum(dim=-1).mean()
            
            # Total loss
            loss = actor_loss.mean() + 0.5 * critic_loss - entropy_weight * entropy
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            optimizer.step()
            
            # Track metrics
            total_actor_loss += actor_loss.mean().item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Actor Loss: {total_actor_loss/(i+1):.4f}, "
                           f"Critic Loss: {total_critic_loss/(i+1):.4f}, "
                           f"Entropy: {total_entropy/(i+1):.4f}")
    
    # Save the fine-tuned model
    actor_critic.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"A3C training complete. Model saved to {output_dir}")
    return actor_critic.base_model, tokenizer