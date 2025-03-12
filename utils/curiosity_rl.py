import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class CuriosityPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(CuriosityPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size
        self.curiosity_module = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        curiosity_reward = self.curiosity_module(outputs.hidden_states[-1][:, -1, :])
        return outputs.logits, curiosity_reward

def train_curiosity_rl(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    curiosity_weight=0.1
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    policy = CuriosityPolicy(model_path, vocab_size, device).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    logger.info("Starting Curiosity-driven RL training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            logits, curiosity_rewards = policy(inputs.input_ids, inputs.attention_mask)
            action = torch.argmax(logits, dim=-1)
            rewards = F.softmax(logits, dim=-1).gather(1, action.unsqueeze(1))
            total_rewards = rewards + curiosity_weight * curiosity_rewards
            
            loss = -torch.mean(total_rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += total_rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Curiosity-driven RL training complete. Model saved to {output_dir}")
    return policy, tokenizer