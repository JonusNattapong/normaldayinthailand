import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class HighLevelPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(HighLevelPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class LowLevelPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(LowLevelPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def train_hierarchical_rl(
    high_level_model_path,
    low_level_model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(high_level_model_path)
    vocab_size = tokenizer.vocab_size
    
    high_level_policy = HighLevelPolicy(high_level_model_path, vocab_size, device).to(device)
    low_level_policy = LowLevelPolicy(low_level_model_path, vocab_size, device).to(device)
    
    high_level_optimizer = torch.optim.Adam(high_level_policy.parameters(), lr=lr)
    low_level_optimizer = torch.optim.Adam(low_level_policy.parameters(), lr=lr)
    
    logger.info("Starting Hierarchical RL training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            high_level_logits = high_level_policy(inputs.input_ids, inputs.attention_mask)
            high_level_action = torch.argmax(high_level_logits, dim=-1)
            
            low_level_inputs = torch.cat([inputs.input_ids, high_level_action.unsqueeze(1)], dim=1)
            low_level_mask = torch.cat([inputs.attention_mask, torch.ones_like(high_level_action).unsqueeze(1)], dim=1)
            low_level_logits = low_level_policy(low_level_inputs, low_level_mask)
            low_level_action = torch.argmax(low_level_logits, dim=-1)
            
            rewards = F.softmax(low_level_logits, dim=-1).gather(1, low_level_action.unsqueeze(1))
            loss = -torch.mean(rewards)
            
            high_level_optimizer.zero_grad()
            low_level_optimizer.zero_grad()
            loss.backward()
            high_level_optimizer.step()
            low_level_optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    high_level_policy.save_pretrained(os.path.join(output_dir, "high_level"))
    low_level_policy.save_pretrained(os.path.join(output_dir, "low_level"))
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Hierarchical RL training complete. Model saved to {output_dir}")
    return high_level_policy, low_level_policy, tokenizer