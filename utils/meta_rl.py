import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class MetaRLPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(MetaRLPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def train_meta_rl(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    policy = MetaRLPolicy(model_path, vocab_size, device).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    logger.info("Starting Meta-Reinforcement Learning (Meta-RL) training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            logits = policy(inputs.input_ids, inputs.attention_mask)
            action = torch.argmax(logits, dim=-1)
            rewards = F.softmax(logits, dim=-1).gather(1, action.unsqueeze(1))
            
            loss = -torch.mean(rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Meta-learning: Adapt to new tasks
            for task in range(5):  # Assume 5 different tasks
                task_inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
                task_logits = policy(task_inputs.input_ids, task_inputs.attention_mask)
                task_action = torch.argmax(task_logits, dim=-1)
                task_rewards = F.softmax(task_logits, dim=-1).gather(1, task_action.unsqueeze(1))
                
                task_loss = -torch.mean(task_rewards)
                
                optimizer.zero_grad()
                task_loss.backward()
                optimizer.step()
                
                total_reward += task_rewards.mean().item()
                total_loss += task_loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Meta-Reinforcement Learning (Meta-RL) training complete. Model saved to {output_dir}")
    return policy, tokenizer