import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class HERPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(HERPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def train_her_rl(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    her_k=4
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    policy = HERPolicy(model_path, vocab_size, device).to(device)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    logger.info("Starting Hindsight Experience Replay (HER) RL training...")
    
    replay_buffer = []

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
            
            replay_buffer.append((inputs.input_ids, inputs.attention_mask, action, rewards))
            
            if len(replay_buffer) > her_k:
                replay_buffer.pop(0)
            
            # HER: Replay with different goals
            for replay in replay_buffer:
                replay_inputs, replay_mask, replay_action, replay_reward = replay
                new_goal = torch.randint(0, vocab_size, replay_action.shape).to(device)
                new_inputs = torch.where(replay_inputs == replay_action, new_goal, replay_inputs)
                new_logits = policy(new_inputs, replay_mask)
                new_rewards = F.softmax(new_logits, dim=-1).gather(1, new_goal.unsqueeze(1))
                total_rewards = (replay_reward + new_rewards) / 2
                
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
    
    logger.info(f"Hindsight Experience Replay (HER) RL training complete. Model saved to {output_dir}")
    return policy, tokenizer