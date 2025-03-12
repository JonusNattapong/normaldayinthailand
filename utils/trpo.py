import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributions import Categorical

logger = logging.getLogger(__name__)

class TRPOPolicy(nn.Module):
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(TRPOPolicy, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def get_action(self, input_ids, attention_mask=None):
        logits = self.forward(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

def train_trpo(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    max_kl=1e-2,
    cg_iters=10,
    cg_damping=1e-2
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    policy = TRPOPolicy(model_path, vocab_size, device).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    logger.info("Starting TRPO training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            actions, log_probs, entropies = policy.get_action(inputs.input_ids, inputs.attention_mask)
            rewards = F.softmax(policy(inputs.input_ids, inputs.attention_mask), dim=-1).gather(1, actions.unsqueeze(1))
            
            loss = -torch.mean(rewards * log_probs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"TRPO training complete. Model saved to {output_dir}")
    return policy, tokenizer