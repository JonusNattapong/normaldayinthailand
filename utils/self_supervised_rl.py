import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class SelfSupervisedRLModel(nn.Module):
    """
    Self-supervised Reinforcement Learning Model
    โมเดลสร้างเป้าหมายของตัวเองและให้รางวัลตัวเอง
    """
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(SelfSupervisedRLModel, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def generate(self, input_ids, attention_mask=None, max_length=30, **kwargs):
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        for _ in range(max_length):
            next_token_logits = self.forward(current_input_ids, current_attention_mask)
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=1)

        return current_input_ids

def train_self_supervised_rl(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5,
    intrinsic_reward_weight=0.1
):
    """
    Train a Self-supervised Reinforcement Learning model
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
        intrinsic_reward_weight: Weight for the intrinsic reward
    
    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    model = SelfSupervisedRLModel(model_path, vocab_size, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info("Starting Self-supervised RL training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            generated_ids = model.generate(inputs.input_ids, inputs.attention_mask, max_length=10)
            generated_part = generated_ids[:, inputs.input_ids.shape[1]:]
            
            # Calculate intrinsic rewards
            intrinsic_rewards = torch.sum(F.log_softmax(model(inputs.input_ids, inputs.attention_mask), dim=-1).gather(1, generated_part[:, 0].unsqueeze(1)), dim=1)
            
            # Policy gradient loss (maximize reward)
            loss = -torch.mean(intrinsic_rewards) * intrinsic_reward_weight
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += intrinsic_rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Self-supervised RL training complete. Model saved to {output_dir}")
    return model, tokenizer