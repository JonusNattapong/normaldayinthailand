import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class TaskDecompositionRLModel(nn.Module):
    """
    Meta-RL with Task Decomposition
    แบ่งงานใหญ่เป็นงานย่อยหลายๆ ชิ้น
    """
    def __init__(self, model_path, vocab_size, device='cuda'):
        super(TaskDecompositionRLModel, self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.vocab_size = vocab_size

        # Task decomposition network
        self.task_network = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]
        
        # Apply task decomposition network
        task_features = self.task_network(hidden_states[:, -1, :])
        
        # Final logits
        logits = F.linear(task_features, self.model.lm_head.weight)
        
        return logits

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

def train_meta_rl_task_decomposition(
    model_path,
    dataset,
    output_dir,
    batch_size=4,
    epochs=1,
    lr=1e-5
):
    """
    Train a Meta-RL with Task Decomposition model
    
    Args:
        model_path: Path to the pre-trained model
        dataset: Dataset for training
        output_dir: Directory to save the model
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
    
    Returns:
        Trained model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tokenizer.vocab_size
    
    model = TaskDecompositionRLModel(model_path, vocab_size, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    logger.info("Starting Meta-RL with Task Decomposition training...")
    
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            generated_ids = model.generate(inputs.input_ids, inputs.attention_mask, max_length=10)
            generated_part = generated_ids[:, inputs.input_ids.shape[1]:]
            
            # Calculate rewards
            rewards = F.softmax(model(inputs.input_ids, inputs.attention_mask), dim=-1).gather(1, generated_part[:, 0].unsqueeze(1))
            
            # Policy gradient loss (maximize reward)
            loss = -torch.mean(rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                            f"Loss: {total_loss/(i+1):.4f}, "
                            f"Avg Reward: {total_reward/(i+1):.4f}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Meta-RL with Task Decomposition training complete. Model saved to {output_dir}")
    return model, tokenizer