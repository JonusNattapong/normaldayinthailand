import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TransfoXLLMHeadModel, TransfoXLConfig
import logging
import os

logger = logging.getLogger(__name__)

class RecurrentRLTransformerXL(nn.Module):
    """Transformer-XL model with recurrent state passing for RL"""
    def __init__(self, base_model, hidden_size=None):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size or base_model.config.hidden_size
        
        # Value head for RL
        self.value_head = nn.Linear(self.hidden_size, 1)
        
        # Memory management
        self.mems = None
        
    def reset_memory(self):
        """Reset recurrent memory"""
        self.mems = None
        
    def forward(self, input_ids, attention_mask=None):
        # Forward pass with memory
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mems=self.mems
        )
        
        # Update memory for next call
        self.mems = outputs.mems
        
        # Extract features and compute values
        hidden_states = outputs.last_hidden_state
        values = self.value_head(hidden_states)
        
        return {
            'logits': outputs.logits,
            'values': values,
            'hidden_states': hidden_states,
            'mems': self.mems
        }

def create_or_load_transformer_xl(base_model_path):
    """Create a Transformer-XL based on the base model architecture"""
    try:
        # Try to load as TransfoXL
        model = TransfoXLLMHeadModel.from_pretrained(base_model_path)
        logger.info(f"Loaded native TransfoXL model from {base_model_path}")
        return model
    except:
        # Create new TransfoXL with similar configuration to base model
        logger.info(f"Creating new TransfoXL based on {base_model_path}")
        
        # Load base model to extract config params
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_config = base_model.config
        
        # Create TransfoXL config
        config = TransfoXLConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=min(12, base_config.num_hidden_layers),  # TransfoXL may need fewer layers
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.hidden_size * 4,
            hidden_act=base_config.hidden_act,
            mem_len=512,  # Memory length
            clamp_len=400,  # Clamp length for positional embeddings
            adaptive=True,  # Use adaptive softmax
            div_val=4,  # Adaptive softmax divident value
        )
        
        # Create model
        model = TransfoXLLMHeadModel(config)
        
        return model

def train_transformer_xl_rl(base_model_path, train_dataset, output_dir, reward_model_path=None,
                          batch_size=4, epochs=1, lr=1e-5, context_length=256):
    """Train a Transformer-XL model with recurrent RL."""
    logger.info("Initializing Transformer-XL with Recurrent RL")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Create or load TransformerXL model
    base_xl_model = create_or_load_transformer_xl(base_model_path)
    
    # Create recurrent RL model
    model = RecurrentRLTransformerXL(base_xl_model)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use simple rewards")
        reward_model = None
        
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0
        
        # Sort dataset by length for efficient batching
        texts = [item['text'] for item in train_dataset]
        lengths = [len(t) for t in texts]
        sorted_indices = sorted(range(len(texts)), key=lambda i: lengths[i])
        
        for i in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[i:i+batch_size]
            batch = [train_dataset[idx] for idx in batch_indices]
            
            # Tokenize inputs
            inputs = tokenizer([item['text'] for item in batch], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_xl_model.device) for k, v in inputs.items()}
            
            # Process in context chunks to leverage memory
            # We'll split long sequences into chunks and process them sequentially
            chunk_size = context_length
            seq_len = inputs['input_ids'].size(1)
            
            # Reset memory at the start of each sequence
            model.reset_memory()
            
            policy_loss = 0
            value_loss = 0
            entropy = 0
            
            # Process sequence in chunks
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                # Get current chunk
                chunk_input_ids = inputs['input_ids'][:, chunk_start:chunk_end]
                chunk_attention_mask = inputs['attention_mask'][:, chunk_start:chunk_end] if 'attention_mask' in inputs else None
                
                # Forward pass (memory is automatically handled in the model)
                outputs = model(chunk_input_ids, chunk_attention_mask)
                logits = outputs['logits']
                values = outputs['values']
                
                # For targets, shift input_ids one position right
                if chunk_end < seq_len:
                    targets = inputs['input_ids'][:, chunk_start+1:chunk_end+1]
                else:
                    # For the last chunk, we'll use EOS tokens as targets for the last position
                    last_targets = torch.full((chunk_input_ids.size(0), 1), 
                                           tokenizer.eos_token_id, 
                                           device=chunk_input_ids.device)
                    if chunk_start > 0:
                        targets = torch.cat([inputs['input_ids'][:, chunk_start+1:seq_len], last_targets], dim=1)
                    else:
                        targets = last_targets
                
                # Calculate rewards (simplified)
                if reward_model:
                    with torch.no_grad():
                        reward_outputs = reward_model(chunk_input_ids)
                        rewards = reward_outputs.logits.mean(dim=-1).unsqueeze(-1)
                else:
                    # Use language modeling accuracy as proxy reward
                    probs = F.softmax(logits, dim=-1)
                    target_probs = torch.gather(
                        probs[:, :-1], 
                        2, 
                        targets[:, :min(chunk_end-chunk_start, targets.size(1))].unsqueeze(-1)
                    )
                    rewards = torch.log(target_probs + 1e-10).detach()
                
                # Policy loss: standard negative log likelihood
                log_probs = F.log_softmax(logits, dim=-1)
                action_log_probs = torch.gather(
                    log_probs[:, :-1], 
                    2, 
                    targets[:, :min(chunk_end-chunk_start, targets.size(1))].unsqueeze(-1)
                ).squeeze(-1)
                
                # Value targets (using rewards)
                value_targets = rewards
                
                # Calculate losses
                chunk_policy_loss = -action_log_probs.mean()
                chunk_value_loss = F.mse_loss(values[:, :-1], value_targets)
                
                # Entropy bonus
                chunk_entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
                
                # Combined loss
                entropy_coef = 0.01
                value_coef = 0.5
                chunk_loss = chunk_policy_loss + value_coef * chunk_value_loss - entropy_coef * chunk_entropy
                
                # Accumulate losses
                policy_loss += chunk_policy_loss.item()
                value_loss += chunk_value_loss.item()
                entropy += chunk_entropy.item()
                
                # Backward pass for this chunk
                optimizer.zero_grad()
                chunk_loss.backward()
                optimizer.step()
            
            # Average losses over chunks
            num_chunks = (seq_len + chunk_size - 1) // chunk_size  # Ceiling division
            policy_loss /= num_chunks
            value_loss /= num_chunks
            entropy /= num_chunks
            
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            total_entropy += entropy
            num_batches += 1
            
            if num_batches % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Policy Loss: {policy_loss:.4f}, "
                           f"Value Loss: {value_loss:.4f}, "
                           f"Entropy: {entropy:.4f}")
        
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Policy Loss: {avg_policy_loss:.4f}, "
                   f"Average Value Loss: {avg_value_loss:.4f}, "
                   f"Average Entropy: {avg_entropy:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # For TransformerXL models, we save both the base model and the RL components
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save additional RL components state dict
    torch.save({
        'value_head': model.value_head.state_dict(),
    }, os.path.join(output_dir, "rl_components.pt"))
    
    return model.base_model, tokenizer
