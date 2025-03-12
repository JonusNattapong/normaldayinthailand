import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

class DiffusionTextModel(nn.Module):
    """Text diffusion model that iteratively denoises text"""
    def __init__(self, base_model, num_diffusion_steps=20):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.num_steps = num_diffusion_steps
        
        # Noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Value function for RL guidance
        self.value_head = nn.Linear(self.hidden_size, 1)
        
    def add_noise(self, embeddings, noise_level):
        """Add noise to embeddings based on noise level (0=clean, 1=pure noise)"""
        noise = torch.randn_like(embeddings)
        noised_embeddings = (1 - noise_level) * embeddings + noise_level * noise
        return noised_embeddings, noise
    
    def denoise_step(self, noisy_embeddings, noise_level):
        """Predict and remove noise for one diffusion step"""
        # Prepare time embedding (noise level)
        batch_size, seq_len, emb_dim = noisy_embeddings.shape
        t_emb = torch.ones((batch_size, seq_len, 1), device=noisy_embeddings.device) * noise_level
        
        # Concatenate with noisy embeddings
        model_input = torch.cat([noisy_embeddings, t_emb], dim=-1)
        
        # Predict noise
        predicted_noise = self.noise_predictor(model_input)
        
        # Compute denoised embeddings
        alpha = 1 - noise_level
        denoised = (noisy_embeddings - noise_level * predicted_noise) / alpha
        
        return denoised
    
    def forward(self, input_ids, attention_mask=None, noise_level=0.5):
        # Get token embeddings
        token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # Add noise
        noisy_embeddings, noise = self.add_noise(token_embeddings, noise_level)
        
        # Run base model on noisy embeddings
        outputs = self.base_model(inputs_embeds=noisy_embeddings, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Predict value
        values = self.value_head(hidden_states)
        
        # Predict noise
        batch_size, seq_len, emb_dim = noisy_embeddings.shape
        t_emb = torch.ones((batch_size, seq_len, 1), device=noisy_embeddings.device) * noise_level
        model_input = torch.cat([noisy_embeddings, t_emb], dim=-1)
        predicted_noise = self.noise_predictor(model_input)
        
        return {
            'hidden_states': hidden_states,
            'values': values,
            'predicted_noise': predicted_noise,
            'actual_noise': noise
        }
    
    def generate(self, input_ids, attention_mask=None, guidance_scale=7.5, reward_model=None):
        """Generate text using diffusion with classifier-free guidance"""
        # Get embeddings for initial context
        token_embeddings = self.base_model.get_input_embeddings()(input_ids)
        batch_size, seq_len, emb_dim = token_embeddings.shape
        
        # Start from pure noise for continuation
        continuation_length = 20  # Generate 20 new tokens
        noise = torch.randn((batch_size, continuation_length, emb_dim), device=token_embeddings.device)
        
        # Initialize embeddings with context followed by noise
        embeddings = torch.cat([token_embeddings, noise], dim=1)
        full_attention_mask = None
        if attention_mask is not None:
            # Extend attention mask for generated tokens
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, continuation_length), device=attention_mask.device)
            ], dim=1)
        
        # Gradually denoise
        for t in reversed(range(self.num_steps)):
            noise_level = t / self.num_steps
            
            # Split embeddings into context (fixed) and continuation (being denoised)
            context_emb = embeddings[:, :seq_len, :]
            continuation_emb = embeddings[:, seq_len:, :]
            
            # Unconditional denoising (classifier-free guidance)
            with torch.no_grad():
                # Use empty context for unconditional denoising
                empty_context = torch.zeros_like(context_emb)
                uncond_embeddings = torch.cat([empty_context, continuation_emb], dim=1)
                
                uncond_outputs = self.base_model(
                    inputs_embeds=uncond_embeddings,
                    attention_mask=full_attention_mask
                )
                uncond_hidden = uncond_outputs.last_hidden_state[:, seq_len:, :]
                
                # Get unconditional noise prediction
                t_emb = torch.ones((batch_size, continuation_length, 1), device=continuation_emb.device) * noise_level
                uncond_input = torch.cat([continuation_emb, t_emb], dim=-1)
                uncond_noise_pred = self.noise_predictor(uncond_input)
            
            # Conditional denoising
            outputs = self.base_model(
                inputs_embeds=embeddings,
                attention_mask=full_attention_mask
            )
            hidden_states = outputs.last_hidden_state[:, seq_len:, :]
            
            # Get conditional noise prediction
            t_emb = torch.ones((batch_size, continuation_length, 1), device=continuation_emb.device) * noise_level
            cond_input = torch.cat([continuation_emb, t_emb], dim=-1)
            cond_noise_pred = self.noise_predictor(cond_input)
            
            # RL guidance if reward model is provided
            if reward_model is not None:
                with torch.no_grad():
                    # Predict next token embeddings after denoising
                    alpha = 1 - noise_level
                    denoised = (continuation_emb - noise_level * cond_noise_pred) / alpha
                    
                    # Map back to tokens and get reward
                    # This is a simplification - in practice, you'd need a better way to map embeddings to tokens
                    logits = self.base_model.lm_head(hidden_states)
                    next_tokens = torch.argmax(logits, dim=-1)
                    reward_scores = reward_model(next_tokens).logits.mean(dim=-1)
                    
                    # Scale noise prediction based on reward
                    reward_scaling = torch.sigmoid(reward_scores).unsqueeze(-1).unsqueeze(-1)
                    cond_noise_pred = cond_noise_pred * reward_scaling
            
            # Apply classifier-free guidance
            noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
            
            # Update continuation embeddings
            alpha = 1 - noise_level
            denoised_continuation = (continuation_emb - noise_level * noise_pred) / alpha
            
            # Add some noise for the next step (except for the last step)
            if t > 0:
                next_noise_level = (t - 1) / self.num_steps
                noise = torch.randn_like(denoised_continuation)
                continuation_emb = (1 - next_noise_level) * denoised_continuation + next_noise_level * noise
            else:
                continuation_emb = denoised_continuation
            
            # Update full embeddings
            embeddings = torch.cat([context_emb, continuation_emb], dim=1)
        
        # Convert final embeddings to tokens
        # Project embeddings to logits and select most likely tokens
        final_outputs = self.base_model(inputs_embeds=embeddings, attention_mask=full_attention_mask)
        final_logits = final_outputs.logits[:, seq_len-1:]  # Include last context token for better continuity
        
        # Get token IDs
        generated_token_ids = torch.argmax(final_logits, dim=-1)
        
        # Combine with input_ids
        full_sequence = torch.cat([input_ids, generated_token_ids[:, 1:]], dim=1)  # Skip first token as it's duplicated
        
        return full_sequence

def train_diffusion_rl(base_model_path, train_dataset, output_dir, reward_model_path=None,
                     batch_size=4, epochs=1, lr=1e-5, diffusion_steps=10):
    """Train a diffusion model with RL for text generation."""
    logger.info("Initializing Diffusion Model + RL")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Create diffusion model
    model = DiffusionTextModel(base_model, num_diffusion_steps=diffusion_steps)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, will use base model for reward estimation")
        reward_model = base_model
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        total_diff_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i:i+batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            
            # Sample different noise levels
            noise_level = torch.rand(1, device=base_model.device).item()
            
            # Forward pass
            outputs = model(inputs['input_ids'], inputs['attention_mask'], noise_level=noise_level)
            predicted_noise = outputs['predicted_noise']
            actual_noise = outputs['actual_noise']
            values = outputs['values']
            
            # Compute diffusion loss (noise prediction)
            diff_loss = F.mse_loss(predicted_noise, actual_noise)
            
            # Get rewards for value function training
            with torch.no_grad():
                reward_outputs = reward_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                rewards = reward_outputs.logits.mean(dim=-1, keepdim=True)
            
            # Value loss
            value_loss = F.mse_loss(values, rewards.expand_as(values))
            
            # Combined loss
            loss = diff_loss + 0.5 * value_loss
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_diff_loss += diff_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {num_batches}, "
                           f"Diffusion Loss: {diff_loss.item():.4f}, "
                           f"Value Loss: {value_loss.item():.4f}")
        
        avg_diff_loss = total_diff_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        
        logger.info(f"Epoch {epoch+1} completed. "
                   f"Average Diffusion Loss: {avg_diff_loss:.4f}, "
                   f"Average Value Loss: {avg_value_loss:.4f}")
        
        # Generate samples after each epoch
        with torch.no_grad():
            sample_input = inputs['input_ids'][:2, :10]  # Use first 2 examples, first 10 tokens
            sample_attn = inputs['attention_mask'][:2, :10] if 'attention_mask' in inputs else None
            
            generated = model.generate(
                sample_input, 
                attention_mask=sample_attn,
                guidance_scale=7.5,
                reward_model=reward_model
            )
            
            for idx in range(generated.size(0)):
                text = tokenizer.decode(generated[idx], skip_special_tokens=True)
                logger.info(f"Sample {idx+1}: {text}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save base model
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save diffusion-specific components
    torch.save({
        'noise_predictor': model.noise_predictor.state_dict(),
        'value_head': model.value_head.state_dict(),
        'num_diffusion_steps': model.num_steps
    }, os.path.join(output_dir, "diffusion_components.pt"))
    
    return model.base_model, tokenizer
