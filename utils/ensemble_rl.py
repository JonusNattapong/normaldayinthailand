import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class EnsembleRLModel(nn.Module):
    """
    Ensemble Reinforcement Learning Model
    ผสมผสานหลายโมเดล RL เข้าด้วยกันเพื่อตัดสินใจร่วมกัน
    """
    def __init__(self, model_paths, model_types, vocab_size, device='cuda'):
        """
        Args:
            model_paths (list): รายการ path ของโมเดลแต่ละตัว
            model_types (list): ชนิดของแต่ละโมเดล (เช่น 'ppo', 'sac', 'dqn')
            vocab_size (int): ขนาดคำศัพท์
            device (str): อุปกรณ์ที่ใช้ในการประมวลผล
        """
        super(EnsembleRLModel, self).__init__()
        self.device = device
        self.models = nn.ModuleList()
        self.model_types = model_types
        self.vocab_size = vocab_size
        
        for path, model_type in zip(model_paths, model_types):
            try:
                model = AutoModelForCausalLM.from_pretrained(path).to(device)
                self.models.append(model)
                logger.info(f"Loaded {model_type} model from {path}")
            except Exception as e:
                logger.error(f"Error loading model {model_type} from {path}: {e}")
                raise e
        
        # Dynamic weight parameters for each model
        self.model_weights = nn.Parameter(torch.ones(len(model_paths)))
        
        # Context-dependent weight prediction network
        self.context_network = nn.Sequential(
            nn.Linear(768, 256),  # Assuming 768 for hidden size
            nn.ReLU(),
            nn.Linear(256, len(model_paths)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get the hidden state from the last token for context
        with torch.no_grad():
            # Using the first model to get context representation
            outputs = self.models[0](input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            context_features = hidden_states[:, -1, :]
        
        # Predict context-dependent weights
        context_weights = self.context_network(context_features)
        
        # Combine normalized base weights with context weights
        combined_weights = F.softmax(self.model_weights, dim=0) * context_weights
        
        # Get predictions from each model
        all_logits = []
        for model in self.models:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Last token logits
                all_logits.append(logits)
        
        # Stack and weight the logits
        stacked_logits = torch.stack(all_logits, dim=0)
        weighted_logits = torch.sum(stacked_logits * combined_weights.view(-1, 1, 1), dim=0)
        
        return weighted_logits
    
    def generate(self, input_ids, attention_mask=None, max_length=30, **kwargs):
        """
        Generate text using the ensemble model
        """
        # Start with the input context
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        
        for _ in range(max_length):
            # Get weighted predictions for next token
            next_token_logits = self.forward(current_input_ids, current_attention_mask)
            
            # Sample from the logits
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # Append to the sequence
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            if current_attention_mask is not None:
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones_like(next_token)], dim=1
                )
        
        return current_input_ids

def train_ensemble_rl(
    model_paths,
    model_types,
    dataset,
    output_dir,
    reward_model_path=None,
    batch_size=4,
    epochs=1,
    lr=1e-5
):
    """
    Train an Ensemble Reinforcement Learning model
    
    Args:
        model_paths (list): รายการ path ของโมเดลแต่ละตัว
        model_types (list): ชนิดของแต่ละโมเดล (เช่น 'ppo', 'sac', 'dqn')
        dataset: Dataset for training
        output_dir: Directory to save the model
        reward_model_path: Path to a pre-trained reward model (optional)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        lr: Learning rate
    
    Returns:
        Trained ensemble model
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer from the first model
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    vocab_size = tokenizer.vocab_size
    
    # Load reward model if provided
    reward_model = None
    if reward_model_path:
        try:
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path).to(device)
            logger.info(f"Loaded reward model from {reward_model_path}")
        except Exception as e:
            logger.warning(f"Could not load reward model: {e}")
    
    # Initialize ensemble model
    ensemble_model = EnsembleRLModel(model_paths, model_types, vocab_size, device).to(device)
    
    # Only train the weights and context network, not the base models
    optimizer = torch.optim.Adam([
        {'params': ensemble_model.model_weights}, 
        {'params': ensemble_model.context_network.parameters()}
    ], lr=lr)
    
    logger.info("Starting Ensemble RL training...")
    
    # Track best performance
    best_reward = float('-inf')
    
    # Training loop
    for epoch in range(epochs):
        total_reward = 0
        total_loss = 0
        
        for i, batch in enumerate(dataset):
            if i >= len(dataset) // batch_size:
                break
                
            # Process batch data
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate completions with ensemble model
            with torch.no_grad():
                generated_ids = ensemble_model.generate(
                    inputs.input_ids, 
                    attention_mask=inputs.attention_mask,
                    max_length=10
                )
                
                # Extract only the generated part
                generated_part = generated_ids[:, inputs.input_ids.shape[1]:]
            
            # Calculate reward
            if reward_model:
                with torch.no_grad():
                    reward_outputs = reward_model(generated_ids)
                    rewards = reward_outputs.logits[:, -1]
            else:
                # Simple heuristic reward based on probability
                with torch.no_grad():
                    # Calculate probability of generated sequence under ensemble model
                    outputs = ensemble_model(inputs.input_ids, attention_mask=inputs.attention_mask)
                    probs = F.softmax(outputs, dim=-1)
                    selected_probs = probs.gather(1, generated_part[:, 0].unsqueeze(1))
                    rewards = torch.log(selected_probs + 1e-10).squeeze()
            
            # Policy gradient loss (maximize reward)
            loss = -torch.mean(rewards)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_reward += rewards.mean().item()
            total_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(dataset)//batch_size}, "
                           f"Loss: {total_loss/(i+1):.4f}, "
                           f"Avg Reward: {total_reward/(i+1):.4f}")
    
    # Save the fine-tuned model
    ensemble_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Ensemble RL training complete. Model saved to {output_dir}")
    return ensemble_model, tokenizer