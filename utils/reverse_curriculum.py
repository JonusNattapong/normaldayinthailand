import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import numpy as np
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CurriculumGenerator:
    """Generates a sequence of increasingly difficult tasks"""
    def __init__(self, base_model, tokenizer, reward_model=None, num_samples=100):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.num_samples = num_samples
        self.device = base_model.device
        
    def generate_curriculum(self, goal_texts, num_tasks=10, max_length=50):
        """
        Generate curriculum starting from the goal and working backwards
        
        Args:
            goal_texts: List of target texts (final state)
            num_tasks: Number of tasks in curriculum
            max_length: Maximum length of generated sequences
            
        Returns:
            List of curriculum tasks, from easiest to hardest
        """
        logger.info(f"Generating curriculum with {num_tasks} tasks")
        
        all_tasks = []
        current_tasks = goal_texts
        
        # Store goal as hardest task
        all_tasks.append(current_tasks)
        
        # Work backwards to create easier tasks
        for step in range(num_tasks - 1):
            logger.info(f"Generating task {num_tasks - step - 1}/{num_tasks}")
            
            easier_tasks = []
            for task in tqdm(current_tasks):
                # Tokenize the task
                inputs = self.tokenizer(task, return_tensors="pt").to(self.device)
                
                # Find key points where we can simplify
                # For simplicity, we'll just trim the text progressively
                # In a more sophisticated implementation, we might use the model itself
                
                # Get the length of the current task
                task_length = len(task.split())
                
                # Simplify by keeping a smaller fraction of the text
                keep_ratio = 0.7  # Keep 70% of the text
                simplified_length = max(1, int(task_length * keep_ratio))
                
                # Get the simplified text (first part of the task)
                simplified_text = " ".join(task.split()[:simplified_length])
                
                # Add some randomness to avoid all tasks being simple prefixes
                if random.random() < 0.3 and simplified_length > 2:
                    # Sometimes replace specific details with generic placeholders
                    words = simplified_text.split()
                    for i in range(len(words)):
                        if random.random() < 0.2:
                            words[i] = "[...]"
                    simplified_text = " ".join(words)
                
                easier_tasks.append(simplified_text)
            
            # Add this set of easier tasks to our curriculum
            all_tasks.append(easier_tasks)
            current_tasks = easier_tasks
            
        # Reverse so we start with the easiest tasks
        all_tasks.reverse()
        
        return all_tasks

def evaluate_curriculum(model, tokenizer, curriculum_tasks, reward_model=None):
    """Evaluate how well the model performs on each level of the curriculum"""
    results = []
    
    for level, tasks in enumerate(curriculum_tasks):
        # Sample a few tasks from this level
        sample_size = min(5, len(tasks))
        sampled_tasks = random.sample(tasks, sample_size)
        
        level_rewards = []
        for task in sampled_tasks:
            # Generate response
            inputs = tokenizer(task, return_tensors="pt").to(model.device)
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Compute reward
            if reward_model:
                with torch.no_grad():
                    reward_outputs = reward_model(outputs)
                    reward = reward_outputs.logits.mean().item()
            else:
                # Simple proxy for reward: length of generation
                reward = len(generated_text.split()) / 50.0  # Normalize
                
            level_rewards.append(reward)
        
        avg_reward = sum(level_rewards) / len(level_rewards)
        results.append({
            'level': level,
            'average_reward': avg_reward,
            'num_tasks': len(tasks)
        })
        
    return results

def train_reverse_curriculum(base_model_path, train_dataset, output_dir, reward_model_path=None,
                           batch_size=4, epochs=1, lr=1e-5, curriculum_steps=5):
    """Train a model using reverse curriculum learning."""
    logger.info("Initializing Reverse Curriculum Generation")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    # Load reward model if provided
    if reward_model_path:
        logger.info(f"Loading reward model from {reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
    else:
        logger.info("No reward model provided, using simple rewards")
        reward_model = None
    
    # Create curriculum generator
    curriculum_gen = CurriculumGenerator(base_model, tokenizer, reward_model)
    
    # Generate curriculum from the training data
    # For curriculum learning, we'll use the full texts as goals
    goal_texts = [item['text'] for item in train_dataset[:100]]  # Use a subset for efficiency
    
    logger.info("Generating curriculum...")
    curriculum = curriculum_gen.generate_curriculum(goal_texts, num_tasks=curriculum_steps)
    
    logger.info(f"Generated curriculum with {len(curriculum)} levels")
    for i, level in enumerate(curriculum):
        logger.info(f"Level {i}: {len(level)} tasks, Example: {level[0][:50]}...")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    
    # Training loop - train progressively on each curriculum level
    logger.info(f"Starting curriculum training for {epochs} epochs")
    base_model.train()
    
    for level, tasks in enumerate(curriculum):
        logger.info(f"Training on curriculum level {level}/{len(curriculum)-1}")
        
        # Create a dataset from this level's tasks
        level_texts = tasks
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Process tasks in batches
            for i in range(0, len(level_texts), batch_size):
                batch_texts = level_texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = base_model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Level {level}, Epoch {epoch+1}, Batch {num_batches}, "
                               f"Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            logger.info(f"Level {level}, Epoch {epoch+1} completed. "
                       f"Average Loss: {avg_loss:.4f}")
        
        # Evaluate on current level
        logger.info(f"Evaluating on curriculum level {level}")
        level_results = evaluate_curriculum(base_model, tokenizer, [curriculum[level]], reward_model)
        logger.info(f"Level {level} average reward: {level_results[0]['average_reward']:.4f}")
    
    # Final evaluation on all curriculum levels
    logger.info("Final evaluation on all curriculum levels")
    final_results = evaluate_curriculum(base_model, tokenizer, curriculum, reward_model)
    
    for result in final_results:
        logger.info(f"Level {result['level']} final average reward: {result['average_reward']:.4f}")
    
    # Save the model
    logger.info(f"Training completed. Saving model to {output_dir}")
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return base_model, tokenizer
