import os
import torch
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SACEnvironment(gym.Env):
    def __init__(self, model, tokenizer, reward_model, prompts):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.prompts = prompts
        self.current_prompt_idx = 0
        
        # Define action and observation space
        # For language models, we'll use a simplified continuous action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        # Observation space is a simplified representation of the language state
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
    
    def reset(self):
        # Select a prompt
        self.current_prompt = self.prompts[self.current_prompt_idx]
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
        
        # Encode prompt
        encoded = self.tokenizer(self.current_prompt, return_tensors="pt").to(self.model.device)
        self.current_input_ids = encoded["input_ids"]
        
        # Return a simplified state representation
        return np.random.rand(128).astype(np.float32)
    
    def step(self, action):
        # Use action to influence generation parameters (simplified)
        temperature = 0.5 + (action[0] + 1) / 4  # Map [-1,1] to [0.5, 1]
        
        # Generate text based on action-influenced parameters
        with torch.no_grad():
            output = self.model.generate(
                self.current_input_ids,
                max_length=self.current_input_ids.shape[1] + 20,
                temperature=temperature,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(self.current_prompt):]
        
        # Get reward from reward model
        with torch.no_grad():
            inputs = self.tokenizer(generated_text, return_tensors="pt").to(self.reward_model.device)
            reward_score = self.reward_model(**inputs).logits.item()
        
        # Simplified next state
        next_state = np.random.rand(128).astype(np.float32)
        
        # Always terminate after one step (episodic)
        done = True
        info = {'generated_text': generated_text}
        
        return next_state, reward_score, done, info

def train_sac(dpo_model_path, train_dataset, output_dir, reward_model_path, batch_size):
    """
    ฝึกโมเดลด้วย Soft Actor-Critic
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    model = AutoModelForCausalLM.from_pretrained(dpo_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    reward_pipeline = pipeline("text-classification", model=reward_model_path, tokenizer=tokenizer, device=0)
    
    # Select prompts for training
    prompts = [example["prompt"] for example in train_dataset[:min(100, len(train_dataset))]]
    
    # Create SAC environment
    env = SACEnvironment(model, tokenizer, reward_pipeline.model, prompts)
    
    # Initialize SAC agent
    sac_agent = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=100,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
    )
    
    # Train SAC agent (limited steps for demonstration)
    total_timesteps = 200  # Adjust based on available computation time
    sac_agent.learn(total_timesteps=total_timesteps)
    
    # Save the trained agent
    sac_agent.save(os.path.join(output_dir, "sac_model"))
    
    return sac_agent