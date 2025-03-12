import os
import torch
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class TextGenerationEnv(gym.Env):
    def __init__(self, model, tokenizer, reward_model, prompts):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.prompts = prompts
        self.current_prompt_idx = 0
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(128,), dtype=np.float32)
        
    def reset(self):
        self.current_prompt = self.prompts[self.current_prompt_idx]
        self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
        
        encoded = self.tokenizer(self.current_prompt, return_tensors="pt").to(self.model.device)
        self.current_input_ids = encoded["input_ids"]
        
        return np.random.rand(128).astype(np.float32)
        
    def step(self, action):
        with torch.no_grad():
            output = self.model.generate(
                self.current_input_ids,
                max_length=self.current_input_ids.shape[1] + 20,
                temperature=0.7,
                do_sample=True
            )
            
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(self.current_prompt):]
        
        # Get reward from reward model
        with torch.no_grad():
            inputs = self.tokenizer(generated_text, return_tensors="pt").to(self.reward_model.device)
            reward_score = self.reward_model(**inputs).logits.item()
        
        observation = np.random.rand(128).astype(np.float32)
        done = True
        
        return observation, reward_score, done, {}

def train_irl(dpo_model_path, train_dataset, output_dir, reward_model_path, batch_size):
    """
    ฝึกโมเดลด้วย Inverse Reinforcement Learning
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    model = AutoModelForCausalLM.from_pretrained(dpo_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    reward_pipeline = pipeline("text-classification", model=reward_model_path, tokenizer=tokenizer, device=0)
    
    # Create prompts for IRL
    prompts = [example["prompt"] for example in train_dataset[:min(100, len(train_dataset))]]
    
    # Create IRL environment
    env = TextGenerationEnv(model, tokenizer, reward_pipeline.model, prompts)
    
    # Train with PPO as the IRL algorithm
    irl_model = PPO("MlpPolicy", env, verbose=1)
    irl_model.learn(total_timesteps=100)
    irl_model.save(os.path.join(output_dir, "irl_model"))
    
    return irl_model