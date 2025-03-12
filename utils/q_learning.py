import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class QNetworkForText(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def train_q_learning(dpo_model_path, train_dataset, output_dir, reward_model_path, batch_size):
    """
    ฝึกโมเดลด้วย Q-Learning
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    model = AutoModelForCausalLM.from_pretrained(dpo_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    reward_pipeline = pipeline("text-classification", model=reward_model_path, tokenizer=tokenizer, device=0)
    
    # Create Q-Network
    state_dim = 128
    action_dim = 10
    q_network = QNetworkForText(state_dim, action_dim).to(model.device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
    
    # Q-learning parameters
    epsilon = 0.1
    gamma = 0.99
    episodes = 10
    
    # Create replay buffer (simplified)
    replay_buffer = []
    
    # Training loop
    for episode in range(episodes):
        # Select sample from training data
        sample_idx = np.random.randint(len(train_dataset))
        prompt = train_dataset[sample_idx]["prompt"]
        
        # Current state (simplified)
        state = np.random.rand(state_dim).astype(np.float32)
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(action_dim)
        else:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(state).to(model.device))
                action = q_values.argmax().item()
        
        # Generate text based on action (simplified)
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 20,
                temperature=0.7,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):]
        
        # Get reward from reward model
        with torch.no_grad():
            inputs = tokenizer(generated_text, return_tensors="pt").to(reward_pipeline.model.device)
            reward = reward_pipeline.model(**inputs).logits.item()
        
        # Next state (simplified)
        next_state = np.random.rand(state_dim).astype(np.float32)
        
        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state))
        
        # Learn from replay buffer
        if len(replay_buffer) > batch_size:
            # Sample batch
            batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in batch_indices]
            
            states = torch.FloatTensor([b[0] for b in batch]).to(model.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(model.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(model.device)
            next_states = torch.FloatTensor([b[3] for b in batch]).to(model.device)
            
            # Q-learning update
            current_q_values = q_network(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_q_values = q_network(next_states)
                max_next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + gamma * max_next_q_values
            
            # Compute loss and update
            loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Save Q-network
    torch.save(q_network.state_dict(), os.path.join(output_dir, "q_network.pt"))
    
    return q_network