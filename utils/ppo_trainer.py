import os
import torch
from trl import PPOConfig, PPOTrainer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def create_enhanced_reward_fn(reward_model_path, q_model_path=None, irl_model_path=None, sac_model_path=None):
    """
    สร้าง reward function ที่ผสมผสานผลลัพธ์จากทุกโมเดล
    """
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    reward_pipeline = pipeline("text-classification", model=reward_model_path, tokenizer=tokenizer, device=0)
    
    # Load Q-network if available
    q_net = None
    if q_model_path:
        from q_learning import QNetworkForText
        state_dim = 128
        action_dim = 10
        q_net = QNetworkForText(state_dim, action_dim).to(reward_pipeline.model.device)
        q_net.load_state_dict(torch.load(os.path.join(q_model_path, "q_network.pt")))
        q_net.eval()
    
    def reward_fn(samples):
        scores = []
        
        for sample in samples:
            # Base reward from reward model
            pred = reward_pipeline(sample, truncation=True, max_length=256)
            rm_score = pred[0]["score"] if pred[0]["label"] == "POSITIVE" else 1 - pred[0]["score"]
            
            # Enhanced score with Q-network if available
            q_score = 0.0
            if q_net:
                # Create a simple state representation from text
                sample_encoding = tokenizer(sample, truncation=True, max_length=128, 
                                           padding="max_length", return_tensors="pt").to(reward_pipeline.model.device)
                state_repr = torch.mean(sample_encoding.input_ids.float(), dim=1).cpu().numpy()[0][:state_dim]
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state_repr).to(reward_pipeline.model.device))
                    q_score = q_values.max().item() / 10.0  # Normalize
            
            # Weight the components (hyperparameters to tune)
            weights = {
                'reward_model': 0.7,
                'q_learning': 0.3,
            }
            
            combined_score = weights['reward_model'] * rm_score
            if q_net:
                combined_score += weights['q_learning'] * q_score
            
            scores.append(torch.tensor(combined_score))
        
        return scores
    
    return reward_fn

def run_ppo_training(
    dpo_model_path, 
    train_dataset, 
    eval_dataset, 
    output_dir, 
    reward_model_path, 
    q_model_path=None, 
    irl_model_path=None,
    sac_model_path=None,
    batch_size=4,
    epochs=1,
    learning_rate=1e-5
):
    """
    ฝึกโมเดลด้วย PPO โดยใช้ reward function ที่ผสมผสานจากทุกโมเดล
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model from DPO
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    model = AutoModelForCausalLM.from_pretrained(dpo_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Create enhanced reward function
    enhanced_reward_fn = create_enhanced_reward_fn(
        reward_model_path, 
        q_model_path, 
        irl_model_path,
        sac_model_path
    )
    
    # PPO configuration
    ppo_config = PPOConfig(
        model_name=dpo_model_path,
        learning_rate=learning_rate,
        batch_size=batch_size * 4,  # Effective batch size
        ppo_epochs=epochs,
        mini_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
    )
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    # Training loop
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Generate responses
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, max_length=50, temperature=0.7)
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            # Get rewards from enhanced reward function
            rewards = enhanced_reward_fn(responses)
            
            # PPO step
            ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save as safetensors
    from safetensors.torch import save_file
    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    
    return model