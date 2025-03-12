import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        chosen_input = self.tokenizer(
            item["chosen"], 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        rejected_input = self.tokenizer(
            item["rejected"], 
            padding="max_length", 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": torch.cat([chosen_input["input_ids"][0], rejected_input["input_ids"][0]]),
            "attention_mask": torch.cat([chosen_input["attention_mask"][0], rejected_input["attention_mask"][0]]),
            "labels": torch.tensor([1.0, 0.0])  # chosen = 1, rejected = 0
        }

def train_reward_model(dpo_model_path, train_dataset, eval_dataset, output_dir, batch_size, epochs, learning_rate):
    """
    ฝึกโมเดลสำหรับการประเมินรางวัล (Reward Model)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(dpo_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        dpo_model_path, 
        num_labels=1, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Create datasets
    reward_train_dataset = RewardDataset(train_dataset, tokenizer)
    reward_eval_dataset = RewardDataset(eval_dataset, tokenizer)
    
    # Configure training
    reward_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=reward_model,
        args=reward_args,
        train_dataset=reward_train_dataset,
        eval_dataset=reward_eval_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    reward_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return reward_model