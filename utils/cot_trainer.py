import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def train_cot(model_name, train_dataset, eval_dataset, output_dir, batch_size, epochs, learning_rate):
    """
    ฝึกโมเดลด้วย Chain-of-Thought (CoT)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Prepare data for CoT
    def prepare_cot_data(dataset):
        cot_data = []
        for example in dataset:
            prompt = example['prompt']
            response = example['chosen']
            cot_example = f"{prompt} Let's think step by step: {response}"
            cot_data.append({"prompt": prompt, "response": cot_example})
        return cot_data

    train_cot_data = prepare_cot_data(train_dataset)
    eval_cot_data = prepare_cot_data(eval_dataset)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        for batch_idx in range(0, len(train_cot_data), batch_size):
            batch = train_cot_data[batch_idx:batch_idx + batch_size]
            inputs = tokenizer([ex['prompt'] for ex in batch], return_tensors='pt', padding=True, truncation=True)
            labels = tokenizer([ex['response'] for ex in batch], return_tensors='pt', padding=True, truncation=True)

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            outputs = model(**inputs, labels=labels.input_ids)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}")

    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer