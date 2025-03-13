import torch
import logging
import deepspeed
import optuna
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
import os
import wandb
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict, Any
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial,
             model: AutoModelForCausalLM,
             train_dataset,
             val_dataset,
             base_args: Dict[str, Any]) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=base_args['output_dir'],
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        **base_args
    )
    
    # Train and evaluate
    results = train_rag(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args
    )
    
    return results['eval_loss']


def train_rag(
    base_model: str,
    train_dataset,
    output_dir: str,
    val_dataset=None,
    reward_model_path: Optional[str] = None,
    batch_size: int = 4,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    use_deepspeed: bool = True,
    n_trials: int = 10,
    distributed_port: int = 29500
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Train a RAG (Retrieval-Augmented Generation) model with distributed training and hyperparameter optimization.
    
    Args:
        base_model (str): Path to base model or model identifier
        train_dataset: Training dataset
        output_dir (str): Output directory for saving model
        val_dataset: Validation dataset
        reward_model_path (str, optional): Path to reward model
        batch_size (int): Base batch size for training
        epochs (int): Number of training epochs
        learning_rate (float): Base learning rate
        use_deepspeed (bool): Whether to use DeepSpeed for distributed training
        n_trials (int): Number of hyperparameter optimization trials
        distributed_port (int): Port for distributed training
    
    Returns:
        tuple: (trained model, tokenizer)
    """
    try:
        logger.info("Initializing RAG training with advanced features...")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize distributed training
        if use_deepspeed and torch.cuda.is_available():
            dist.init_process_group(
                backend='nccl',
                init_method=f'tcp://localhost:{distributed_port}',
                world_size=torch.cuda.device_count(),
                rank=0
            )

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)

        # Hyperparameter optimization
        study = optuna.create_study(direction="minimize")
        base_args = {
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "evaluation_strategy": "steps" if val_dataset else "no",
            "save_strategy": "steps",
            "save_steps": 500,
            "logging_steps": 100,
            "report_to": "wandb",
            "load_best_model_at_end": True if val_dataset else False
        }

        logger.info("Starting hyperparameter optimization...")
        study.optimize(
            lambda trial: objective(
                trial, model, train_dataset, val_dataset, base_args
            ),
            n_trials=n_trials
        )

        # Get best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")

        # Configure DeepSpeed
        if use_deepspeed and torch.cuda.is_available():
            ds_config = {
                "train_batch_size": best_params.get("batch_size", batch_size),
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": best_params.get("learning_rate", learning_rate)
                    }
                },
                "fp16": {"enabled": True},
                "zero_optimization": {"stage": 2}
            }
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config
            )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_params.get("learning_rate", learning_rate))

        # Load reward model if provided
        reward_model = None
        if reward_model_path:
            logger.info(f"Loading reward model from {reward_model_path}")
            reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)
            reward_model = reward_model.to(model.device)
            reward_model.eval()

        # Create distributed dataloader
        train_sampler = DistributedSampler(train_dataset) if use_deepspeed else None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=best_params.get("batch_size", batch_size),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True
        )

        if val_dataset:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_deepspeed else None
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=best_params.get("batch_size", batch_size),
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True
            )

        # Training loop with improved error handling and logging
        logger.info("Starting main training loop...")
        best_val_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                if train_sampler:
                    train_sampler.set_epoch(epoch)
                
                for step, batch in enumerate(train_dataloader):
                    try:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        
                        # Forward pass
                        outputs = model(**batch)
                        loss = outputs.loss

                        # Add reward-based loss if available
                        if reward_model:
                            with torch.no_grad():
                                reward_outputs = reward_model(**batch)
                                rewards = reward_outputs.logits.mean(dim=1)
                            reward_loss = -torch.mean(rewards * outputs.logits.mean(dim=1))
                            loss = loss + reward_loss

                        # Backward pass
                        if use_deepspeed:
                            model.backward(loss)
                            model.step()
                        else:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        total_loss += loss.item()

                        # Log progress
                        if step % 10 == 0:
                            logger.info(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
                            wandb.log({
                                "epoch": epoch + 1,
                                "step": step,
                                "loss": loss.item()
                            })

                    except Exception as e:
                        logger.error(f"Error in training step: {str(e)}")
                        continue

                # Validation
                if val_dataset:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_batch in val_dataloader:
                            val_batch = {k: v.to(model.device) for k, v in val_batch.items()}
                            val_outputs = model(**val_batch)
                            val_loss += val_outputs.loss.item()

                    val_loss = val_loss / len(val_dataloader)
                    wandb.log({"val_loss": val_loss})

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"New best validation loss: {val_loss:.4f}")
                        model.save_pretrained(os.path.join(output_dir, "best_model"))
                        tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))

                # Log epoch results
                avg_loss = total_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs} completed. Average loss: {avg_loss:.4f}")
                wandb.log({
                    "epoch": epoch + 1,
                    "average_loss": avg_loss
                })

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise

        finally:
            # Save final model
            logger.info(f"Saving final model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Clean up distributed training
            if use_deepspeed and torch.cuda.is_available():
                dist.destroy_process_group()

        return model, tokenizer

    except Exception as e:
        logger.error(f"Fatal error in train_rag: {str(e)}")
        raise
