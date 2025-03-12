import os
import torch
import argparse
import logging
from datetime import datetime
from google.cloud import storage
import wandb
import json

from .utils.data_loader import prepare_dataset, create_preference_data
from .utils.dpo_trainer import run_dpo_training
from .utils.reward_model import train_reward_model
from .utils.irl_training import train_irl
from .utils.q_learning import train_q_learning
from .utils.sac_training import train_sac
from .utils.ppo_trainer import run_ppo_training
from .utils.cot_trainer import train_cot

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    parser.add_argument("--base-model", type=str, default="scb10x/llama3.2-typhoon2-t1-3b-research-preview")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="/app/output")
    parser.add_argument("--gcs-output-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wandb-project", type=str, default="llm-rl-training")
    parser.add_argument("--dataset", type=str, default="wisesight_sentiment")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--run-stage", type=str, default="all", 
                        choices=["all", "dpo", "reward", "irl", "q_learning", "sac", "ppo", "cot"])
    return parser.parse_args()

def upload_to_gcs(source_dir, gcs_path):
    storage_client = storage.Client()
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    bucket = storage_client.bucket(bucket_name)
    
    for local_file in os.listdir(source_dir):
        local_path = os.path.join(source_dir, local_file)
        if os.path.isfile(local_path):
            blob_name = os.path.join(prefix, local_file)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to {gcs_path}/{local_file}")

def main():
    args = parse_args()
    
    # สร้าง output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ตรวจสอบ GPU
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ตั้งค่า Weights & Biases
    wandb.login()
    run = wandb.init(project=args.wandb_project, config=vars(args))
    
    # เตรียม dataset
    logger.info("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        args.dataset, 
        max_samples=args.max_samples
    )
    
    # บันทึกแยกตาม stage
    results = {}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ทำงานตาม stage ที่ต้องการ
    stages_to_run = ['dpo', 'reward', 'irl', 'q_learning', 'sac', 'ppo', 'cot'] if args.run_stage == "all" else [args.run_stage]
    
    # 1. DPO Training
    dpo_output_dir = os.path.join(args.output_dir, "dpo")
    if 'dpo' in stages_to_run:
        logger.info("Starting DPO training...")
        dpo_model, tokenizer = run_dpo_training(
            args.base_model,
            train_dataset,
            eval_dataset,
            dpo_output_dir,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['dpo'] = {"completed": True, "path": dpo_output_dir}
        # อัพโหลด checkpoint ไปยัง GCS
        upload_to_gcs(dpo_output_dir, f"{args.gcs_output_path}/dpo")
    else:
        logger.info("Skipping DPO training...")
    
    # 2. Reward Model Training
    reward_output_dir = os.path.join(args.output_dir, "reward_model")
    if 'reward' in stages_to_run:
        logger.info("Starting Reward Model training...")
        reward_model = train_reward_model(
            dpo_output_dir,
            train_dataset,
            eval_dataset,
            reward_output_dir,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['reward'] = {"completed": True, "path": reward_output_dir}
        upload_to_gcs(reward_output_dir, f"{args.gcs_output_path}/reward_model")
    
    # 3. IRL Training
    irl_output_dir = os.path.join(args.output_dir, "irl_model")
    if 'irl' in stages_to_run:
        logger.info("Starting IRL training...")
        irl_model = train_irl(
            dpo_output_dir,
            train_dataset,
            irl_output_dir,
            reward_output_dir,
            args.batch_size
        )
        results['irl'] = {"completed": True, "path": irl_output_dir}
        upload_to_gcs(irl_output_dir, f"{args.gcs_output_path}/irl_model")
    
    # 4. Q-Learning
    q_output_dir = os.path.join(args.output_dir, "q_model")
    if 'q_learning' in stages_to_run:
        logger.info("Starting Q-Learning...")
        q_model = train_q_learning(
            dpo_output_dir,
            train_dataset,
            q_output_dir,
            reward_output_dir,
            args.batch_size
        )
        results['q_learning'] = {"completed": True, "path": q_output_dir}
        upload_to_gcs(q_output_dir, f"{args.gcs_output_path}/q_model")
    
    # 5. SAC Training
    sac_output_dir = os.path.join(args.output_dir, "sac_model")
    if 'sac' in stages_to_run:
        logger.info("Starting SAC training...")
        sac_model = train_sac(
            dpo_output_dir,
            train_dataset,
            sac_output_dir,
            reward_output_dir,
            args.batch_size
        )
        results['sac'] = {"completed": True, "path": sac_output_dir}
        upload_to_gcs(sac_output_dir, f"{args.gcs_output_path}/sac_model")
    
    # 6. PPO with enhancements from other methods
    final_output_dir = os.path.join(args.output_dir, "final_model")
    if 'ppo' in stages_to_run:
        logger.info("Starting enhanced PPO training...")
        ppo_model = run_ppo_training(
            dpo_output_dir,
            train_dataset,
            eval_dataset,
            final_output_dir,
            reward_output_dir,
            q_output_dir if 'q_learning' in stages_to_run else None,
            irl_output_dir if 'irl' in stages_to_run else None,
            sac_output_dir if 'sac' in stages_to_run else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['ppo'] = {"completed": True, "path": final_output_dir}
        upload_to_gcs(final_output_dir, f"{args.gcs_output_path}/final_model")
    
    # 7. Chain-of-Thought training
    cot_output_dir = os.path.join(args.output_dir, "cot_model")
    if 'cot' in stages_to_run:
        logger.info("Starting Chain-of-Thought (CoT) training...")
        cot_model, cot_tokenizer = train_cot(
            args.base_model,
            train_dataset,
            eval_dataset,
            cot_output_dir,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['cot'] = {"completed": True, "path": cot_output_dir}
        upload_to_gcs(cot_output_dir, f"{args.gcs_output_path}/cot_model")
    
    # บันทึกผลลัพธ์ทั้งหมด
    with open(os.path.join(args.output_dir, f"training_results_{run_id}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # อัพโหลด results
    upload_to_gcs(args.output_dir, f"{args.gcs_output_path}")
    
    # จบการทำงาน
    logger.info("Training pipeline completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main()