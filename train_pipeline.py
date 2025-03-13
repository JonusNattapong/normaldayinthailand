import os
import torch
import argparse
import logging
from datetime import datetime
from google.cloud import storage
import wandb
import json

from .utils.model_manager import ModelManager
from .utils.data_loader import prepare_dataset, create_preference_data
from .utils.dpo_trainer import run_dpo_training
from .utils.reward_model import train_reward_model
from .utils.irl_training import train_irl
from .utils.q_learning import train_q_learning
from .utils.sac_training import train_sac
from .utils.ppo_trainer import run_ppo_training
from .utils.cot_trainer import train_cot
from .utils.ensemble_rl import train_ensemble_rl

# เพิ่มอัลกอริทึมใหม่
from .utils.a3c_training import train_a3c
from .utils.ddpg_training import train_ddpg
from .utils.td3_training import train_td3
from .utils.dqn_training import train_dqn
from .utils.rainbow_dqn_training import train_rainbow_dqn
from .utils.contrastive_rl import train_contrastive_rl
from .utils.bayesian_rl import train_bayesian_rl
from .utils.curriculum_rl import train_curriculum_rl
from .utils.self_supervised_rl import train_self_supervised_rl
from .utils.graph_based_rl import train_graph_based_rl
from .utils.intrinsic_motivation import train_intrinsic_motivation
from .utils.meta_rl_task_decomposition import train_meta_rl_task_decomposition
from .utils.distributional_rl import train_distributional_rl
from .utils.world_models import train_world_models
from .utils.advantage_weighted_regression import train_advantage_weighted_regression
from .utils.tsallis_entropy_rl import train_tsallis_entropy_rl
from .utils.hybrid_model_rl import train_hybrid_model_rl
from .utils.off_policy_correction import train_off_policy_correction
from .utils.information_exploration import train_information_exploration
from .utils.diffusion_rl import train_diffusion_rl
from .utils.multi_objective_rl import train_multi_objective_rl
from .utils.rag_training import train_rag


# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Pipeline")
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--base-model", type=str, default="scb10x/llama3.2-typhoon2-t1-3b-research-preview",
                            help="Base model name or path from Hugging Face Hub")
    model_group.add_argument("--model-type", choices=["pretrained", "finetuned"], default="pretrained",
                           help="Type of model to use")
    model_group.add_argument("--load-from-hub", action="store_true",
                           help="Download model from Hugging Face Hub if not available locally")
    
    # Dataset arguments
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument("--dataset", type=str, default="wisesight_sentiment",
                          help="Dataset name from Hugging Face Hub")
    data_group.add_argument("--dataset-subset", type=str,
                          help="Specific subset/configuration of the dataset")
    data_group.add_argument("--max-samples", type=int, default=5000,
                          help="Maximum number of samples to use from dataset")
    data_group.add_argument("--load-dataset-from-hub", action="store_true",
                          help="Download dataset from Hugging Face Hub if not available locally")
    
    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--batch-size", type=int, default=4,
                           help="Training batch size")
    train_group.add_argument("--epochs", type=int, default=1,
                           help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=1e-5,
                           help="Learning rate")
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument("--output-dir", type=str, default="output",
                            help="Local output directory")
    output_group.add_argument("--gcs-output-path", type=str, required=True,
                            help="Google Cloud Storage output path")
    output_group.add_argument("--wandb-project", type=str, default="llm-rl-training",
                            help="Weights & Biases project name")
    parser.add_argument("--run-stage", type=str, default="all",
                        choices=["all", "dpo", "reward", "irl", "q_learning", "sac", "ppo", "cot",
                                "a3c", "ddpg", "td3", "dqn", "rainbow_dqn", "contrastive_rl",
                                "bayesian_rl", "curriculum_rl", "self_supervised_rl", "graph_based_rl",
                                "intrinsic_motivation", "meta_rl_task_decomposition", "distributional_rl",
                                "world_models", "advantage_weighted_regression", "tsallis_entropy_rl",
                                "hybrid_model_rl", "off_policy_correction", "information_exploration",
                                "diffusion_rl", "rag"])
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
    all_stages = ['dpo', 'reward', 'irl', 'q_learning', 'sac', 'ppo', 'cot', 'a3c', 'ddpg', 'td3', 'dqn', 'rainbow_dqn',
                 'contrastive_rl', 'bayesian_rl', 'curriculum_rl', 'self_supervised_rl', 'graph_based_rl',
                 'intrinsic_motivation', 'meta_rl_task_decomposition', 'distributional_rl', 'world_models',
                 'advantage_weighted_regression', 'tsallis_entropy_rl', 'hybrid_model_rl', 'off_policy_correction',
                 'information_exploration', 'diffusion_rl', 'rag']
    stages_to_run = all_stages if args.run_stage == "all" else [args.run_stage]
    
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
    
    # 6. A3C Training
    a3c_output_dir = os.path.join(args.output_dir, "a3c_model")
    if 'a3c' in stages_to_run:
        logger.info("Starting A3C training...")
        a3c_model, a3c_tokenizer = train_a3c(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            a3c_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['a3c'] = {"completed": True, "path": a3c_output_dir}
        upload_to_gcs(a3c_output_dir, f"{args.gcs_output_path}/a3c_model")
    
        # 7. DDPG Training
    ddpg_output_dir = os.path.join(args.output_dir, "ddpg_model")
    if 'ddpg' in stages_to_run:
        logger.info("Starting DDPG training...")
        ddpg_model, ddpg_tokenizer = train_ddpg(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            ddpg_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            actor_lr=args.lr,
            critic_lr=args.lr*10
        )
        results['ddpg'] = {"completed": True, "path": ddpg_output_dir}
        upload_to_gcs(ddpg_output_dir, f"{args.gcs_output_path}/ddpg_model")
    
    # 8. TD3 Training
    td3_output_dir = os.path.join(args.output_dir, "td3_model")
    if 'td3' in stages_to_run:
        logger.info("Starting TD3 training...")
        td3_model, td3_tokenizer = train_td3(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            td3_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            actor_lr=args.lr,
            critic_lr=args.lr*10
        )
        results['td3'] = {"completed": True, "path": td3_output_dir}
        upload_to_gcs(td3_output_dir, f"{args.gcs_output_path}/td3_model")
    
    # 9. DQN Training
    dqn_output_dir = os.path.join(args.output_dir, "dqn_model")
    if 'dqn' in stages_to_run:
        logger.info("Starting DQN training...")
        dqn_model, dqn_tokenizer = train_dqn(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            dqn_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['dqn'] = {"completed": True, "path": dqn_output_dir}
        upload_to_gcs(dqn_output_dir, f"{args.gcs_output_path}/dqn_model")
    
    # 10. Rainbow DQN Training
    rainbow_output_dir = os.path.join(args.output_dir, "rainbow_dqn_model")
    if 'rainbow_dqn' in stages_to_run:
        logger.info("Starting Rainbow DQN training...")
        rainbow_model, rainbow_tokenizer = train_rainbow_dqn(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            rainbow_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['rainbow_dqn'] = {"completed": True, "path": rainbow_output_dir}
        upload_to_gcs(rainbow_output_dir, f"{args.gcs_output_path}/rainbow_dqn_model")
    
    # 11. PPO with enhancements from other methods
    final_output_dir = os.path.join(args.output_dir, "final_model")
    if 'ppo' in stages_to_run:
        logger.info("Starting enhanced PPO training...")
        ppo_model = run_ppo_training(
            dpo_output_dir if 'dpo' in results else args.base_model,
            train_dataset,
            eval_dataset,
            final_output_dir,
            reward_output_dir if 'reward' in results else None,
            q_output_dir if 'q_learning' in results else None,
            irl_output_dir if 'irl' in results else None,
            sac_output_dir if 'sac' in results else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['ppo'] = {"completed": True, "path": final_output_dir}
        upload_to_gcs(final_output_dir, f"{args.gcs_output_path}/final_model")
    
    # 12. Chain-of-Thought training
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
    
        # 13. Ensemble RL Training
    ensemble_output_dir = os.path.join(args.output_dir, "ensemble_rl_model")
    if 'ensemble_rl' in stages_to_run:
        logger.info("Starting Ensemble RL training...")
        model_paths = [dpo_output_dir, q_output_dir, sac_output_dir]  # Add other model paths as needed
        model_types = ['dpo', 'q_learning', 'sac']  # Add corresponding model types
        ensemble_model, ensemble_tokenizer = train_ensemble_rl(
            model_paths,
            model_types,
            train_dataset,
            ensemble_output_dir,
            reward_output_dir if 'reward' in results else None,
            args.batch_size,
            args.epochs,
            args.lr
        )
        results['ensemble_rl'] = {"completed": True, "path": ensemble_output_dir}
        upload_to_gcs(ensemble_output_dir, f"{args.gcs_output_path}/ensemble_rl_model")
    
        # 14. Contrastive RL Training
        contrastive_output_dir = os.path.join(args.output_dir, "contrastive_rl_model")
        if 'contrastive_rl' in stages_to_run:
            logger.info("Starting Contrastive RL training...")
            contrastive_model = train_contrastive_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                contrastive_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['contrastive_rl'] = {"completed": True, "path": contrastive_output_dir}
            upload_to_gcs(contrastive_output_dir, f"{args.gcs_output_path}/contrastive_rl_model")
    
        # 15. Bayesian RL Training
        bayesian_output_dir = os.path.join(args.output_dir, "bayesian_rl_model")
        if 'bayesian_rl' in stages_to_run:
            logger.info("Starting Bayesian RL training...")
            bayesian_model = train_bayesian_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                bayesian_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['bayesian_rl'] = {"completed": True, "path": bayesian_output_dir}
            upload_to_gcs(bayesian_output_dir, f"{args.gcs_output_path}/bayesian_rl_model")
    
        # 16. Curriculum RL Training
        curriculum_output_dir = os.path.join(args.output_dir, "curriculum_rl_model")
        if 'curriculum_rl' in stages_to_run:
            logger.info("Starting Curriculum RL training...")
            curriculum_model = train_curriculum_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                curriculum_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['curriculum_rl'] = {"completed": True, "path": curriculum_output_dir}
            upload_to_gcs(curriculum_output_dir, f"{args.gcs_output_path}/curriculum_rl_model")
    
        # 17. Self-Supervised RL Training
        self_supervised_output_dir = os.path.join(args.output_dir, "self_supervised_rl_model")
        if 'self_supervised_rl' in stages_to_run:
            logger.info("Starting Self-Supervised RL training...")
            self_supervised_model = train_self_supervised_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                self_supervised_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['self_supervised_rl'] = {"completed": True, "path": self_supervised_output_dir}
            upload_to_gcs(self_supervised_output_dir, f"{args.gcs_output_path}/self_supervised_rl_model")
    
        # 18. Graph-Based RL Training
        graph_based_output_dir = os.path.join(args.output_dir, "graph_based_rl_model")
        if 'graph_based_rl' in stages_to_run:
            logger.info("Starting Graph-Based RL training...")
            graph_based_model = train_graph_based_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                graph_based_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['graph_based_rl'] = {"completed": True, "path": graph_based_output_dir}
            upload_to_gcs(graph_based_output_dir, f"{args.gcs_output_path}/graph_based_rl_model")
    
        # 19. Intrinsic Motivation Training
        intrinsic_output_dir = os.path.join(args.output_dir, "intrinsic_motivation_model")
        if 'intrinsic_motivation' in stages_to_run:
            logger.info("Starting Intrinsic Motivation training...")
            intrinsic_model = train_intrinsic_motivation(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                intrinsic_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['intrinsic_motivation'] = {"completed": True, "path": intrinsic_output_dir}
            upload_to_gcs(intrinsic_output_dir, f"{args.gcs_output_path}/intrinsic_motivation_model")
    
        # 20. Meta RL Task Decomposition Training
        meta_rl_output_dir = os.path.join(args.output_dir, "meta_rl_task_decomposition_model")
        if 'meta_rl_task_decomposition' in stages_to_run:
            logger.info("Starting Meta RL Task Decomposition training...")
            meta_rl_model = train_meta_rl_task_decomposition(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                meta_rl_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['meta_rl_task_decomposition'] = {"completed": True, "path": meta_rl_output_dir}
            upload_to_gcs(meta_rl_output_dir, f"{args.gcs_output_path}/meta_rl_task_decomposition_model")
    
        # 21. Distributional RL Training
        distributional_output_dir = os.path.join(args.output_dir, "distributional_rl_model")
        if 'distributional_rl' in stages_to_run:
            logger.info("Starting Distributional RL training...")
            distributional_model = train_distributional_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                distributional_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['distributional_rl'] = {"completed": True, "path": distributional_output_dir}
            upload_to_gcs(distributional_output_dir, f"{args.gcs_output_path}/distributional_rl_model")
    
        # 22. World Models Training
        world_models_output_dir = os.path.join(args.output_dir, "world_models_model")
        if 'world_models' in stages_to_run:
            logger.info("Starting World Models training...")
            world_models_model = train_world_models(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                world_models_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['world_models'] = {"completed": True, "path": world_models_output_dir}
            upload_to_gcs(world_models_output_dir, f"{args.gcs_output_path}/world_models_model")
    
        # 23. Advantage Weighted Regression Training
        awr_output_dir = os.path.join(args.output_dir, "advantage_weighted_regression_model")
        if 'advantage_weighted_regression' in stages_to_run:
            logger.info("Starting Advantage Weighted Regression training...")
            awr_model = train_advantage_weighted_regression(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                awr_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['advantage_weighted_regression'] = {"completed": True, "path": awr_output_dir}
            upload_to_gcs(awr_output_dir, f"{args.gcs_output_path}/advantage_weighted_regression_model")
    
        # 24. Tsallis Entropy RL Training
        tsallis_output_dir = os.path.join(args.output_dir, "tsallis_entropy_rl_model")
        if 'tsallis_entropy_rl' in stages_to_run:
            logger.info("Starting Tsallis Entropy RL training...")
            tsallis_model = train_tsallis_entropy_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                tsallis_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['tsallis_entropy_rl'] = {"completed": True, "path": tsallis_output_dir}
            upload_to_gcs(tsallis_output_dir, f"{args.gcs_output_path}/tsallis_entropy_rl_model")
    
        # 25. Hybrid Model RL Training
        hybrid_output_dir = os.path.join(args.output_dir, "hybrid_model_rl_model")
        if 'hybrid_model_rl' in stages_to_run:
            logger.info("Starting Hybrid Model RL training...")
            hybrid_model = train_hybrid_model_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                hybrid_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['hybrid_model_rl'] = {"completed": True, "path": hybrid_output_dir}
            upload_to_gcs(hybrid_output_dir, f"{args.gcs_output_path}/hybrid_model_rl_model")
    
        # 26. Off-Policy Correction Training
        off_policy_output_dir = os.path.join(args.output_dir, "off_policy_correction_model")
        if 'off_policy_correction' in stages_to_run:
            logger.info("Starting Off-Policy Correction training...")
            off_policy_model = train_off_policy_correction(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                off_policy_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['off_policy_correction'] = {"completed": True, "path": off_policy_output_dir}
            upload_to_gcs(off_policy_output_dir, f"{args.gcs_output_path}/off_policy_correction_model")
    
        # 27. Information Exploration Training
        info_exploration_output_dir = os.path.join(args.output_dir, "information_exploration_model")
        if 'information_exploration' in stages_to_run:
            logger.info("Starting Information Exploration training...")
            info_exploration_model = train_information_exploration(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                info_exploration_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['information_exploration'] = {"completed": True, "path": info_exploration_output_dir}
            upload_to_gcs(info_exploration_output_dir, f"{args.gcs_output_path}/information_exploration_model")
    
        # 28. Diffusion RL Training
        diffusion_output_dir = os.path.join(args.output_dir, "diffusion_rl_model")
        if 'diffusion_rl' in stages_to_run:
            logger.info("Starting Diffusion RL training...")
            diffusion_model = train_diffusion_rl(
                dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset,
                diffusion_output_dir,
                reward_output_dir if 'reward' in results else None,
                args.batch_size,
                args.epochs,
                args.lr
            )
            results['diffusion_rl'] = {"completed": True, "path": diffusion_output_dir}
            upload_to_gcs(diffusion_output_dir, f"{args.gcs_output_path}/diffusion_rl_model")

        # 29. RAG Training
        rag_output_dir = os.path.join(args.output_dir, "rag_model")
        if 'rag' in stages_to_run:
            logger.info("Starting RAG training...")
            rag_model, rag_tokenizer = train_rag(
                base_model=dpo_output_dir if 'dpo' in results else args.base_model,
                train_dataset=train_dataset,
                output_dir=rag_output_dir,
                val_dataset=eval_dataset,
                reward_model_path=reward_output_dir if 'reward' in results else None,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                use_deepspeed=torch.cuda.device_count() > 1,
                n_trials=5 if torch.cuda.is_available() else 1,
                distributed_port=29500
            )
            results['rag'] = {"completed": True, "path": rag_output_dir}
            upload_to_gcs(rag_output_dir, f"{args.gcs_output_path}/rag_model")

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