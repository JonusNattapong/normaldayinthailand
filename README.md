# LLM Training with Google Cloud Vertex AI

## Overview

This repository provides a comprehensive pipeline for training a Large Language Model (LLM) using various Reinforcement Learning (RL) techniques on Google Cloud Vertex AI. The techniques include Direct Preference Optimization (DPO), Inverse Reinforcement Learning (IRL), Q-Learning, Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), and Chain-of-Thought (CoT).

## Repository Structure

```plaintext
.
├── Dockerfile
├── requirements.txt
├── train_pipeline.py
├── save_to_hf.py
├── setup_vertex_ai.sh
├── run_training.sh
├── utils
│   ├── __init__.py
│   ├── data_loader.py
│   ├── dpo_trainer.py
│   ├── reward_model.py
│   ├── irl_training.py
│   ├── q_learning.py
│   ├── sac_training.py
│   ├── ppo_trainer.py
│   └── cot_trainer.py
└── README.md
```

## Setup and Installation

### Prerequisites

1. **Google Cloud SDK**: Ensure you have the Google Cloud SDK installed and configured.
2. **Docker**: Make sure Docker is installed on your system.
3. **Python**: Ensure you have Python 3.8 or later installed.

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JonusNattapong/normaldayinthailand.git
    cd normaldayinthailand
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Setup Google Cloud Vertex AI**:
    ```bash
    chmod +x setup_vertex_ai.sh
    ./setup_vertex_ai.sh
    ```

## Training the Model

### One-step Training

To train the model using all techniques in one go:

```bash
chmod +x run_training.sh
./run_training.sh us-central1 all 4 1
```

### Step-by-step Training

To train the model step-by-step for better resource management:

```bash
# Step 1: DPO Training
./run_training.sh us-central1 dpo 4 1

# Step 2: Reward Model Training
./run_training.sh us-central1 reward 4 1

# Step 3: IRL Training
./run_training.sh us-central1 irl 4 1

# Step 4: Q-Learning Training
./run_training.sh us-central1 q_learning 4 1

# Step 5: SAC Training
./run_training.sh us-central1 sac 4 1

# Step 6: PPO Training
./run_training.sh us-central1 ppo 4 1

# Step 7: CoT Training
./run_training.sh us-central1 cot 4 1
```

## Saving the Model to Hugging Face

You can save your trained model to Hugging Face using the `save_to_hf.py` script:

1. **Configure Hugging Face credentials**:
    - Obtain your Hugging Face token from [Hugging Face](https://huggingface.co/settings/tokens).

2. **Run the script**:
    ```bash
    python save_to_hf.py
    ```

## Monitoring and Logs

Monitor the training jobs and logs using Google Cloud Console:
- Go to [Vertex AI](https://console.cloud.google.com/vertex-ai)
- Check the status of your jobs and view logs

## Notes

- Be mindful of the costs associated with using GPU resources on Google Cloud.
- Regularly check the logs to ensure that the training process is running smoothly.
- Ensure that you have sufficient quota for GPU resources on Google Cloud.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
