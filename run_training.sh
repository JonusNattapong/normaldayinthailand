#!/bin/bash

# รับพารามิเตอร์จาก command line
REGION=${1:-"us-central1"}
RUN_STAGE=${2:-"all"}  # all, dpo, reward, irl, q_learning, sac, ppo
BATCH_SIZE=${3:-4}
EPOCHS=${4:-1}

# Load project info
PROJECT_ID=$(gcloud config get-value project)
GCS_BUCKET_NAME="gs://${PROJECT_ID}-llm-training"
IMAGE_URI="gcr.io/${PROJECT_ID}/llm-rl-training:v1"

# Create timestamp for job name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="llm_training_${RUN_STAGE}_${TIMESTAMP}"

# สร้างไฟล์ config สำหรับ Vertex AI Custom Job
cat > vertex_job_config_${TIMESTAMP}.json << EOF
{
  "displayName": "${JOB_NAME}",
  "jobSpec": {
    "workerPoolSpecs": [
      {
        "machineSpec": {
          "machineType": "n1-standard-16",
          "acceleratorType": "NVIDIA_TESLA_A100",
          "acceleratorCount": 1
        },
        "replicaCount": 1,
        "diskSpec": {
          "bootDiskType": "pd-ssd",
          "bootDiskSizeGb": 100
        },
        "containerSpec": {
          "imageUri": "${IMAGE_URI}",
          "args": [
            "--base-model", "scb10x/llama3.2-typhoon2-t1-3b-research-preview",
            "--batch-size", "${BATCH_SIZE}",
            "--gcs-output-path", "${GCS_BUCKET_NAME}/output/${TIMESTAMP}",
            "--epochs", "${EPOCHS}",
            "--run-stage", "${RUN_STAGE}",
            "--wandb-project", "llm-rl-training"
          ]
        }
      }
    ]
  }
}
EOF

echo "Creating Vertex AI custom job..."
gcloud ai custom-jobs create --region=${REGION} --config=vertex_job_config_${TIMESTAMP}.json

echo "Job submitted! Monitor at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "After completion, find outputs at: ${GCS_BUCKET_NAME}/output/${TIMESTAMP}"