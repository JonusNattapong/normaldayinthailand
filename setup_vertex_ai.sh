#!/bin/bash

# คำสั่งสำหรับการสร้าง custom container และอัพโหลดไปยัง Container Registry
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"  # เลือก region ที่มี A100 GPU
IMAGE_NAME="llm-rl-training"
IMAGE_TAG="v1"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

# Create a temporary directory for our Dockerfile
TMP_DIR=$(mktemp -d)
cp Dockerfile ${TMP_DIR}/
mkdir -p ${TMP_DIR}/utils
cp train_pipeline.py ${TMP_DIR}/
cp utils/*.py ${TMP_DIR}/utils/

# Build and push container image
echo "Building container image..."
cd ${TMP_DIR}
gcloud builds submit --tag ${IMAGE_URI} .

# Clean up
cd -
rm -rf ${TMP_DIR}

echo "Container image built and pushed to ${IMAGE_URI}"

# สร้าง Cloud Storage bucket สำหรับเก็บโมเดลและผลลัพธ์
GCS_BUCKET_NAME="gs://${PROJECT_ID}-llm-training"
gcloud storage buckets create ${GCS_BUCKET_NAME} --location=${REGION} --uniform-bucket-level-access

echo "Created Cloud Storage bucket: ${GCS_BUCKET_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "GCS Output Path: ${GCS_BUCKET_NAME}/output"

# สร้างไฟล์ config สำหรับ Vertex AI Custom Job
cat > vertex_job_config.json << EOF
{
  "displayName": "LLM Training with RL techniques",
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
            "--batch-size", "4",
            "--gcs-output-path", "${GCS_BUCKET_NAME}/output",
            "--epochs", "1",
            "--wandb-project", "llm-rl-training",
            "--max-samples", "5000"
          ]
        }
      }
    ]
  }
}
EOF

echo "Created Vertex AI job config: vertex_job_config.json"
echo "To start training job run:"
echo "gcloud ai custom-jobs create --region=${REGION} --display-name=\"LLM Training Job\" --config=vertex_job_config.json"