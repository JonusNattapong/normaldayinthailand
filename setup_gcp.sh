#!/bin/bash
# ติดตั้ง Google Cloud SDK ถ้ายังไม่มี
if ! command -v gcloud &> /dev/null
then
    echo "กำลังติดตั้ง Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
    gcloud init
fi

# สร้าง project หรือใช้ project ที่มีอยู่
PROJECT_ID="llm-training-project"
gcloud projects create $PROJECT_ID --name="LLM Training Project"

# ตั้งค่า project เป็น default
gcloud config set project $PROJECT_ID

# เปิดใช้งาน Vertex AI API, Compute Engine API และ IAM API
gcloud services enable compute.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable iam.googleapis.com

# สร้าง service account สำหรับ Vertex AI
gcloud iam service-accounts create vertex-ai-training \
    --display-name="Vertex AI Training Service Account"

# ให้สิทธิ์ที่จำเป็น
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vertex-ai-training@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:vertex-ai-training@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"