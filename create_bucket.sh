#!/bin/bash
# สร้าง bucket สำหรับเก็บข้อมูลและโมเดล
BUCKET_NAME="gs://llm-training-bucket-$(date +%s)"
REGION="us-central1" # เลือก region ที่มี A100 GPUs

gcloud storage buckets create $BUCKET_NAME --location=$REGION

echo "สร้าง bucket $BUCKET_NAME เสร็จสิ้น"