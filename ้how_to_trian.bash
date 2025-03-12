# normaldayinthailand
Google Cloud Vertex AI (DPO + RLHF + PPO + IRL + Q-Learning + SAC + COT ) / Model:  scb10x/llama3.2-typhoon2-t1-3b-research-preview
## Description
เพื่อฝึกโมเดล Language Model (LLM) ด้วยเทคนิค Direct Preference Optimization (DPO), Inverse Reinforcement Learning (IRL), Q-Learning, Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), และ Chain-of-Thought (COT) บน Google Cloud Vertex AI คุณสามารถทำตามขั้นตอนต่อไปนี้:

## 🔍 ขั้นตอนการเตรียมการ

### 1. ตั้งค่า Google Cloud Project

```bash
# ติดตั้ง Google Cloud SDK บนเครื่องของคุณ (ถ้ายังไม่มี)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# สร้าง project ใหม่หรือเลือก project ที่มีอยู่
gcloud projects create llm-training-project --name="LLM Training Project"  # หรือข้ามขั้นตอนนี้หากมี project อยู่แล้ว
gcloud config set project llm-training-project  # แทนที่ด้วยชื่อ project ของคุณ

# เปิดใช้งาน API ที่จำเป็น
gcloud services enable compute.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### 2. ตรวจสอบและขอ quota สำหรับ GPU

```bash
# ตรวจสอบ quota ปัจจุบัน
gcloud compute regions describe us-central1 | grep -A 10 "quotas:"

# ถ้า quota ไม่เพียงพอ ต้องยื่นคำขอเพิ่ม quota ในเว็บ Google Cloud Console
# ไปที่: IAM & Admin > Quotas & system limits > ค้นหา "NVIDIA_A100" และขอเพิ่ม quota
```

### 3. สร้าง Storage Bucket สำหรับเก็บข้อมูลและโมเดล

```bash
# สร้าง bucket
export BUCKET_NAME="llm-training-$(gcloud config get-value project)"
gcloud storage buckets create gs://$BUCKET_NAME --location=us-central1
```

### 4. สร้างไฟล์โค้ดทั้งหมด

```bash
# สร้างโฟลเดอร์สำหรับโปรเจค
mkdir -p llm-training-project/utils
cd llm-training-project

# สร้างไฟล์ source code ตามที่ให้ไว้ก่อนหน้านี้
# - Dockerfile
# - train_pipeline.py
# - utils/*.py
# - setup_vertex_ai.sh
# - run_training.sh
# - test_model.py
```

## 🏗️ ขั้นตอนการสร้าง Docker Image และอัพโหลดไปยัง Container Registry

### 5. สร้างและอัพโหลด Docker Image

```bash
# ทำให้สคริปต์สามารถรันได้
chmod +x setup_vertex_ai.sh

# รันสคริปต์เพื่อสร้างและอัพโหลด Docker image
./setup_vertex_ai.sh
```

## 🚄 ขั้นตอนการฝึกโมเดล

### 6. เริ่มการฝึกโมเดลตามลำดับ

#### วิธีที่ 1: ฝึกทั้งหมดพร้อมกัน (ใช้เวลานาน, ต้องการ GPU ราคาแพง)

```bash
# ฝึกทุกเทคนิคในรอบเดียว
chmod +x run_training.sh
./run_training.sh us-central1 all 4 1  # region, stage, batch_size, epochs
```

#### วิธีที่ 2: ฝึกทีละขั้นตอน (แนะนำ - ประหยัดทรัพยากรและง่ายต่อการดีบัก)

```bash
# ขั้นตอนที่ 1: ฝึกด้วย DPO ก่อน
./run_training.sh us-central1 dpo 4 1

# ขั้นตอนที่ 2: ฝึก Reward Model (รอให้ขั้นตอน DPO เสร็จก่อน)
./run_training.sh us-central1 reward 4 1

# ขั้นตอนที่ 3: ฝึก IRL
./run_training.sh us-central1 irl 4 1

# ขั้นตอนที่ 4: ฝึก Q-Learning
./run_training.sh us-central1 q_learning 4 1

# ขั้นตอนที่ 5: ฝึก SAC
./run_training.sh us-central1 sac 4 1

# ขั้นตอนที่ 6: ฝึก PPO โดยใช้โมเดลจากทุกขั้นตอนก่อนหน้า
./run_training.sh us-central1 ppo 4 1

# ขั้นตอนที่ 7: ฝึก CoT
./run_training.sh us-central1 cot 4 1
```

### 7. ตรวจสอบสถานะการฝึกโมเดล

```bash
# ดูรายการและสถานะการทำงาน
gcloud ai custom-jobs list --region=us-central1

# ดูรายละเอียดของงานฝึกโมเดลเฉพาะ
gcloud ai custom-jobs describe JOB_ID --region=us-central1
```

### 8. ตรวจสอบ logs ระหว่างการฝึกโมเดล

```bash
# ดู logs ของงานฝึกโมเดล
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

## 🔍 ขั้นตอนการทดสอบและใช้งานโมเดล

### 9. ทดสอบโมเดลที่ฝึกเสร็จแล้ว

```bash
# ค้นหา path ของโมเดลสุดท้าย (ตัวอย่าง)
export MODEL_PATH="gs://$BUCKET_NAME/output/TIMESTAMP/final_model"

# ทดสอบโมเดล
python test_model.py --model-path $MODEL_PATH --test-prompts "รีวิวร้านอาหาร: " "รีวิวร้านอาหาร: อยากกินอาหารไทยรสชาติดั้งเดิม"
```

### 10. เตรียมโมเดลสำหรับการใช้งาน

```bash
# ดาวน์โหลดโมเดลจาก GCS มาเก็บไว้บนเครื่อง (ถ้าต้องการ)
gsutil -m cp -r $MODEL_PATH ./local_model

# หรืออัพโหลดไปยัง Model Registry บน Vertex AI (แนะนำ)
MODEL_NAME="typhoon2-rl-combined"
MODEL_DISPLAY_NAME="Typhoon2 with RL techniques"

gcloud ai models upload \
  --region=us-central1 \
  --display-name=$MODEL_DISPLAY_NAME \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest \
  --artifact-uri=$MODEL_PATH
```

## ⚠️ คำแนะนำสำคัญ

1. **ระวังค่าใช้จ่าย**: A100 GPU มีราคาประมาณ $3-4 ต่อชั่วโมง ควรตรวจสอบการทำงานบ่อยๆ
2. **ตรวจสอบความคืบหน้า**: ใช้ Weights & Biases (ถ้าตั้งค่าไว้) หรือ Cloud Logging เพื่อติดตามความคืบหน้า
3. **เริ่มทีละขั้น**: แนะนำให้เริ่มจากการฝึกทีละขั้นตอนแทนการฝึกทั้งหมดพร้อมกัน
4. **ตรวจสอบ errors**: หมั่นตรวจสอบ logs เพื่อแก้ไขปัญหาได้ทันที
5. **สำรองโมเดล**: คอยสำรองไฟล์โมเดลสำคัญไว้เสมอ

เมื่อทำตามขั้นตอนทั้งหมด คุณจะได้โมเดล LLM ที่ผ่านการฝึกด้วยเทคนิค DPO, RLHF, IRL, Q-Learning, SAC, PPO และ COT ซึ่งมีความสามารถในการเข้าใจความต้องการของผู้ใช้ได้ดีขึ้น และตอบสนองได้อย่างเหมาะสมตามบริบท