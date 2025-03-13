# การฝึกฝน Large Language Model ด้วย Google Cloud Vertex AI

## ภาพรวม

โปรเจกต์นี้นำเสนอไปป์ไลน์ที่ครอบคลุมสำหรับการฝึกฝน Large Language Model (LLM) โดยใช้เทคนิคการเรียนรู้แบบเสริมแรง (Reinforcement Learning) หลากหลายรูปแบบบน Google Cloud Vertex AI เทคนิคเหล่านี้ประกอบด้วย Direct Preference Optimization (DPO), Inverse Reinforcement Learning (IRL), Q-Learning, Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO) และ Chain-of-Thought (CoT)

โปรเจกต์นี้ถูกพัฒนาขึ้นเพื่อเป็นแหล่งเรียนรู้และทดลองสำหรับนักวิจัย นักศึกษา และผู้ที่สนใจในการพัฒนา LLM ภาษาไทย ซึ่งสามารถนำไปต่อยอดเพื่อสร้างโมเดลภาษาที่มีความสามารถในการเข้าใจและโต้ตอบกับผู้ใช้ภาษาไทยได้อย่างเป็นธรรมชาติ

## โครงสร้างของโปรเจกต์

```plaintext
.
├── Dockerfile                    # ไฟล์สำหรับสร้าง Docker container
├── requirements.txt              # รายการไลบรารีที่จำเป็น
├── train_pipeline.py             # ไปป์ไลน์หลักสำหรับการฝึกฝนโมเดล
├── save_to_hf.py                 # สคริปต์สำหรับบันทึกโมเดลไปยัง Hugging Face
├── setup_vertex_ai.sh            # สคริปต์สำหรับตั้งค่า Google Cloud Vertex AI
├── run_training.sh               # สคริปต์สำหรับเริ่มการฝึกฝนโมเดล
├── utils
│   ├── __init__.py
│   ├── data_loader.py            # โมดูลสำหรับโหลดข้อมูล
│   ├── dpo_trainer.py            # โมดูลสำหรับการฝึกฝนแบบ DPO
│   ├── reward_model.py           # โมดูลสำหรับโมเดลรางวัล
│   ├── irl_training.py           # โมดูลสำหรับการฝึกฝนแบบ IRL
│   ├── q_learning.py             # โมดูลสำหรับการฝึกฝนแบบ Q-Learning
│   ├── sac_training.py           # โมดูลสำหรับการฝึกฝนแบบ SAC
│   ├── ppo_trainer.py            # โมดูลสำหรับการฝึกฝนแบบ PPO
│   └── cot_trainer.py            # โมดูลสำหรับการฝึกฝนแบบ CoT
└── README.md                     # ไฟล์นี้
```

## การติดตั้งและการตั้งค่า

### สิ่งที่ต้องมีก่อน

1. **Google Cloud SDK**: ตรวจสอบว่าคุณได้ติดตั้งและตั้งค่า Google Cloud SDK แล้ว
2. **Docker**: ตรวจสอบว่ามี Docker ติดตั้งในระบบของคุณ
3. **Python**: ตรวจสอบว่าคุณมี Python 3.8 หรือใหม่กว่าติดตั้งอยู่

### ขั้นตอน

1. **โคลนโปรเจกต์**:
    ```bash
    git clone https://github.com/JonusNattapong/normaldayinthailand.git
    cd normaldayinthailand
    ```

2. **ติดตั้งไลบรารีที่จำเป็น**:
    ```bash
    pip install -r requirements.txt
    ```

3. **ตั้งค่า Google Cloud Vertex AI**:
    ```bash
    chmod +x setup_vertex_ai.sh
    ./setup_vertex_ai.sh
    ```

## การฝึกฝนโมเดล

### การฝึกฝนแบบครบวงจร

สำหรับการฝึกฝนโมเดลโดยใช้ทุกเทคนิคในคราวเดียว:

```bash
chmod +x run_training.sh
./run_training.sh us-central1 all 4 1
```

### การฝึกฝนทีละขั้นตอน

สำหรับการฝึกฝนโมเดลทีละขั้นตอนเพื่อจัดการทรัพยากรได้ดีขึ้น:

```bash
# ขั้นตอนที่ 1: การฝึกฝนแบบ DPO
./run_training.sh us-central1 dpo 4 1

# ขั้นตอนที่ 2: การฝึกฝนโมเดลรางวัล
./run_training.sh us-central1 reward 4 1

# ขั้นตอนที่ 3: การฝึกฝนแบบ IRL
./run_training.sh us-central1 irl 4 1

# ขั้นตอนที่ 4: การฝึกฝนแบบ Q-Learning
./run_training.sh us-central1 q_learning 4 1

# ขั้นตอนที่ 5: การฝึกฝนแบบ SAC
./run_training.sh us-central1 sac 4 1

# ขั้นตอนที่ 6: การฝึกฝนแบบ PPO
./run_training.sh us-central1 ppo 4 1

# ขั้นตอนที่ 7: การฝึกฝนแบบ CoT
./run_training.sh us-central1 cot 4 1
```

## การบันทึกโมเดลไปยัง Hugging Face

คุณสามารถบันทึกโมเดลที่ฝึกฝนแล้วไปยัง Hugging Face ได้โดยใช้สคริปต์ `save_to_hf.py`:

1. **ตั้งค่าข้อมูลประจำตัว Hugging Face**:
    - รับโทเค็น Hugging Face จาก [Hugging Face](https://huggingface.co/settings/tokens)

2. **รันสคริปต์**:
    ```bash
    python save_to_hf.py
    ```

## การติดตามและบันทึก

ติดตามงานฝึกฝนและบันทึกโดยใช้ Google Cloud Console:
- ไปที่ [Vertex AI](https://console.cloud.google.com/vertex-ai)
- ตรวจสอบสถานะของงานและดูบันทึก

## แนวทางการวิจัยและพัฒนาต่อยอด

โปรเจกต์นี้สามารถนำไปวิจัยและพัฒนาต่อยอดได้หลากหลายรูปแบบ โดยมีแนวทางที่น่าสนใจดังนี้

| แนวทางการวิจัย | รายละเอียด | อัลกอริทึมที่เกี่ยวข้อง | ระดับความซับซ้อน | ประโยชน์ต่อการพัฒนา AI |
|--------------|-----------|------------------|--------------|-----------------|
| การเรียนรู้แบบมีลำดับขั้น | การแยกงานซับซ้อนเป็นงานย่อยๆ | Hierarchical RL, Meta-RL | สูง | เพิ่มความสามารถในการจัดการงานซับซ้อน |
| การเรียนรู้แบบหลายวัตถุประสงค์ | การฝึกฝนที่ต้องบรรลุหลายเป้าหมายพร้อมกัน | Multi-Objective RL, TRPO | สูง | พัฒนาระบบ AI ที่สมดุลหลายด้าน |
| การเรียนรู้แบบเลียนแบบ | การเรียนรู้จากตัวอย่างการกระทำของผู้เชี่ยวชาญ | Adversarial IRL, Inverse RL | ปานกลาง | สร้าง AI ที่เลียนแบบผู้เชี่ยวชาญได้ |
| การเรียนรู้แบบสำรวจด้วยตนเอง | การค้นพบวิธีแก้ปัญหาใหม่ๆ | Curiosity RL, Information Exploration | สูง | ค้นพบวิธีการแก้ปัญหาที่สร้างสรรค์ |
| การเรียนรู้แบบปรับตัว | การรับมือกับสถานการณ์ที่เปลี่ยนแปลง | Meta-RL Task Decomposition, Rainbow DQN | สูง | AI ที่ปรับตัวได้ตามสถานการณ์ |
| การเรียนรู้แบบฝังความรู้ | การใช้ความรู้ที่มีอยู่เดิมในการเรียนรู้ | Knowledge Grounded RL, World Models | ปานกลาง | เพิ่มประสิทธิภาพด้วยความรู้เฉพาะทาง |
| การเรียนรู้แบบเสริมแรงขั้นสูง | เทคนิคการเรียนรู้ที่ซับซ้อน | TD3, SAC, DDPG | สูงมาก | เพิ่มประสิทธิภาพการเรียนรู้ |
| การเรียนรู้แบบหลายรูปแบบ | การผสมผสานข้อมูลหลายประเภท | Multi-Modal RL, Graph-Based RL | สูง | รองรับข้อมูลที่หลากหลาย |
| การเรียนรู้แบบมีโครงสร้าง | การใช้โครงสร้างข้อมูลพิเศษ | Graph-Based RL, Transformer XL RL | สูง | เพิ่มความสามารถในการเข้าใจโครงสร้าง |
| การเรียนรู้แบบกระจาย | การจัดการกับความไม่แน่นอน | Distributional RL, Bayesian RL | สูงมาก | รองรับความไม่แน่นอนในข้อมูล |

## จุดเด่นของโปรเจกต์สำหรับนักวิจัยไทย

โปรเจกต์นี้มีจุดเด่นสำหรับนักวิจัยและนักพัฒนาชาวไทยดังนี้:

1. **โครงสร้างที่ครบถ้วน**: มีเครื่องมือและโค้ดพร้อมใช้งานสำหรับการฝึกฝนโมเดลด้วยเทคนิคหลากหลาย
2. **รองรับการทำงานบนคลาวด์**: ลดภาระเรื่องทรัพยากรฮาร์ดแวร์สำหรับนักวิจัยที่มีงบประมาณจำกัด
3. **โอเพนซอร์ส**: สามารถนำไปต่อยอดได้อย่างอิสระ เหมาะสำหรับการศึกษาและวิจัย
4. **เทคโนโลยีทันสมัย**: ใช้เทคนิคการเรียนรู้แบบเสริมแรงที่เป็นที่นิยมในปัจจุบัน

เราหวังว่าโปรเจกต์นี้จะเป็นรากฐานสำคัญในการพัฒนา AI ภาษาไทยที่มีคุณภาพ และช่วยผลักดันให้เกิดนวัตกรรมด้านปัญญาประดิษฐ์ในประเทศไทย

## การมีส่วนร่วม

เรายินดีรับการมีส่วนร่วมจากทุกท่าน! หากคุณมีข้อเสนอแนะหรือพบข้อผิดพลาด กรุณาเปิด issue หรือส่ง pull request เข้ามาได้เลย

## ขอบเขตและความสามารถของโปรเจกต์

โปรเจกต์นี้รวบรวมเทคนิคการเรียนรู้แบบเสริมแรงที่ทันสมัยและหลากหลาย ประกอบด้วย:

1. **การเรียนรู้พื้นฐาน**
   - DQN (Deep Q-Network)
   - DDPG (Deep Deterministic Policy Gradient)
   - TD3 (Twin Delayed DDPG)
   - Rainbow DQN

2. **การเรียนรู้ขั้นสูง**
   - PPO (Proximal Policy Optimization)
   - TRPO (Trust Region Policy Optimization)
   - SAC (Soft Actor-Critic)

3. **การเรียนรู้แบบปรับตัว**
   - Meta-RL
   - Hierarchical RL
   - Curriculum RL
   - Reverse Curriculum

4. **การเรียนรู้แบบมีโครงสร้าง**
   - Graph-based RL
   - Transformer XL RL
   - World Models

5. **การเรียนรู้แบบพิเศษ**
   - Contrastive RL
   - Diffusion RL
   - Self-supervised RL

6. **การเรียนรู้แบบสำรวจ**
   - Curiosity-driven RL
   - Information Exploration
   - Intrinsic Motivation

7. **การเรียนรู้แบบผสมผสาน**
   - Multi-modal RL
   - Ensemble RL
   - Hybrid Model RL

8. **การเรียนรู้แบบมีเป้าหมาย**
   - Multi-objective RL
   - HER (Hindsight Experience Replay)
   - Knowledge-grounded RL

9. **การเรียนรู้แบบวิเคราะห์**
   - Bayesian RL
   - Distributional RL
   - Tsallis Entropy RL

## ลิขสิทธิ์

โปรเจกต์นี้ได้รับอนุญาตภายใต้ Apache License 2.0 - ดูรายละเอียดเพิ่มเติมได้ที่ไฟล์ [LICENSE](LICENSE)
