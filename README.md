# 🤖 Advanced Reinforcement Learning Training Framework

## 📝 ภาพรวมโครงการ
ระบบฝึกฝนและพัฒนาโมเดล Reinforcement Learning แบบครบวงจร รองรับอัลกอริทึมที่หลากหลายและมีเครื่องมือสำหรับการฝึกฝนโมเดลอย่างมีประสิทธิภาพ

## ✨ คุณสมบัติหลัก
- รองรับอัลกอริทึม RL หลากหลายรูปแบบ:
  - Policy Gradient: PPO, TRPO, A3C
  - Value-Based: DQN, Rainbow DQN
  - Actor-Critic: DDPG, TD3, SAC
  - Meta-RL และ Hierarchical RL
  - Self-Supervised RL
  - Curiosity-driven RL
- ระบบการจัดการโมเดลอัตโนมัติ
- รองรับการฝึกฝนแบบ Multi-modal
- มีระบบ Curriculum Learning
- รองรับการทำ Distributional RL
- มีระบบ Knowledge-grounded RL
- รองรับการใช้งานร่วมกับ Diffusion Models
- มีระบบ World Models สำหรับการเรียนรู้แบบ Model-based

## 📋 ความต้องการของระบบ
- Python 3.8+
- CUDA compatible GPU (แนะนำ)
- เครื่องมือและไลบรารีที่จำเป็น (ติดตั้งผ่าน requirements.txt)

## 🚀 การติดตั้ง

1. โคลนโปรเจค:
```bash
git clone <repository-url>
cd normaldayinthailand
```

2. ติดตั้ง dependencies:
```bash
bash install_requirements.bash
```

3. ตั้งค่าไดเรกทอรีที่จำเป็น:
```bash
bash setup_directories.sh
```

## 💻 การใช้งาน

### การเริ่มต้นฝึกฝนโมเดล
```bash
bash how_to_train.bash
```

### การบันทึกโมเดลไปยัง Hugging Face
```bash
bash save_to_hf.bash
```

## 📁 โครงสร้างไดเรกทอรี

```
.
├── configs/                  # ไฟล์การตั้งค่าต่างๆ
├── utils/                   # โมดูลและฟังก์ชันสนับสนุน
│   ├── ppo_trainer.py       # การฝึกฝนด้วย PPO
│   ├── dqn_training.py      # การฝึกฝนด้วย DQN
│   ├── a3c_training.py      # การฝึกฝนด้วย A3C
│   └── ...                  # อัลกอริทึมอื่นๆ
├── train_pipeline.py        # ไปป์ไลน์หลักสำหรับการฝึกฝน
└── test_model.py           # สคริปต์สำหรับทดสอบโมเดล
```

## 🛠️ รายละเอียดโมดูล Utils

### Policy-Based Algorithms
- **ppo_trainer.py**: การฝึกฝนด้วยอัลกอริทึม Proximal Policy Optimization สำหรับการเรียนรู้นโยบายที่มีเสถียรภาพ
- **trpo.py**: Trust Region Policy Optimization สำหรับการปรับปรุงนโยบายภายในขอบเขตที่กำหนด
- **a3c_training.py**: Asynchronous Advantage Actor-Critic สำหรับการเรียนรู้แบบขนาน

### Value-Based Methods
- **dqn_training.py**: Deep Q-Network สำหรับการเรียนรู้ฟังก์ชันมูลค่า
- **rainbow_dqn_training.py**: Rainbow DQN รวมการปรับปรุง DQN หลายรูปแบบ
- **distributional_rl.py**: การเรียนรู้แบบกระจายสำหรับการประมาณค่าผลตอบแทน

### Actor-Critic Methods
- **ddpg_training.py**: Deep Deterministic Policy Gradient สำหรับ continuous action spaces
- **td3_training.py**: Twin Delayed DDPG เพิ่มความเสถียรในการเรียนรู้
- **sac_training.py**: Soft Actor-Critic สำหรับการเรียนรู้แบบ maximum entropy

### Advanced RL Techniques
- **meta_rl.py**: Meta Reinforcement Learning สำหรับการปรับตัวกับงานใหม่
- **hierarchical_rl.py**: การเรียนรู้แบบลำดับชั้นสำหรับงานซับซ้อน
- **curriculum_rl.py**: การเรียนรู้แบบลำดับขั้นตอน
- **world_models.py**: การสร้างโมเดลสภาพแวดล้อมเสมือน

### Exploration Strategies
- **curiosity_rl.py**: การสำรวจด้วยความอยากรู้
- **information_exploration.py**: การสำรวจโดยใช้ทฤษฎีข้อมูล
- **intrinsic_motivation.py**: แรงจูงใจภายในสำหรับการสำรวจ

### Imitation and Inverse RL
- **adversarial_irl.py**: Inverse RL แบบ Adversarial
- **her_rl.py**: Hindsight Experience Replay

### Multi-Task and Meta-Learning
- **meta_rl_task_decomposition.py**: การแยกงานย่อยใน Meta-RL
- **multi_objective_rl.py**: การเรียนรู้แบบหลายวัตถุประสงค์

### Advanced Model Architectures
- **transformer_xl_rl.py**: Transformer-XL สำหรับการเรียนรู้แบบ long-term dependencies
- **diffusion_rl.py**: การใช้ Diffusion Models ใน RL
- **hybrid_model_rl.py**: การผสมผสานโมเดลหลายรูปแบบ

### Ensemble and Robust Methods
- **ensemble_rl.py**: การใช้หลายโมเดลเพื่อเพิ่มความแม่นยำ
- **bayesian_rl.py**: การเรียนรู้แบบ Bayesian
- **off_policy_correction.py**: การแก้ไขการเรียนรู้แบบ Off-policy

### Knowledge-Based Methods
- **knowledge_grounded_rl.py**: การใช้ความรู้พื้นฐานในการเรียนรู้
- **rag_training.py**: Retrieval-Augmented Generation Training
- **graph_based_rl.py**: การเรียนรู้บนโครงสร้างกราฟ

### Other Utilities
- **model_manager.py**: ระบบจัดการโมเดลและการบันทึก รวมถึงการจัดการเวอร์ชัน การโหลด/บันทึกโมเดล การติดตามการทดลอง และการจัดการ checkpoints
- **data_loader.py**: การโหลดและจัดการข้อมูล รองรับหลายรูปแบบข้อมูล (episodes, transitions, demonstrations) พร้อมระบบ preprocessing และ augmentation
- **reward_model.py**: การสร้างและปรับแต่งฟังก์ชันรางวัล รวมถึงการเรียนรู้ฟังก์ชันรางวัลจากผู้เชี่ยวชาญและการปรับแต่งแบบ inverse RL
- **contrastive_rl.py**: การเรียนรู้แบบ Contrastive เพื่อสร้าง representations ที่มีประสิทธิภาพสำหรับสถานะและการกระทำ
- **self_supervised_rl.py**: การเรียนรู้แบบไม่ต้องการผู้สอนเพื่อปรับปรุง representations และนโยบายการเรียนรู้
- **q_learning.py**: การเรียนรู้แบบ Q-Learning พื้นฐานและการขยายความสามารถสำหรับ deep learning
- **tsallis_entropy_rl.py**: การใช้ Tsallis entropy ในการควบคุมการสำรวจและการเรียนรู้นโยบาย
- **advantage_weighted_regression.py**: การถดถอยแบบถ่วงน้ำหนักด้วย advantage สำหรับการปรับปรุงนโยบาย
- **cot_trainer.py**: Chain of Thought Trainer สำหรับการฝึกฝนการคิดเป็นลำดับขั้นตอน
- **dpo_trainer.py**: Direct Preference Optimization สำหรับการเรียนรู้จากการเปรียบเทียบคู่
- **irl_training.py**: Inverse Reinforcement Learning สำหรับการเรียนรู้ฟังก์ชันรางวัลจากการสาธิต
- **multi_modal_rl.py**: การผสมผสานข้อมูลหลายรูปแบบ (ภาพ, ข้อความ, เสียง) ในการเรียนรู้

## ⚙️ การกำหนดค่า
การตั้งค่าหลักสามารถปรับแต่งได้ผ่านไฟล์ในโฟลเดอร์ `configs/`
- `rag_config.json`: การตั้งค่าสำหรับ Retrieval-Augmented Generation
- อื่นๆ: สามารถเพิ่มไฟล์คอนฟิกเพิ่มเติมตามความต้องการ

## 🎯 การฝึกฝนโมเดล

1. **การเตรียมข้อมูล**
   - จัดเตรียมข้อมูลในรูปแบบที่เหมาะสม
   - ตั้งค่าพารามิเตอร์ในไฟล์คอนฟิก

2. **การเริ่มฝึกฝน**
   ```bash
   python train_pipeline.py --config configs/your_config.json
   ```

3. **การติดตามผล**
   - ระบบจะแสดงผลการฝึกฝนในระหว่างการทำงาน
   - สามารถดูผลลัพธ์เพิ่มเติมได้จากไฟล์ล็อก

## 🌟 คุณสมบัติพิเศษ

### 🔄 Curriculum Learning
- รองรับการเรียนรู้แบบลำดับขั้น
- ปรับความยากของงานอัตโนมัติ

### 🧠 Meta-RL
- เรียนรู้การปรับตัวกับงานใหม่ได้อย่างรวดเร็ว
- รองรับการแยกงานย่อยอัตโนมัติ

### 🎯 Multi-Objective RL
- ฝึกฝนโมเดลสำหรับเป้าหมายหลายอย่างพร้อมกัน
- ปรับสมดุลระหว่างวัตถุประสงค์ต่างๆ

## 🛠️ การใช้งานบน Cloud

### Google Cloud Platform
```bash
bash setup_gcp.sh
```

### Vertex AI
```bash
bash setup_vertex_ai.sh
```

## 📈 การทดสอบและประเมินผล
```bash
python test_model.py --model-path path/to/your/model
```

## 🔍 การแก้ไขปัญหาทั่วไป

1. **ปัญหาหน่วยความจำ**
   - ลดขนาด batch size
   - ใช้ gradient accumulation

2. **ปัญหา CUDA**
   - ตรวจสอบการติดตั้ง CUDA
   - ตรวจสอบความเข้ากันได้ของเวอร์ชัน

## 📚 อ้างอิงและทรัพยากร
- [อัลกอริทึม PPO](https://arxiv.org/abs/1707.06347)
- [Rainbow DQN](https://arxiv.org/abs/1710.02298)
- [Meta-RL](https://arxiv.org/abs/1611.05763)

## 🤝 การมีส่วนร่วม
ยินดีรับ Pull Requests และการรายงานปัญหา สามารถเปิด Issue ได้ที่ repository

## 📄 ลิขสิทธิ์
โปรเจคนี้อยู่ภายใต้ลิขสิทธิ์ MIT License
