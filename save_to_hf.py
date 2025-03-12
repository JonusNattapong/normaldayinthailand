import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder
from safetensors.torch import save_file

def save_model_to_hf(model_name, save_dir, hf_repo_name, hf_token):
    # โหลดโมเดลและ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    # สร้างโฟลเดอร์สำหรับบันทึกโมเดล
    os.makedirs(save_dir, exist_ok=True)
    
    # บันทึกโมเดลและ tokenizer ในรูปแบบ .safetensors
    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
    tokenizer.save_pretrained(save_dir)

    # อัพโหลดไปยัง Hugging Face
    api = HfApi()
    api.upload_folder(
        folder_path=save_dir,
        path_in_repo="",
        repo_id=hf_repo_name,
        token=hf_token
    )

if __name__ == "__main__":
    # กำหนดค่าพารามิเตอร์
    model_name = "scb10x/llama3.2-typhoon2-t1-3b-research-preview"
    save_dir = "./saved_model"
    hf_repo_name = "JonusNattapong/llama3.2-typhoon2-t1-3b"
    hf_token = HfFolder.get_token()

    # เรียกใช้งานฟังก์ชันเพื่อบันทึกโมเดลไปยัง Hugging Face
    save_model_to_hf(model_name, save_dir, hf_repo_name, hf_token)