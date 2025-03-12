import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from google.cloud import storage

def download_from_gcs(gcs_path, local_dir):
    """
    ดาวน์โหลดโมเดลจาก GCS มายังเครื่องเฉพาะกาล
    """
    if not gcs_path.startswith("gs://"):
        return local_dir
        
    os.makedirs(local_dir, exist_ok=True)
    
    storage_client = storage.Client()
    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
    prefix = "/".join(gcs_path.replace("gs://", "").split("/")[1:])
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        name = blob.name
        rel_path = name[len(prefix):] if name.startswith(prefix) else name
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]
            
        if not rel_path:
            continue
            
        target_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        blob.download_to_filename(target_path)
        print(f"Downloaded {name} to {target_path}")
    
    return local_dir

def main():
    parser = argparse.ArgumentParser(description="Test trained LLM model")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to model directory, can be GCS path or local")
    parser.add_argument("--test-prompts", type=str, nargs="+", 
                        default=["รีวิวร้านอาหาร: ", "รีวิวร้านอาหาร: อยากกินอะไรอร่อยๆ"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--num-return-sequences", type=int, default=2)
    args = parser.parse_args()
    
    # ดาวน์โหลดโมเดลถ้าเป็น GCS path
    local_model_dir = "./downloaded_model"
    if args.model_path.startswith("gs://"):
        print(f"Downloading model from {args.model_path}...")
        download_from_gcs(args.model_path, local_model_dir)
        model_path = local_model_dir
    else:
        model_path = args.model_path
    
    # โหลดโมเดลและ tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # สร้าง text generation pipeline
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # ทดสอบโมเดล
    print("\n===== ผลการทดสอบโมเดล =====\n")
    for prompt in args.test_prompts:
        print(f"\nPrompt: {prompt}")
        outputs = generator(
            prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            do_sample=True,
        )
        
        for i, output in enumerate(outputs):
            generated_text = output['generated_text']
            print(f"Output {i+1}: {generated_text}")
    
    print("\n===== ทดสอบเสร็จสิ้น =====")

if __name__ == "__main__":
    main()