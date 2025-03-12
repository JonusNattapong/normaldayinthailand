from datasets import load_dataset, Dataset
import random
import torch

def prepare_dataset(dataset_name, max_samples=5000):
    """
    โหลดและเตรียม dataset พร้อมสร้าง preference pairs
    """
    if dataset_name == "wisesight_sentiment":
        dataset = load_dataset("wisesight_sentiment")
        train_data = dataset["train"]
        
        preference_data = {"prompt": [], "chosen": [], "rejected": []}
        negative_responses = [
            "ร้านนี้แย่มาก อาหารไม่อร่อย",
            "บริการแย่ ราคาแพงเกินไป ไม่คุ้มค่า",
            "อาหารรสชาติแย่ ไม่อร่อยเลย ไม่แนะนำ"
        ]
        positive_responses = [
            "ร้านนี้ดีมาก อาหารอร่อย",
            "บริการประทับใจ คุ้มค่ากับราคา",
            "อาหารรสชาติดีเยี่ยม แนะนำให้ลอง"
        ]

        sample_count = min(max_samples, len(train_data))
        sampled_indices = random.sample(range(len(train_data)), sample_count)

        for i in sampled_indices:
            prompt = "รีวิวร้านอาหาร: "
            text = train_data[i]["text"]
            label = train_data[i]["category"]
            
            if label == "positive":
                preference_data["prompt"].append(prompt)
                preference_data["chosen"].append(text)
                preference_data["rejected"].append(random.choice(negative_responses))
            elif label == "negative":
                preference_data["prompt"].append(prompt)
                preference_data["chosen"].append(random.choice(positive_responses))
                preference_data["rejected"].append(text)
            elif label == "neutral":
                preference_data["prompt"].append(prompt)
                preference_data["chosen"].append(text if random.random() > 0.5 else random.choice(positive_responses))
                preference_data["rejected"].append(random.choice(negative_responses))

        preference_dataset = Dataset.from_dict(preference_data)
        train_test_split = preference_dataset.train_test_split(test_size=0.2, seed=42)
        return train_test_split["train"], train_test_split["test"]
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

def create_preference_data(dataset):
    """
    สร้าง preference data จาก dataset ที่มีอยู่
    """
    return dataset