import os
import logging
from typing import Optional
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.pretrained_dir = "models/pretrained"
        self.finetuned_dir = "models/finetuned"
        self.datasets_dir = "datasets"
        self.api = HfApi()
        
        # Ensure directories exist
        os.makedirs(self.pretrained_dir, exist_ok=True)
        os.makedirs(self.finetuned_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def download_model(self, model_name: str, model_type: str = "pretrained") -> str:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            model_name (str): Name of the model on Hugging Face Hub
            model_type (str): Either "pretrained" or "finetuned"
            
        Returns:
            str: Path to the downloaded model
        """
        try:
            logger.info(f"Downloading model: {model_name}")
            
            # Choose directory based on model type
            target_dir = self.pretrained_dir if model_type == "pretrained" else self.finetuned_dir
            model_dir = os.path.join(target_dir, model_name.replace("/", "_"))
            
            # Download model and tokenizer
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save locally
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            logger.info(f"Model downloaded to: {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            raise
    
    def download_dataset(self, dataset_name: str, subset: Optional[str] = None) -> str:
        """
        Download a dataset from Hugging Face Hub.
        
        Args:
            dataset_name (str): Name of the dataset on Hugging Face Hub
            subset (str, optional): Specific subset/configuration of the dataset
            
        Returns:
            str: Path to the downloaded dataset
        """
        try:
            logger.info(f"Downloading dataset: {dataset_name}" + (f" ({subset})" if subset else ""))
            
            # Create dataset directory
            dataset_dir = os.path.join(self.datasets_dir, "raw", dataset_name.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download dataset
            if subset:
                dataset = load_dataset(dataset_name, subset)
            else:
                dataset = load_dataset(dataset_name)
            
            # Save dataset info
            dataset_info = {
                "name": dataset_name,
                "subset": subset,
                "splits": list(dataset.keys()),
                "features": str(dataset["train"].features if "train" in dataset else list(dataset.keys())[0])
            }
            
            # Save each split
            for split_name, split_data in dataset.items():
                split_path = os.path.join(dataset_dir, f"{split_name}.arrow")
                split_data.save_to_disk(split_path)
            
            logger.info(f"Dataset downloaded to: {dataset_dir}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_name}: {str(e)}")
            raise
    
    def list_available_models(self, filter_tags: Optional[list] = None) -> list:
        """
        List available models from Hugging Face Hub.
        
        Args:
            filter_tags (list, optional): List of tags to filter models
            
        Returns:
            list: List of model names
        """
        try:
            models = self.api.list_models(filter=filter_tags)
            return [model.modelId for model in models]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise
    
    def list_available_datasets(self, filter_tags: Optional[list] = None) -> list:
        """
        List available datasets from Hugging Face Hub.
        
        Args:
            filter_tags (list, optional): List of tags to filter datasets
            
        Returns:
            list: List of dataset names
        """
        try:
            datasets = self.api.list_datasets(filter=filter_tags)
            return [dataset.id for dataset in datasets]
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            raise

    def get_model_info(self, model_name: str) -> dict:
        """
        Get detailed information about a model.
        
        Args:
            model_name (str): Name of the model on Hugging Face Hub
            
        Returns:
            dict: Model information
        """
        try:
            model_info = self.api.model_info(model_name)
            return {
                "name": model_info.modelId,
                "tags": model_info.tags,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "library": model_info.library_name
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {str(e)}")
            raise
    
    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Get detailed information about a dataset.
        
        Args:
            dataset_name (str): Name of the dataset on Hugging Face Hub
            
        Returns:
            dict: Dataset information
        """
        try:
            dataset_info = self.api.dataset_info(dataset_name)
            return {
                "name": dataset_info.id,
                "tags": dataset_info.tags,
                "downloads": dataset_info.downloads,
                "likes": dataset_info.likes,
                "size": dataset_info.size_categories
            }
        except Exception as e:
            logger.error(f"Error getting dataset info for {dataset_name}: {str(e)}")
            raise