#!/usr/bin/env python3
import argparse
import logging
from .utils.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Manage models and datasets from Hugging Face Hub")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download model command
    download_model_parser = subparsers.add_parser("download-model", help="Download a model from Hugging Face Hub")
    download_model_parser.add_argument("--name", required=True, help="Model name on Hugging Face Hub")
    download_model_parser.add_argument("--type", choices=["pretrained", "finetuned"], default="pretrained",
                                     help="Type of model to download")
    
    # Download dataset command
    download_dataset_parser = subparsers.add_parser("download-dataset", help="Download a dataset from Hugging Face Hub")
    download_dataset_parser.add_argument("--name", required=True, help="Dataset name on Hugging Face Hub")
    download_dataset_parser.add_argument("--subset", help="Specific subset/configuration of the dataset")
    
    # List available models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models on Hugging Face Hub")
    list_models_parser.add_argument("--tags", nargs="*", help="Filter models by tags")
    
    # List available datasets command
    list_datasets_parser = subparsers.add_parser("list-datasets", help="List available datasets on Hugging Face Hub")
    list_datasets_parser.add_argument("--tags", nargs="*", help="Filter datasets by tags")
    
    # Get model info command
    model_info_parser = subparsers.add_parser("model-info", help="Get detailed information about a model")
    model_info_parser.add_argument("--name", required=True, help="Model name on Hugging Face Hub")
    
    # Get dataset info command
    dataset_info_parser = subparsers.add_parser("dataset-info", help="Get detailed information about a dataset")
    dataset_info_parser.add_argument("--name", required=True, help="Dataset name on Hugging Face Hub")
    
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        manager = ModelManager()
        
        if args.command == "download-model":
            model_path = manager.download_model(args.name, args.type)
            logger.info(f"Model downloaded successfully to: {model_path}")
        
        elif args.command == "download-dataset":
            dataset_path = manager.download_dataset(args.name, args.subset)
            logger.info(f"Dataset downloaded successfully to: {dataset_path}")
        
        elif args.command == "list-models":
            models = manager.list_available_models(args.tags)
            logger.info("Available models:")
            for model in models:
                print(f"- {model}")
        
        elif args.command == "list-datasets":
            datasets = manager.list_available_datasets(args.tags)
            logger.info("Available datasets:")
            for dataset in datasets:
                print(f"- {dataset}")
        
        elif args.command == "model-info":
            info = manager.get_model_info(args.name)
            logger.info(f"Model information for {args.name}:")
            for key, value in info.items():
                print(f"{key}: {value}")
        
        elif args.command == "dataset-info":
            info = manager.get_dataset_info(args.name)
            logger.info(f"Dataset information for {args.name}:")
            for key, value in info.items():
                print(f"{key}: {value}")
        
        else:
            logger.error("No command specified. Use -h for help.")
            return 1
        
        return 0
    
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())