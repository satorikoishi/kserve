import os
import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer
import subprocess
import shutil
from kubernetes import client, config
from kubernetes.stream import stream

def get_model_basename(model_name):
    """
    Get the basename of the model name, removing any prefix.

    Parameters:
    model_name (str): The original model name.

    Returns:
    str: The basename of the model name.
    """
    return model_name.split("/")[-1]

def download_and_save_model(model_name, save_directory):
    """
    Download a pre-trained model from Hugging Face along with its configuration and tokenizer.

    Parameters:
    model_name (str): The name of the model on Hugging Face.
    save_directory (str): Directory where the model, config, and tokenizer will be saved.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(os.path.join(save_directory, "config"), exist_ok=True)
    os.makedirs(os.path.join(save_directory, "model-store"), exist_ok=True)

    # Download the configuration
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # config.save_pretrained(save_directory)
    
    # Download the model
    model.save_pretrained(save_directory, safe_serialization=False)

    # Download the tokenizer
    tokenizer.save_pretrained(save_directory)
    print(f"Model, config, and tokenizer saved in {save_directory}")
    
def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model with its config and tokenizer.")
    parser.add_argument("--model_name", "-m", type=str, help="The name of the model to download.")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_basename = get_model_basename(model_name)
    save_directory = os.path.join(os.path.dirname(__file__), f"../model_archive/{model_basename}")

    download_and_save_model(model_name, save_directory)
    
if __name__ == "__main__":
    main()