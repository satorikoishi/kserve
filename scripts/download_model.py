import os
import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer, T5ForConditionalGeneration
import subprocess
import shutil
from kubernetes import client, config
from kubernetes.stream import stream
import torch

def get_model_basename(model_name):
    """
    Get the basename of the model name, removing any prefix.

    Parameters:
    model_name (str): The original model name.

    Returns:
    str: The basename of the model name.
    """
    return model_name.split("/")[-1]

def download_and_save_model(model_name, save_directory, save_mode="pretrained"):
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

    # Download
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # config.save_pretrained(save_directory)
    
    if save_mode == "pretrained":
        # Try safetensor
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model, config, and tokenizer saved in {save_directory}")
    elif save_mode == "ts":
        print("WARNING: NOT STABLE, ONLY FOR TESTING LATENCY")
        # dummy_input = "This is a dummy input for torch jit trace"
        # inputs = tokenizer.encode_plus(
        #     dummy_input,
        #     max_length=None,
        #     pad_to_max_length=True,
        #     add_special_tokens=True,
        #     return_tensors="pt",
        # )
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Device: {device}")
        # model.to(device).eval()
        # input_ids = inputs["input_ids"].to(device)
        # attention_mask = inputs["attention_mask"].to(device)
        # traced_model = torch.jit.trace(model, (input_ids, attention_mask), strict=False)
        # torch.jit.save(traced_model, os.path.join(save_directory, "traced_model.pt"))
        model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        torch.save(model, os.path.join(save_directory, "model.pt"))
        # torch.save(model.state_dict(), os.path.join(save_directory, "model.sd"))
        print(f"TS Model saved in {save_directory}")
    else:
        assert False, f"Unknown save mode: {save_mode}"
    
def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model with its config and tokenizer.")
    parser.add_argument("--model_name", "-m", type=str, help="The name of the model to download.")
    parser.add_argument("--save_mode", "-s", required=False, type=str, default="pretrained", help="Save mode(pretained or ts).")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_basename = get_model_basename(model_name)
    save_directory = os.path.join(os.path.dirname(__file__), f"../model_archive/{model_basename}")

    download_and_save_model(model_name, save_directory, args.save_mode)
    
if __name__ == "__main__":
    main()