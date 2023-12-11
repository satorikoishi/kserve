import os
import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer
import subprocess

def get_model_basename(model_name):
    """
    Get the basename of the model name, removing any prefix.

    Parameters:
    model_name (str): The original model name.

    Returns:
    str: The basename of the model name.
    """
    return model_name.split("/")[-1]

def get_model_seriesname(model_basename):
    series_parts = model_basename.split('-')
    if len(series_parts) > 1:
        series_name = '-'.join(series_parts[:-1])
    else:
        series_name = series_parts[0]
    return series_name

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
    # config.save_pretrained(save_directory)
    
    # Download the model
    model = AutoModel.from_pretrained(model_name, config=config)
    model.save_pretrained(save_directory, safe_serialization=False)

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    print(f"Model, config, and tokenizer saved in {save_directory}")
    
def create_mar_file(model_name, version, model_file, handler_file, extra_files=None, requirements_file=None):
    """
    Create a .mar file for TorchServe.

    Parameters:
    model_name (str): The name of the model.
    version (str): Model version.
    model_file (str): Path to the serialized model file (.pt or .pth).
    handler_file (str): Path to the handler file.
    extra_files (str, optional): Comma-separated paths to extra files needed by the model.
    requirements_file (str, optional): Path to the requirements.txt file.
    """
    mar_command = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", version,
        "--model-file", model_file,
        "--handler", handler_file
    ]

    if extra_files:
        mar_command.extend(["--extra-files", extra_files])
    
    if requirements_file:
        mar_command.extend(["--requirements-file", requirements_file])

    print(f"Running {mar_command} to generate .mar file")
    # Create the .mar file
    subprocess.run(mar_command)

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model with its config and tokenizer.")
    parser.add_argument("--model_name", "-m", type=str, help="The name of the model to download.")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_basename = get_model_basename(model_name)
    model_seriesname = get_model_seriesname(model_basename)
    save_directory = os.path.join(os.path.dirname(__file__), f"../model_archive/{model_basename}")
    handler_dir = os.path.join(os.path.dirname(__file__), f"../model_archive/handler")
    extra_json_files = ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]

    download_and_save_model(model_name, save_directory)
    create_mar_file(model_basename, "1.0", os.path.join(save_directory, "pytorch_model.bin"), os.path.join(handler_dir, f"{model_seriesname}_handler.py"),
        extra_files=','.join([os.path.join(save_directory, j) for j in extra_json_files]),
    )
    # Move to model store
    subprocess.run(f"mv {model_basename}.mar {os.path.join(save_directory, 'model-store')}", shell=True)
    
if __name__ == "__main__":
    main()