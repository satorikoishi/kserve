import os
import json
import argparse
from transformers import AutoModel, AutoConfig, AutoTokenizer
import subprocess
import shutil
from kubernetes import client, config
from kubernetes.stream import stream
from utils import get_model_basename, get_model_seriesname

def prepare_setup_config(setup_config_file_path, args):
    print(f"Preparing setup_config file: {setup_config_file_path}")
    if os.path.exists(setup_config_file_path):
        # Open and read the existing configuration file
        with open(setup_config_file_path, 'r') as file:
            config = json.load(file)
    else:
        # If the file does not exist, create a default configuration
        config = {}
    # torchload setup
    if args.tl:
        config["use_torchload"] = True
    else:
        config["use_torchload"] = False
    # TODO: gpu, ... setup
    # Write the modified or new configuration back to the file
    with open(setup_config_file_path, 'w') as file:
        json.dump(config, file, indent=4)

def extract_extra_files(save_directory):
    possible_extra_files = ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "spiece.model", "setup_config.json"]
    possible_extra_files = [os.path.join(save_directory, x) for x in possible_extra_files]
    existing_extra_files = [x for x in possible_extra_files if os.path.exists(x)]
    return ','.join([x for x in existing_extra_files])
    
def create_mar_file(model_name, version, model_file, handler_file, extra_files=None, requirements_file=None, noarch=False):
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
    
    # Only mar file needs install dep
    if requirements_file and not noarch:
        mar_command.extend(["--requirements-file", requirements_file])
    
    if noarch:
        mar_command.extend(["--archive-format", "no-archive"])

    print(f"Running {mar_command} to generate .mar file")
    # Create the .mar file
    subprocess.run(mar_command)

def setup_model_store(model_basename, model_seriesname, save_directory, config_template_dir, noarch=False):
    # Reset model store
    model_store_path = os.path.join(save_directory, "model-store")
    shutil.rmtree(model_store_path)
    os.makedirs(model_store_path)
    # Move mar or dir to model store
    model_file = f'{model_basename}' if noarch else f'{model_basename}.mar'
    shutil.move(model_file, os.path.join(save_directory, f'model-store/{model_file}'))
    # Generate config.properties from template
    template_config = os.path.join(config_template_dir, "config.properties")
    target_config = os.path.join(save_directory, 'config/config.properties')
    replacements = {
        "MODEL_NAME": model_seriesname,
        "MAR_FILE_NAME": model_file
    }
    try:
        with open(template_config, 'r') as file:
            content = file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return
    for old, new in replacements.items():
        content = content.replace(old, new)
    try:
        with open(target_config, 'w') as file:
            file.write(content)
        print(f"File '{target_config}' updated successfully.")
    except IOError as e:
        print(f"Error writing file: {e}")

def setup_deployment(model_basename, model_seriesname, save_directory, yaml_dir):
    # mkdir in storage pod
    config.load_kube_config()
    v1 = client.CoreV1Api()
    try:
        # Executing the command
        resp = stream(v1.connect_get_namespaced_pod_exec, "model-store-pod", "default",
                      command=f"mkdir -p /pv/{model_basename}/config && mkdir -p /pv/{model_basename}/model-store".split(),
                      stderr=True, stdin=False,
                      stdout=True, tty=False)
        print("Response: " + resp)
    except Exception as e:
        print(f"Exception when executing command in pod: {e}")
    # Copy config and mar
    subprocess.run(f"kubectl cp {save_directory}/config/config.properties model-store-pod:/pv/{model_basename}/config", shell=True)
    subprocess.run(f"kubectl exec model-store-pod -- rm -r /pv/{model_basename}/model-store", shell=True)
    subprocess.run(f"kubectl cp {save_directory}/model-store model-store-pod:/pv/{model_basename}/model-store", shell=True)
    # Generate yaml from template
    template_yaml = os.path.join(yaml_dir, "template.yaml")
    target_yaml = os.path.join(yaml_dir, f"{model_basename}.yaml")
    replacements = {
        "METADATA_NAME": model_seriesname,
        "STORAGE_DIR": model_basename
    }
    try:
        with open(template_yaml, 'r') as file:
            content = file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return
    for old, new in replacements.items():
        content = content.replace(old, new)
    try:
        with open(target_yaml, 'w') as file:
            file.write(content)
        print(f"File '{target_yaml}' updated successfully.")
    except IOError as e:
        print(f"Error writing file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Pack mar file and deploy to pv.")
    parser.add_argument("--model_name", "-m", type=str, required=True, help="The name of the model to download.")
    parser.add_argument("--nogpu", action='store_true', help="Use handler with no gpu.")
    parser.add_argument("--tl", action='store_true', help="Use torch load pt file.")
    parser.add_argument("--noarch", action='store_true', help="Use no-archive instead of .mar file.")
    
    args = parser.parse_args()
    model_name = args.model_name
    model_basename = get_model_basename(model_name)
    model_seriesname = get_model_seriesname(model_basename)
    save_directory = os.path.join(os.path.dirname(__file__), f"../model_archive/{model_basename}")
    handler_dir = os.path.join(os.path.dirname(__file__), f"../model_archive/handler")
    config_template_dir = os.path.join(os.path.dirname(__file__), f"../model_archive/config")
    yaml_dir = os.path.join(os.path.dirname(__file__), f"../yaml/test")
    requirements_file = os.path.join(save_directory, "requirements.txt")
    requirements_template_dir = os.path.join(os.path.dirname(__file__), f"../model_archive/requirements")
    model_fname = "model.pt" if args.tl else "model.safetensors"
    handler_fname=f"{model_seriesname}_handler.py"
    
    # if args.tl:
    #     handler_fname=f"{model_seriesname}_handler_tl.py"
    # elif args.nogpu:
    #     handler_fname=f"{model_seriesname}_handler_no_gpu.py"
    # else:
    #     handler_fname=f"{model_seriesname}_handler.py"

    if model_seriesname in ('flan-t5', 'bloom'):
        # Need requirements file
        shutil.copy(os.path.join(requirements_template_dir, f'{model_seriesname}.txt'), requirements_file)
    
    prepare_setup_config(os.path.join(save_directory, "setup_config.json"), args)
    create_mar_file(model_basename, "1.0", os.path.join(save_directory, model_fname), 
                    os.path.join(handler_dir, handler_fname),
        extra_files=extract_extra_files(save_directory),
        requirements_file=requirements_file if os.path.exists(requirements_file) else None,
        noarch=args.noarch
    )
    setup_model_store(model_basename, model_seriesname, save_directory, config_template_dir, args.noarch)
    # Split mar and noarch
    if not args.noarch:
        model_basename += "-mar"
        if args.tl:
            model_basename += "-tl"
    setup_deployment(model_basename, model_seriesname, save_directory, yaml_dir)

if __name__ == "__main__":
    main()