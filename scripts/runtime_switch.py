import argparse
import yaml
import os
import subprocess

kustomization_dir = os.path.join(os.path.dirname(__file__), "../config/runtimes")
runtime_config_file_path = os.path.join(kustomization_dir, "kustomization.yaml")
        
def switch_torchserve_config(runtime):
    with open(runtime_config_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Check and update the torchserve image config
    for image in data.get('images', []):
        if image['name'] == 'kserve-torchserve':
            if runtime == 'opt':
                image['newName'] = 'jwkaguya/torchserve-kfs'
                image['newTag'] = 'latest'
            elif runtime == 'base':
                image['newName'] = 'pytorch/torchserve-kfs'
                image['newTag'] = '0.8.2'
            else:
                assert False, f"Unexpected runtime {runtime}"

    with open(runtime_config_file_path, 'w') as file:
        yaml.safe_dump(data, file)
    print(f"kserve-torchserve configuration switched to: {runtime}.")
    
    # Apply config
    try:
        subprocess.run(["kubectl", "apply", "-k", kustomization_dir], check=True)
        print(f"Successfully applied kustomization from {kustomization_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply kustomization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch runtime between base and opt.")
    parser.add_argument("--runtime", "-r", required=False, type=str, default="opt", help="Runtime: base or opt(default).")

    args = parser.parse_args()
    switch_torchserve_config(args.runtime)
