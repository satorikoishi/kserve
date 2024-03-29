import argparse
import yaml
import os
import subprocess
        
def switch_torchserve_config(runtime):
    runtime_config_dir = os.path.join(os.path.dirname(__file__), "../config/runtimes")
    runtime_config_path = os.path.join(runtime_config_dir, "kustomization.yaml")
    storage_config_dir = os.path.join(os.path.dirname(__file__), "../config/configmap")
    storage_config_path = os.path.join(storage_config_dir, "inferenceservice.yaml")

    ######### Runtime Config ############
    with open(runtime_config_path, 'r') as file:
        runtime_config_data = yaml.safe_load(file)

    # Check and update the torchserve image config
    for image in runtime_config_data.get('images', []):
        if image['name'] == 'kserve-torchserve':
            if runtime == 'opt':
                image['newName'] = 'jwkaguya/torchserve-kfs'
                image['newTag'] = 'latest'
            elif runtime == 'base':
                image['newName'] = 'pytorch/torchserve-kfs'
                image['newTag'] = '0.8.2'
            else:
                assert False, f"Unexpected runtime {runtime}"

    with open(runtime_config_path, 'w') as file:
        yaml.safe_dump(runtime_config_data, file)
    
    ######### Storage Initializer Config ############
    with open(storage_config_path, 'r') as file:
        storage_config_data = file.read()
    
    # Locate the line containing 'enableDirectPvcVolumeMount' and modify it
    # Reverse the list to find the last occurrence first
    lines_reversed = storage_config_data.split('\n')[::-1]
    for i, line in enumerate(lines_reversed):
        if '"enableDirectPvcVolumeMount"' in line:
            indent = line[:line.index('"enableDirectPvcVolumeMount"')]
            if runtime == 'opt':
                new_line = f'{indent}"enableDirectPvcVolumeMount": true'
            elif runtime == 'base':
                new_line = f'{indent}"enableDirectPvcVolumeMount": false'
            else:
                assert False, f"Unexpected runtime {runtime}"
            lines_reversed[i] = new_line
            break
    modified_content = '\n'.join(lines_reversed[::-1])
    
    with open(storage_config_path, 'w') as file:
        file.write(modified_content)
    
    # Apply config
    print(f"kserve-torchserve configuration switched to: {runtime}.")
    try:
        subprocess.run(["kubectl", "apply", "-k", runtime_config_dir], check=True)
        subprocess.run(["kubectl", "apply", "-k", storage_config_dir], check=True)
        print(f"Successfully applied kustomization from {runtime_config_dir} and {storage_config_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to apply kustomization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch runtime between base and opt.")
    parser.add_argument("--runtime", "-r", required=False, type=str, default="opt", help="Runtime: base or opt(default).")

    args = parser.parse_args()
    switch_torchserve_config(args.runtime)
