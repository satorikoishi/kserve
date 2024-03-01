import yaml
import os
import subprocess
import time
from utils import wait_for_pods_termination

yaml_path = './yaml/test/flan-t5-large.yaml'
pod_prefix = "flan-t5"

def modify_yaml(cpu_value):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Modify the CPU values in limits and requests
    data['spec']['predictor']['model']['resources']['limits']['cpu'] = str(cpu_value)
    data['spec']['predictor']['model']['resources']['requests']['cpu'] = str(cpu_value)

    # # Save the modified data to a new YAML file
    # new_yaml_path = f'./yaml/test/flan-t5-large-cpu{cpu_value}.yaml'
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

def main():
    # List of CPU values to iterate over
    cpu_values = [1, 2, 4, 8, 16]

    # Modify and save a new YAML file for each CPU value
    for cpu in cpu_values:
        wait_for_pods_termination(pod_prefix)
        modify_yaml(cpu)
        subprocess.run(f"kubectl apply -f {yaml_path}", shell=True, check=True)
        time.sleep(60)
        cmd = f"python3 ./scripts/simplified_init_duration.py -p flan-t5 --resdir resource --suffix {cpu}cpu"
        subprocess.run(cmd, shell=True, check=True)
        subprocess.run(f"kubectl delete -f {yaml_path}", shell=True, check=True)
    
if __name__ == "__main__":
    main()