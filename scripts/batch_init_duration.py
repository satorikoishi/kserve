import yaml
import os
import subprocess
import time

yaml_path = './yaml/test/flan-t5-large.yaml'
pod_prefix = "flan-t5"

def are_pods_terminating(prefix):
    cmd = f"kubectl get pods --field-selector=status.phase=Running -o custom-columns=:metadata.name"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error in fetching pod details")
        return False

    pod_names = result.stdout.decode().split()
    # Filter pods with the given prefix and check if any are still running
    return any(pod.startswith(prefix) for pod in pod_names)

def wait_for_pods_termination(prefix, timeout=600):
    print(f"Waiting for pods with prefix {prefix} to terminate...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if not are_pods_terminating(prefix):
            print(f"All pods with prefix {prefix} are terminated.")
            return True
        time.sleep(10)
        print("Checking again...")

    print(f"Timeout reached. Some pods with prefix {prefix} may still be running.")
    return False

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