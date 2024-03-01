import subprocess
import time

def get_model_basename(model_name):
    """
    Get the basename of the model name, removing any prefix.

    Parameters:
    model_name (str): The original model name.

    Returns:
    str: The basename of the model name.
    """
    return model_name.split("/")[-1] if "/" in model_name else model_name

def get_model_seriesname(model_basename):
    series_parts = model_basename.split('-')
    if len(series_parts) > 2:
        if 't5' in model_basename:
            series_name = '-'.join(series_parts[:-1])
        elif 'bert' in model_basename:
            series_name = series_parts[0]
        else:
            assert False, f"Unknown model {model_basename}, need further check"
    else:
        series_name = series_parts[0]
    return series_name

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

def prepare_deployment(model_name, runtime_config):
    cmd = f"python3 ./scripts/prepare_deployment.py -m {model_name}"
    if runtime_config == "base":
        pass
    elif runtime_config == "opt":
        cmd += " --tl --noarch"
    else:
        assert False, f"Unknown runtime config: {runtime_config}"
    subprocess.run(cmd, shell=True)