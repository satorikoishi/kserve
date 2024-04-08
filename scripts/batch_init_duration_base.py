import yaml
import os
import subprocess
import time
from utils import wait_for_pods_termination, get_model_seriesname, prepare_deployment
from runtime_switch import switch_torchserve_config

model_name_list = ["bloom-560m", 
                   "flan-t5-small", "flan-t5-base", "flan-t5-large", 
                   "bert-base-uncased", "bert-large-uncased"]
runtime_config = ["base", "baseplus", "opt"]
# runtime_config = ["baseplus"]

def main():
    for runtime in runtime_config:
        # Switch runtime
        if runtime == "baseplus":
            switch_torchserve_config("base")
        else:
            switch_torchserve_config(runtime)
        for model_name in model_name_list:
            try:
                # # Setup deployment (Only once is enough)
                # prepare_deployment(model_name, runtime)
                # Apply model
                model_seriesname = get_model_seriesname(model_name)
                if runtime == "base":
                    yaml_path = f'./yaml/test/{model_name}-mar.yaml'
                elif runtime == "baseplus":
                    yaml_path = f'./yaml/test/{model_name}-mar-tl.yaml'
                elif runtime == "opt":
                    yaml_path = f'./yaml/test/{model_name}.yaml'
                else:
                    assert False, f"Unknown runtime: {runtime}"
                subprocess.run(f"kubectl apply -f {yaml_path}", shell=True, check=True)
                # Wait and collect init data
                time.sleep(600)
                cmd = f"python3 ./scripts/init_duration.py -p {model_seriesname} --resdir comparison/{runtime}"
                if runtime == "opt":
                    cmd += " --pp"
                subprocess.run(cmd, shell=True, check=True)
            except Exception as e:
                print(f"An unexpected error occurred: {str(e)}")
            finally:
                # Delete and reclaim GPU resource
                subprocess.run(f"kubectl delete -f {yaml_path}", shell=True, check=True)
                wait_for_pods_termination(model_seriesname)
    
if __name__ == "__main__":
    main()