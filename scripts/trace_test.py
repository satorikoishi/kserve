import pandas as pd
import os
import subprocess
from utils import switch_torchserve_config, set_stable_window, get_model_seriesname

model_list = ['flan-t5-large']
trace_labels = ['Sporadic', 'Bursty', 'Periodic']
runtime_list = ['opt']
# runtime_list = ['base', 'baseplus', 'opt']
stable_window_list = ['1m']

def run_trace_once(model_name, trace):
    pass

def summary_data():
    pass

def run_trace():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), f"../results/trace/chosen_data.csv")).to_numpy()
    df = df[:, 4:14]    # Micro test for a few minutes
    # df = df[:, 4:]
    for runtime in runtime_list:
        switch_torchserve_config(runtime)
        for stable_window in stable_window_list:
            set_stable_window(stable_window)
            for model_name in model_list:
                # if runtime == "base":
                #     yaml_path = f'./yaml/test/scale/{model_name}-mar.yaml'
                # elif runtime == "baseplus":
                #     yaml_path = f'./yaml/test/scale/{model_name}-mar-tl.yaml'
                # elif runtime == "opt":
                #     yaml_path = f'./yaml/test/scale/{model_name}.yaml'
                # else:
                #     assert False, f"Unknown runtime: {runtime}"
                # subprocess.run(f"kubectl apply -f {yaml_path}", shell=True, check=True)
                for i, trace in enumerate(trace_labels):
                    e2e_lat_list, upstream_lat_list = run_trace_once(model_name, df[i])
                    summary_data(e2e_lat_list, upstream_lat_list, model_name, trace)
                
if __name__ == "__main__":
    run_trace()