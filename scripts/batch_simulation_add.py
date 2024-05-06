import os
import subprocess
import time

def main():
    for num_nodes in [2000]:
        for num_models in [100000]:
            for num_requests in [100000]:
                for stress_level in [100]:
                    for alpha in [1.1, 2.2]:
                        cmd = f"python3 ./simulation/test.py -n {num_nodes} -m {num_models} -r {num_requests} -i {stress_level} -a {alpha} --save_file add.csv"
                        print(f"Running: {cmd}")
                        subprocess.run(cmd, shell=True, check=True)
    
if __name__ == "__main__":
    main()