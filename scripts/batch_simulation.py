import os
import subprocess
import time

def main():
    for num_nodes in [2, 20, 200, 2000]:
        for num_models in [1000, 10000, 100000]:
            for num_requests in [1000, 100000]:
                for stress_level in [0.2, 1, 10]:
                    for alpha in [0, 1.1, 1.6, 2.2, 4]:
                        cmd = f"python3 ./simulation/test.py -n {num_nodes} -m {num_models} -r {num_requests} -i {stress_level} -a {alpha}"
                        print(f"Running: {cmd}")
                        subprocess.run(cmd, shell=True, check=True)
    
if __name__ == "__main__":
    main()