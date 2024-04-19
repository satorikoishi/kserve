import os
import subprocess
import time

last_completed = [200, 10000, 100000, 0.2, 0.0]

resuming = False

def main():
    for num_nodes in [2, 20, 200, 2000]:
        if num_nodes < last_completed[0]:
            continue
        if num_nodes == last_completed[0]:
            resuming = True
        for num_models in [1000, 10000, 100000]:
            if resuming and num_models < last_completed[1]:
                continue
            if resuming and num_models == last_completed[1]:
                resuming = True
            for num_requests in [1000, 100000]:
                if resuming and num_requests < last_completed[2]:
                    continue
                if resuming and num_requests == last_completed[2]:
                    resuming = True
                for stress_level in [0.2, 1, 10]:
                    if resuming and stress_level < last_completed[3]:
                        continue
                    if resuming and stress_level == last_completed[3]:
                        resuming = True
                    for alpha in [0, 1.1, 1.6, 2.2, 4]:
                        if resuming and alpha < last_completed[4]:
                            continue
                        resuming = False
                        cmd = f"python3 ./simulation/test.py -n {num_nodes} -m {num_models} -r {num_requests} -i {stress_level} -a {alpha}"
                        print(f"Running: {cmd}")
                        subprocess.run(cmd, shell=True, check=True)
    
if __name__ == "__main__":
    main()