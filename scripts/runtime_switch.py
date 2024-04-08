import argparse
from utils import switch_torchserve_config, set_stable_window

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch runtime between base and opt.")
    parser.add_argument("--runtime", "-r", required=False, type=str, default="opt", help="Runtime: base or opt(default).")
    parser.add_argument("--stablewindow", "-s", required=False, type=str, default="1m", help="Autoscaler stable window.")

    args = parser.parse_args()
    switch_torchserve_config(args.runtime)
    set_stable_window(args.stablewindow)
