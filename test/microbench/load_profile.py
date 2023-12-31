import cProfile
import pstats
import torch
import time
import argparse
import mmap
from transformers import AutoModel, T5Model, AutoConfig
import io
import pickle
import sys

def load_model_state_dict(model_directory):
    config = AutoConfig.from_pretrained(model_directory)
    model = T5Model(config)
    state_dict = torch.load(f"{model_directory}/model.sd")
    model.load_state_dict(state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare model loading times.")
    parser.add_argument("save_directory", type=str, help="Directory where the models are saved")

    args = parser.parse_args()
    model_directory = args.save_directory

    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    # Measure load time for loading state dict
    model_from_state_dict = load_model_state_dict(model_directory)
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    
if __name__ == "__main__":
    main()