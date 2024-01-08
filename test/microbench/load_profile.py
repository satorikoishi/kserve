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
import numpy as np

def load_model_state_dict(model_directory):
    config = AutoConfig.from_pretrained(model_directory)
    model = T5Model(config)
    state_dict = torch.load(f"{model_directory}/model.sd")
    model.load_state_dict(state_dict)
    return model

def load_model_with_torch_load(model_directory):
    model = torch.load(f"{model_directory}/model.pt")
    return model

def load_model_with_from_pretrained(model_directory):
    model = AutoModel.from_pretrained(model_directory)
    return model

def profile_model_loading(load_function, model_directory):
    """Profiles a given model loading function."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Measure load time for the given load function
    model = load_function(model_directory)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
    
    if hasattr(torch.serialization, 'load_tensor_durations'):
        print_load_tensor_latency()

    return model

def print_load_tensor_latency():
    # Assuming load_tensor_durations contains all the recorded durations
    num_records = len(torch.serialization.load_tensor_durations)
    median = np.percentile(torch.serialization.load_tensor_durations, 50)  # p50
    p90 = np.percentile(torch.serialization.load_tensor_durations, 90)    # p90
    p99 = np.percentile(torch.serialization.load_tensor_durations, 99)    # p99
    # Print the results
    print(f"Number of Records: {num_records}")
    print(f"Median (p50) Latency: {median} seconds")
    print(f"p90 Latency: {p90} seconds")
    print(f"p99 Latency: {p99} seconds")

def main():
    parser = argparse.ArgumentParser(description="Compare model loading times.")
    parser.add_argument("save_directory", type=str, help="Directory where the models are saved")

    args = parser.parse_args()

    print("Profiling load_model_state_dict...")
    profile_model_loading(load_model_state_dict, args.save_directory)
    
    print("---------------------------------- Splitter ----------------------------------\n\n\n")

    print("\nProfiling load_model_with_torch_load...")
    profile_model_loading(load_model_with_torch_load, args.save_directory)
        
    print("---------------------------------- Splitter ----------------------------------\n\n\n")
    
    print("\nProfiling load_model_with_pretrained...")
    profile_model_loading(load_model_with_from_pretrained, args.save_directory)
    
    print("Model loading profiling completed.")
    
if __name__ == "__main__":
    main()