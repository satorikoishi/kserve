import torch
import time
import argparse
from transformers import AutoModel
import io

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return end_time - start_time, result

def load_model_with_torch_load(model_path):
    model = torch.load(model_path)
    return model

def load_model_with_from_pretrained(model_directory):
    model = AutoModel.from_pretrained(model_directory)
    return model

def load_model_from_memory(model_stream):
    model = torch.load(model_stream)
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare model loading times.")
    parser.add_argument("save_directory", type=str, help="Directory where the models are saved")

    args = parser.parse_args()
    model_directory = args.save_directory

    # File paths
    # bin_model_path = f'{model_directory}/pytorch_model.bin'
    model_path = f'{model_directory}/model.pt'

    # Measure load time for from_pretrained()
    from_pretrained_time, model_from_pretrained = measure_time(load_model_with_from_pretrained, model_directory)
    print(f"Time taken to load with from_pretrained(): {from_pretrained_time} seconds")

    # Measure load time for torch.load()
    torch_load_time, model_from_torch_load = measure_time(load_model_with_torch_load, model_path)
    print(f"Time taken to load with torch.load(): {torch_load_time} seconds")

    # Measure access time for model already in memory (from_pretrained)
    memory_access_time_pretrained, _ = measure_time(lambda m: m, model_from_pretrained)
    print(f"Time taken to access model already in memory (from_pretrained): {memory_access_time_pretrained} seconds")

    # Measure access time for model already in memory (torch.load)
    memory_access_time_torch_load, _ = measure_time(lambda m: m, model_from_torch_load)
    print(f"Time taken to access model already in memory (torch.load): {memory_access_time_torch_load} seconds")

if __name__ == "__main__":
    main()