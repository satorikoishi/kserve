import torch
import time
import argparse
import mmap
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

def load_model_from_memory_mapped_file(mmap_file_path):
    with open(mmap_file_path, "r+b") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        model = torch.load(io.BytesIO(mmapped_file.read()))
        mmapped_file.close()
        return model

def load_model_multiple_times_from_memory_mapped_file(mmap_file_path, num_iterations=5):
    load_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        with open(mmap_file_path, "r+b") as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            model = torch.load(io.BytesIO(mmapped_file.read()))
            mmapped_file.close()
        end_time = time.time()
        load_times.append(end_time - start_time)
    return load_times

def main():
    parser = argparse.ArgumentParser(description="Compare model loading times.")
    parser.add_argument("save_directory", type=str, help="Directory where the models are saved")

    args = parser.parse_args()
    model_directory = args.save_directory

    # File paths
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

    # Measure load time from memory-mapped file
    mmap_load_time, model_from_mmap = measure_time(load_model_from_memory_mapped_file, model_path)
    print(f"Time taken to load from memory-mapped file: {mmap_load_time} seconds")
    
    # Measure load time from memory-mapped file multiple times
    mmap_load_times = load_model_multiple_times_from_memory_mapped_file(model_path)
    print(f"Load times from memory-mapped file across {len(mmap_load_times)} iterations: {mmap_load_times}")

if __name__ == "__main__":
    main()