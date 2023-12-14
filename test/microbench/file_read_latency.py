import time
import argparse

def read_from_disk(file_path):
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(1024 * 1024)  # Read in chunks of 1MB
            if not data:
                break

def read_from_memory(file_data):
    data = file_data

def measure_time(func, *args):
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser(description="Measure file read latency from disk and memory.")
    parser.add_argument("file_path", type=str, help="Path to the file to be read")

    args = parser.parse_args()

    # Measure time taken to read from disk
    disk_read_time = measure_time(read_from_disk, args.file_path)
    print(f"Time taken to read from disk: {disk_read_time} seconds")

    # Load the file into memory
    with open(args.file_path, 'rb') as file:
        file_data = file.read()

    # Measure time taken to read from memory
    memory_read_time = measure_time(read_from_memory, file_data)
    print(f"Time taken to read from memory: {memory_read_time} seconds")

if __name__ == "__main__":
    main()
