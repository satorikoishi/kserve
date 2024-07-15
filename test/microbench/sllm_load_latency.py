import argparse
import gc
import json
import os
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_inference(model: nn.Module, model_path: str):
    # Inference
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_to_end_time = end_time - start_time
    throughput = outputs.shape[1] / end_to_end_time

    del outputs, tokenizer, inputs
    gc.collect()
    torch.cuda.empty_cache()

    return end_to_end_time, throughput, output_text


def measure(
    model_name: str, model_format: str, model_dir: str, loading_order: list
):
    results = []
    print(
        f"Measuring loading time for {model_format} model={model_name}, repeating {len(loading_order)} times"
    )
    # loading_order = torch.randperm(num_replicas)
    for model_idx in loading_order:
        print(f"Loading {model_name}_{model_idx}")
        model_record = {"model_name": f"{model_name}_{model_idx}"}

        # Model Loading
        if model_format == "safetensors":
            model_path = model_dir
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            end_time = time.time()
        elif model_format == "tl":
            model_path = model_dir
            start_time = time.time()
            model = torch.load(os.path.join(model_path, "model.pt"), map_location=torch.device('cuda:0'))
            end_time = time.time()
        model_record["loading_time"] = end_time - start_time

        # Inference
        end_to_end_time, throughput, output_text = benchmark_inference(
            model, model_path
        )

        model_record["end_to_end_time"] = end_to_end_time
        model_record["throughput"] = throughput
        model_record["output_text"] = output_text

        results.append(model_record)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results

def get_args():
    parser = argparse.ArgumentParser(description="Load test")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to serve",
    )
    parser.add_argument(
        "--model-format",
        type=str,
        required=True,
        choices=["safetensors", "tl"],
        help="Format to save the model in",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory to load models",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of replicas to load",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/sllm_load",
        help="Directory to save results",
    )
    return parser.parse_args()


def main():
    args = get_args()
    # _warmup_cuda()
    # _warmup_inference()

    model_format = args.model_format
    model_name = args.model_name
    model_dir = args.model_dir
    num_replicas = args.num_replicas
    output_dir = args.output_dir

    # Check if model_dir exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory {model_dir} does not exist")

    results = measure(model_name, model_format, model_dir, [0])

    output_filename = (
        f"{model_name}_{model_format}_{num_replicas}.json"
    )
    output_filename = output_filename.replace("/", "_")
    output_filename = os.path.join(output_dir, output_filename)

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()