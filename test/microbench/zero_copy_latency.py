from typing import Tuple, List, Dict
import time
import transformers
import torch
import torchvision
import numpy as np
import pandas as pd
import urllib
import argparse
from transformers import AutoTokenizer
import pickle

import copy
import os
import sys

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    return end_time - start_time, result

def serialize_with_pickle(model):
    serialized_model = pickle.dumps(model)
    return serialized_model

def deserialize_with_pickle(serialized_model):
    model = pickle.loads(serialized_model)
    return model

def load_model_with_torch_load(model_path):
    model = torch.load(model_path)
    return model

def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # temporary buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in (
                [name for name, _ in module.named_parameters(recurse=False)]
                + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)

    # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors

def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
    """
    Restore the tensors that extract_tensors() stripped out of a 
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in 
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    with torch.inference_mode():
        modules = [module for _, module in m.named_modules()]
        for module, tensor_dict in zip(modules, tensors):
            # There are separate APIs to set parameters and buffers.
            for name, array in tensor_dict["params"].items():
                module.register_parameter(
                    name, torch.nn.Parameter(torch.as_tensor(array)))
            for name, array in tensor_dict["buffers"].items():
                module.register_buffer(name, torch.as_tensor(array))

def do_inference(model, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors='pt').input_ids
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids
    decoder_input_ids = model._shift_right(decoder_input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        last_hidden_states = output.last_hidden_state
        print(f'Last hidden state: {last_hidden_states}')
        # result = tokenizer.decode(output, skip_special_tokens=True)
        # print(f'Inference result: {result}')

def main():
    parser = argparse.ArgumentParser(description="Compare model loading times.")
    parser.add_argument("save_directory", type=str, help="Directory where the models are saved")

    args = parser.parse_args()
    model_directory = args.save_directory

    # File paths
    model_path = f'{model_directory}/model.pt'
    
    # Measure load time for torch.load()
    torch_load_time, model_from_torch_load = measure_time(load_model_with_torch_load, model_path)
    print(f"Time taken to load with torch.load(): {torch_load_time} seconds")
    
    print("Original model's output:")
    do_inference(model_from_torch_load, model_directory)
    
    # Replace tensor
    model_skeleton, model_weights = extract_tensors(model_from_torch_load)
    print(f"Model size: {sys.getsizeof(model_skeleton)}, Weight size: {sys.getsizeof(model_weights)}")
    
    replace_tensor_time, _ = measure_time(replace_tensors, model_skeleton, model_weights)
    print(f"Time taken to replace tensor: {replace_tensor_time} seconds")
    
    print("Model output after zero-copy model loading:")
    do_inference(model_skeleton, model_directory)

    # Deserialize
    serialized_time, serialized_model = measure_time(serialize_with_pickle, model_from_torch_load)
    print(f"Time taken to serialize torch.load with pickle: {serialized_time} seconds")

    deserialized_time, deserialized_model = measure_time(deserialize_with_pickle, serialized_model)
    print(f"Time taken to deserialize torch.load with pickle: {deserialized_time} seconds")

    print("Model output after deserialization:")
    do_inference(deserialized_model, model_directory)
    
    # Deserialize skeleton only
    serialized_skeleton_time, serialized_skeleton_model = measure_time(serialize_with_pickle, model_skeleton)
    print(f"Time taken to serialize skeleton: {serialized_skeleton_time} seconds")

    deserialized_skeleton_time, deserialized_skeleton_model = measure_time(deserialize_with_pickle, serialized_skeleton_model)
    print(f"Time taken to deserialize skeleton: {deserialized_skeleton_time} seconds")
    
    replace_tensor_time_second, _ = measure_time(replace_tensors, deserialized_skeleton_model, model_weights)
    print(f"Time taken to replace tensor: {replace_tensor_time_second} seconds")

    print("Model output after skeleton deserialization:")
    do_inference(deserialized_skeleton_model, model_directory)

if __name__ == "__main__":
    main()