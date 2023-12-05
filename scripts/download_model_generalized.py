#!/usr/bin/python3
import json
import os
import sys

import torch
import transformers
from transformers import (
    AutoConfig,
    BloomModel,
    AutoTokenizer,
    set_seed,
)

print("Transformers version", transformers.__version__)
set_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def transformers_model_dowloader(repo, pretrained_model_name):
    """This function, save the checkpoint, config file along with tokenizer config and vocab files
    of a transformer model of your choice.
    """
    source = f"{repo}/{pretrained_model_name}"
    print("Download model and tokenizer", source)
    # loading pre-trained model and tokenizer
    
    config = AutoConfig.from_pretrained(
        source
    )
    model = BloomModel.from_pretrained(
        source, config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source
    )
    # NOTE : for demonstration purposes, we do not go through the fine-tune processing here.
    # A Fine_tunining process based on your needs can be added.
    # An example of  Fine_tuned model has been provided in the README.

    NEW_DIR = os.path.join(os.path.dirname(__file__), f"../model_archive/{pretrained_model_name}")
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print("Creation of directory %s failed" % NEW_DIR)
    else:
        print("Successfully created directory %s " % NEW_DIR)

    print(
        "Save model and tokenizer/ Torchscript model based on the setting from setup_config",
        source,
        "in directory",
        NEW_DIR,
    )
    
    model.save_pretrained(NEW_DIR, safe_serialization=False)
    tokenizer.save_pretrained(NEW_DIR)
    
    return

if __name__ == "__main__":
    transformers_model_dowloader("bigscience", "bloom-560m")