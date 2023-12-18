from transformers import AutoTokenizer, T5Model
import torch

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = torch.load("model_archive/flan-t5-large/model.pt")
# model = T5Model.from_pretrained("google/flan-t5-large")

input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)