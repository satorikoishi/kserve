from transformers import AutoModelForCausalLM

models = ['deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'facebook/opt-30b']

for model_name in models:
    model = None
    while model is None:
        try:
            # Attempt to download the model
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            # Print the error and retry
            print(f"Error downloading model: {e}")
            print("Retrying...")

    print("Model downloaded successfully!")