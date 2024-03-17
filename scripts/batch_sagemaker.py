import subprocess

model_name_list = ["bigscience/bloom-560m", 
                   "bert-base-uncased", "bert-large-uncased", 
                   "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]

def main():
    for model_name in model_name_list:
        subprocess.run(f"python3 ./scripts/sagemaker_deployment.py -m {model_name}", shell=True, check=True)
        # TODO: invoke and collect data
    
if __name__ == "__main__":
    main()