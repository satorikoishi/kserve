# python3 ./scripts/prepare_deployment.py -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -d /home/data/huggingface/fallserve-archive
# python3 ./scripts/prepare_deployment.py -m deepseek-ai/DeepSeek-R1-Distill-Qwen-32B -d /home/data/huggingface/fallserve-archive --tl
# python3 ./scripts/prepare_deployment.py -m facebook/opt-30b -d /home/data/huggingface/fallserve-archive
# python3 ./scripts/prepare_deployment.py -m facebook/opt-30b -d /home/data/huggingface/fallserve-archive --tl
python3 ./scripts/prepare_deployment.py -m meta-llama/Llama-2-13b-hf -d /home/data/huggingface/fallserve-archive
python3 ./scripts/prepare_deployment.py -m meta-llama/Llama-2-13b-hf -d /home/data/huggingface/fallserve-archive --tl
# python3 ./scripts/prepare_deployment.py -m meta-llama/Llama-2-13b-hf -d /home/data/huggingface/fallserve-archive --tl --noarch
