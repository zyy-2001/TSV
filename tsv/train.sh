CUDA_VISIBLE_DEVICES=3 python tsv_main.py  --model_name qwen2.5-7B  --dataset_name tqa --most_likely 1   > train_qwen_tqa.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python tsv_main.py  --model_name llama3.1-8B  --dataset_name tqa --most_likely 1

