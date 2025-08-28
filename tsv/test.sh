CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 3 --num_B 6 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a3b6.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 4 --num_B 5 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a4b5.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 4 --num_B 6 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a4b6.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 5 --num_B 2 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a5b2.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 5 --num_B 6 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a5b6.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 6 --num_B 1 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a6b1.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 6 --num_B 2 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a6b2.log 2>&1

CUDA_VISIBLE_DEVICES=2 python tsv_main3.py --model_name qwen2.5-7B --num_A 6 --num_B 6 --r 8 --dataset_name tqa --most_likely 1 > train_qwen_tqa_weight_a6b6.log 2>&1
