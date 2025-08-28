CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 1 --dataset_name tqa --most_likely 1 > train_qwen_tqa_new_r1.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 2 --dataset_name tqa --most_likely 1 > train_qwen_tqa_new_r2.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 4 --dataset_name tqa --most_likely 1 > train_qwen_tqa_new_r4.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 16 --dataset_name tqa --most_likely 1 > train_qwen_tqa_new_r16.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 32 --dataset_name tqa --most_likely 1 > train_qwen_tqa_new_r32.log 2>&1


CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 1 --batch_size 64 --dataset_name triviaqa --most_likely 1 > train_qwen_triviaqa_new_r1.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 2 --batch_size 64 --dataset_name triviaqa --most_likely 1 > train_qwen_triviaqa_new_r2.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 4 --batch_size 64 --dataset_name triviaqa --most_likely 1 > train_qwen_triviaqa_new_r4.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 16 --batch_size 64 --dataset_name triviaqa --most_likely 1 > train_qwen_triviaqa_new_r16.log 2>&1
CUDA_VISIBLE_DEVICES=2 python tsv_main2.py --model_name qwen2.5-7B --num_A 1 --num_B 1 --r 32 --batch_size 64 --dataset_name triviaqa --most_likely 1 > train_qwen_triviaqa_new_r32.log 2>&1

