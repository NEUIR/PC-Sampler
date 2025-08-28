# pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
# pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet


# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

echo "---------------------------PC-Sampler---------------------------"

echo "---------------------------Eval HumanEval---------------------------"

python eval.py \
    --task 'humaneval' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:5' \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 10 \
    --data_path ../data/humaneval.jsonl \
    --result_path results/humaneval_pc_sampler

python ../utils/judge_python_code.py \
    --folder_path results/humaneval_pc_sampler \
    --output_path results/humaneval_pc_sampler.txt

echo "---------------------------Eval MBPP---------------------------"

python eval.py \
    --task 'mbpp' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:5' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 10 \
    --data_path ../data/sanitized-mbpp.json \
    --result_path results/mbpp_pc_sampler

python ../utils/judge_python_code.py \
    --folder_path results/mbpp_pc_sampler \
    --output_path results/mbpp_pc_sampler.txt

echo "---------------------------Eval MATH-500---------------------------"

python eval.py \
    --task 'math500' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:5' \
    --gen_length 1024 \
    --steps 1024 \
    --block_length 1024 \
    --mode pc_sampler \
    --lambd 0.25 \
    --alpha 10 \
    --data_path ../data/math500.jsonl \
    --result_path results/math500_pc_sampler.txt

echo "---------------------------Eval Sudoku---------------------------"

python eval.py \
    --task 'sudoku' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:5' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --mode pc_sampler \
    --lambd 0.0 \
    --alpha 10 \
    --data_path ../data/sudoku.csv \
    --result_path results/sudoku_pc_sampler.txt

echo "---------------------------Eval Countdown---------------------------"

python eval.py \
    --task 'countdown' \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:5' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --mode pc_sampler \
    --lambd 0.5 \
    --alpha 10 \
    --data_path ../data/countdown.jsonl \
    --result_path results/countdown_pc_sampler.txt

echo "---------------------------Eval GSM8k---------------------------"

accelerate launch eval_llada.py \
    --tasks gsm8k \
    --num_fewshot 4 \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=256,lambd=0.25,alpha=10,mode=pc_sampler

echo "---------------------------Eval GPQA---------------------------"

accelerate launch eval_llada.py \
    --tasks gpqa \
    --num_fewshot 5 \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=256,lambd=0.25,alpha=10,mode=pc_sampler