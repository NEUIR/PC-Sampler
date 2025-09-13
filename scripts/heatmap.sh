
python method_comparison_heatmap.py \
    --task humaneval \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --data_path ../data/humaneval.jsonl \
    --samples_num 200 \

python paint_heatmap.py \
    --task humaneval \
    --data_path heatmap_params/humaneval.json \

python method_comparison_heatmap.py \
    --task mbpp \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --data_path ../data/sanitized-mbpp.json \
    --samples_num 200 \

python paint_heatmap.py \
    --task mbpp \
    --data_path heatmap_params/mbpp.json \

python method_comparison_heatmap.py \
    --task math500 \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 1024 \
    --steps 1024 \
    --block_length 1024 \
    --data_path ../data/math500.jsonl \
    --samples_num 200 \

python paint_heatmap.py \
    --task math500 \
    --data_path heatmap_params/math500.json \

python method_comparison_heatmap.py \
    --task sudoku \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --data_path ../data/sudoku.csv \
    --samples_num 200 \

python paint_heatmap.py \
    --task sudoku \
    --data_path heatmap_params/sudoku.json \

python method_comparison_heatmap.py \
    --task countdown \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 128 \
    --steps 128 \
    --block_length 128 \
    --data_path ../data/countdown.jsonl \
    --samples_num 200 \

python paint_heatmap.py \
    --task countdown \
    --data_path heatmap_params/countdown.json \

python method_comparison_heatmap.py \
    --task gsm8k \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --data_path ../data/gsm8k.jsonl \
    --samples_num 200 \

python paint_heatmap.py \
    --task gsm8k \
    --data_path heatmap_params/gsm8k.json \

python method_comparison_heatmap.py \
    --task gpqa \
    --model_name 'GSAI-ML/LLaDA-8B-Instruct' \
    --device 'cuda:4' \
    --gen_length 256 \
    --steps 256 \
    --block_length 256 \
    --data_path ../data/gpqa.jsonl \
    --samples_num 200 \

python paint_heatmap.py \
    --task gpqa \
    --data_path heatmap_params/gpqa.json \