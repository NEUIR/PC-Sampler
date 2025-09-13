from transformers import AutoTokenizer, AutoModel
import torch
import argparse
from tqdm import tqdm
import os, sys, json
import numpy as np
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import query_extract, load_dataset

def generate(model, tokenizer, input, task, steps, gen_length, block_length, temperature, mode, lambd, alpha, baseline_name):

    query = query_extract(input, task)
    m = [{"role": "user", "content": query}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer(user_input)['input_ids']
    prompt = torch.tensor(prompt).to(model.device).unsqueeze(0)
    if mode == 'original':
        from src.generate import generate
        _, orders = generate(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence', return_order=True)
    elif mode == 'pc_sampler':
        from src.generate import generate_with_pc_sampler
        _, orders = generate_with_pc_sampler(model, prompt, steps, gen_length, block_length, lambd, alpha, baseline_name, temperature, cfg_scale=0., remasking='low_confidence', return_order=True)
    elif mode == 'entropy':
        from src.generate import generate_with_entropy
        _, orders = generate_with_entropy(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence', return_order=True)
    elif mode == 'margin':
        from src.generate import generate_with_margin
        _, orders = generate_with_margin(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence', return_order=True)
    elif mode == 'linear':
        from src.generate import generate_with_linear_position
        _, orders = generate_with_linear_position(model, prompt, steps, gen_length, block_length, lambd, alpha, baseline_name, temperature, cfg_scale=0., remasking='low_confidence', return_order=True)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")
    orders_result = []
    determined = []
    for block_index in orders.keys():
        for step in range(len(orders[block_index])):
            this_step = torch.full((1, gen_length), 0, dtype=torch.float16)
            this_step[0, orders[block_index][step]] = 1
            if determined != []:
                this_step[0, determined] = 1
            orders_result.append(this_step)
            determined.extend([j for j in orders[block_index][step]])
            
    return orders_result

def save_heatmap_params(confidence_result, entropy_result, margin_result, task):
    confidence_result_list = confidence_result.tolist() if isinstance(confidence_result, np.ndarray) else confidence_result
    entropy_result_list = entropy_result.tolist() if isinstance(entropy_result, np.ndarray) else entropy_result
    margin_result_list = margin_result.tolist() if isinstance(margin_result, np.ndarray) else margin_result
    data = {
        "confidence_result": confidence_result_list,
        "entropy_result": entropy_result_list,
        "margin_result": margin_result_list,
    }
    filename = f"heatmap_params/{task}.json"
    if not os.path.exists("heatmap_params"):
        os.makedirs("heatmap_params")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def main(args):
    
    task = args.task
    model_name = args.model_name
    device = args.device
    gen_length = args.gen_length
    steps = args.steps
    block_length = args.block_length
    temperature = args.temperature
    data_path = args.data_path
    samples_num = args.samples_num
    
    confidence_result = []
    entropy_result = []
    margin_result = []
    modes = ['original', 'entropy', 'margin']
    
    dataset = load_dataset(data_path, task)
    samples_num = min(samples_num, len(dataset))
    dataset = dataset[:samples_num]

    print('----------------- Load model -------------------')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    
    print('----------------- Start Answering -------------------')
    
    for mode in modes:
        print(f'----------------- Start Answering with {mode} -------------------')
        for input in tqdm(dataset):
            results = generate(model, tokenizer, input, task, steps, gen_length, block_length, temperature, mode, lambd=0, alpha=0, baseline_name='')
            if mode == 'original':
                if confidence_result == []:
                    confidence_result = results
                else:
                    for i in range(len(results)):
                        confidence_result[i] += results[i]
            elif mode == 'entropy':
                if entropy_result == []:
                    entropy_result = results
                else:
                    for i in range(len(results)):
                        entropy_result[i] += results[i]
            elif mode == 'margin':
                if margin_result == []:
                    margin_result = results
                else:
                    for i in range(len(results)):
                        margin_result[i] += results[i]
                
    print('----------------- Finish Answering -------------------')
    
    confidence_result = np.array(confidence_result) / samples_num
    entropy_result = np.array(entropy_result) / samples_num
    margin_result = np.array(margin_result) / samples_num
    save_heatmap_params(confidence_result, entropy_result, margin_result, task)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='humaneval')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--data_path', type=str, default='./data/humaneval.jsonl')
    parser.add_argument('--samples_num', type=int, default=10)
    args = parser.parse_args()
    main(args)