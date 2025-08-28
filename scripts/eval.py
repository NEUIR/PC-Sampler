import random, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import argparse
from tqdm import tqdm
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.eval_utils import query_extract, load_dataset, eval
from src.llama_template import llama_prompt

random.seed(42)

def generate(model, tokenizer, input, task, steps, gen_length, block_length, temperature, mode, lambd, alpha, baseline_name, thread, gamma):

    query = query_extract(input, task)
    m = [{"role": "user", "content": query}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    prompt = tokenizer(user_input)['input_ids']
    prompt = torch.tensor(prompt).to(model.device).unsqueeze(0)
    if mode == 'original':
        from src.generate import generate
        out = generate(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence')
    elif mode == 'pc_sampler':
        from src.generate import generate_with_pc_sampler
        out = generate_with_pc_sampler(model, prompt, steps, gen_length, block_length, lambd, alpha, baseline_name, temperature, cfg_scale=0., remasking='low_confidence')
    elif mode == 'eb_sampler':
        from src.generate import generate_with_eb_sampler
        out = generate_with_eb_sampler(model, prompt, gamma, gen_length, temperature, cfg_scale=0.)
    elif mode == 'fast_dllm':
        from src.generate import generate_with_fast_dllm
        out = generate_with_fast_dllm(model, prompt, steps, gen_length, block_length, temperature, remasking='low_confidence', threshold=thread)[0]
    elif mode == 'entropy':
        from src.generate import generate_with_entropy
        out = generate_with_entropy(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence')
    elif mode == 'margin':
        from src.generate import generate_with_margin
        out = generate_with_margin(model, prompt, steps, gen_length, block_length, temperature, cfg_scale=0., remasking='low_confidence')
    elif mode == 'linear':
        from src.generate import generate_with_linear_position
        out = generate_with_linear_position(model, prompt, steps, gen_length, block_length, lambd, alpha, baseline_name, temperature, cfg_scale=0., remasking='low_confidence')
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")
    
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    return answer

def llama_generate(model, tokenizer, input, task, gen_length):

    query = query_extract(input, task)
    user_input = llama_prompt(query, task)
    input_ids = tokenizer(user_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)
    prompt = input_ids
    out = model.generate(inputs=prompt, max_length=gen_length+prompt.shape[1], do_sample=False)
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    return answer

def mistral_generate(model, tokenizer, input, task, gen_length):

    query = query_extract(input, task)
    conversation = [{"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt = inputs['input_ids']
    out = model.generate(**inputs, max_new_tokens=gen_length, do_sample=False)
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    if task == 'mbpp':
        answer = answer.split('```python')[1].split('```')[0]
        answer = answer.replace('```python', '\n').replace('```', '\n')
        answer = input['prompt'].replace(input['entry_point'], input['entry_point']+'_prompt') + answer
    return answer

def Qwen_25_generate(model, tokenizer, input, task, gen_length):

    query = query_extract(input, task)
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(**model_inputs, max_new_tokens=gen_length, do_sample=False)
    prompt = model_inputs['input_ids']
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    if task =='mbpp':
        answer = answer.replace('```python', '\n').replace('```', '\n')
        answer = input['prompt'].replace(input['entry_point'], input['entry_point']+'_prompt') + answer
    return answer

def main(args):

    task = args.task
    model_name = args.model_name
    device = args.device
    gen_length = args.gen_length
    steps = args.steps
    block_length = args.block_length
    temperature = args.temperature
    mode = args.mode
    lambd = args.lambd
    alpha = args.alpha
    baseline_name = args.baseline_name
    thread = args.thread
    gamma = args.gamma
    data_path = args.data_path
    result_path = args.result_path

    dataset = load_dataset(data_path, task)

    print('----------------- Load model -------------------')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if 'LLaDA' in model_name:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
        
    print('----------------- Start Answering -------------------')
    
    results = []
    for input in tqdm(dataset):
        
        if 'llama' in model_name:
            answer = llama_generate(model, tokenizer, input, task, gen_length)
        elif 'mistral' in model_name:
            answer = mistral_generate(model, tokenizer, input, task, gen_length)
        elif 'Qwen2.5' in model_name:
            answer = Qwen_25_generate(model, tokenizer, input, task, gen_length)
        else:
            answer = generate(model, tokenizer, input, task, steps, gen_length, block_length, temperature, mode, lambd, alpha, baseline_name, thread, gamma)
        results.append(answer)

    eval(task, results, dataset, result_path, args)
    
    print('----------------- Finish -------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='humaneval')
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--mode', type=str, default='original')
    parser.add_argument('--lambd', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=10)
    parser.add_argument('--baseline_name', type=str, default='../data/baseline/reference_corpus.json')
    parser.add_argument('--thread', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--data_path', type=str, default='./data/humaneval.jsonl')
    parser.add_argument('--result_path', type=str, default='../results/humaneval_results')
    args = parser.parse_args()
    main(args)