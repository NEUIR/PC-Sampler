import os, json, re, csv, sys
from pathlib import Path
from typing import Dict, List, Optional
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.template import *
from utils.load_json_or_jsonl import load_json_or_jsonl

def query_extract(input, task):
    if task == 'humaneval':
        return humaneval_prompt(input['prompt'])
    elif task == 'mbpp':
        return mbpp_prompt(input['prompt'], input['code'])
    elif task == 'math500':
        return math_500_prompt(input['problem'])
    elif task =='countdown':
        return countdown_prompt(input['input'])
    elif task =='sudoku':
        return sudoku_prompt(input['Puzzle'])
    elif task =='gsm8k':
        return gsm8k_prompt(input['question'])
    elif task =='gpqa':
        return gpqa_prompt(input['question'], input['correct_answer'], input['option_A'], input['option_B'], input['option_C'])
    else:
        raise NotImplementedError(f"Mode {task} not implemented.")
    
def load_dataset(data_path, task):
    if task == 'sudoku':
        dataset = load_sudoku_dataset(data_path)
        if not dataset:
            raise ValueError(f"Error: Dataset file '{data_path}' not found.")
        return dataset
    else:
        data_json = load_json_or_jsonl(data_path)
        dataset = []
        for key in data_json.keys():
            dataset.append(data_json[key])
        return dataset
    
    

def countdown_check(model_answer, ground_truth):
    if ground_truth in model_answer:
        return True
    else:
        return False

def eval_countdown(results, dataset, result_path, args):
    true_num = 0
    for index, answer in enumerate(results):
        result = dataset[index]
        if countdown_check(answer, result['output']):
            true_num += 1

    print('----------------- Finish Answering -------------------')


    with open(result_path, 'a', encoding='utf-8') as file:

        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n\n")

        file.write(f"Total Accuracy: {true_num / len(dataset)}\n")
        file.write("\n\n")
        
        

def eval_humaneval(results, dataset, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    for index in range(len(results)):
        answer = results[index]
        answer = answer.replace('```python', '\n').replace('```', '\n')
        answer = dataset[index]['prompt'].replace(dataset[index]['entry_point'], dataset[index]['entry_point']+'_prompt') + answer
        results[index] = answer
    
    for index, answer in enumerate(results):
        code_path = f'{result_dir}/{index + 1}.py'
        with open(code_path, 'w', encoding='utf-8') as file:
            file.write(answer + '\n')
            file.write(dataset[index]['test'] + '\n')
            file.write('if __name__ == "__main__":\n')
            file.write(f'    check({dataset[index]["entry_point"]})')
            
            
            
def generate_mbpp_test_files(
    samples: List[Dict],
    model_outputs: List[str],
    output_dir: Path,
    template_path: Optional[Path] = None,
    prefix: str = "test_index_"
) -> List[Path]:

    if len(samples) != len(model_outputs):
        raise ValueError("The number of samples and model outputs must be the same")

    default_template = """\"\"\"
Test file for task_id: {task_id}
Problem description: {text}
\"\"\"

{setup_code}

{model_code}

{test_code}
"""
    template = default_template
    if template_path and template_path.exists():
        template = template_path.read_text()
    output_paths = []
    for i, (sample, model_code) in enumerate(zip(samples, model_outputs)):
        if isinstance(sample, str):
            sample = json.loads(sample)
        required_fields = ["prompt", "task_id", "test_list"]
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Sample is missing required field: {field}")
        task_id = sample["task_id"]
        try:
            extracted_func = model_code.split('```python')[1].split('```')[0]
            extracted_func = extracted_func.replace('```python', '\n').replace('```', '\n')
        except:
            extracted_func = model_code.replace('```python', '\n').replace('```', '\n')

        test_code = "\n\n".join(sample["test_list"])
        if "challenge_test_list" in sample:
            test_code += "\n\n" + "\n\n".join(sample["challenge_test_list"])
        test_file_content = template.format(
            task_id=task_id,
            text=sample["prompt"],
            setup_code=sample.get("test_setup_code", ""),
            model_code=extracted_func,
            test_code=test_code
        )
        output_path = output_dir / f"{prefix}{task_id}.py"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(test_file_content)
        output_paths.append(output_path)
    return output_paths

def eval_mbpp(results, dataset, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    generate_mbpp_test_files(dataset, results, Path(result_dir))
    


def collect_answer_from_response(response):
    regex_list = [r"boxed{(.*)}","framebox{(.*)}"]
    _res = ""
    try:
        for regex in regex_list:
            _res = re.findall(regex, response, flags=re.MULTILINE)
            _res = _res[-1] if _res and len(_res)>0 else ""
            if _res != "":
                break
    except Exception:
        pass
    _res = _res.strip('.')
    return _res

def eval_math500(results, dataset, result_path, args):
    true_num = 0
    for index, answer in enumerate(results):
        if dataset[index]['answer'] in collect_answer_from_response(answer):
            true_num += 1

    print('----------------- Finish Answering -------------------')

    with open(result_path, 'a', encoding='utf-8') as file:

        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n\n")

        file.write(f"Total Accuracy: {true_num / len(dataset)}\n")
        file.write("\n\n")
        
        
        
def load_sudoku_dataset(file_path: str) -> List[Dict[str, str]]:
    dataset = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dataset.append(row)
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
    return dataset

def check_solution(prediction: str, ground_truth: str) -> bool:
    match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
    if not match:
        return ground_truth in prediction.replace(" ", "").replace("\n", "")
    solution_part = match.group(1).strip()
    return solution_part == ground_truth

def eval_sudoku(results, dataset, result_path, args):
    true_num = 0
    for index, answer in enumerate(results):
        puzzle_data = dataset[index]
        if check_solution(answer, puzzle_data['Solution']):
            true_num += 1

    print('----------------- Finish Answering -------------------')
    
    accuracy = true_num / len(dataset)
    print(f"Final Accuracy: {accuracy:.4f} ({true_num}/{len(dataset)})")

    with open(result_path, 'a', encoding='utf-8') as file:
        file.write("----------------- Args Configuration -------------------\n")
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
        file.write("\n")
        
        file.write(f"Total Accuracy: {accuracy}\n")
        file.write("\n\n")
        


def eval(task, results, dataset, result_path, args):
    if task == 'humaneval':
        eval_humaneval(results, dataset, result_path)
    elif task == 'mbpp':
        eval_mbpp(results, dataset, result_path)
    elif task == 'math500':
        eval_math500(results, dataset, result_path, args)
    elif task =='countdown':
        eval_countdown(results, dataset, result_path, args)
    elif task =='sudoku':
        eval_sudoku(results, dataset, result_path, args)
    else:
        raise NotImplementedError(f"Mode {task} not implemented.")