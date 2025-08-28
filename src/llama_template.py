def countdown_llama_prompt(query):
    user_input = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You are a helpful AI assistant for playing countdown game<|eot_id|><|start_header_id|>user<|end_header_id|>
    
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return user_input

def humaneval_llama_prompt(query):
    user_input = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
You are a helpful AI assistant for python code generation<|eot_id|><|start_header_id|>user<|end_header_id|>
 
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return user_input

def mbpp_llama_prompt(query):
    user_input = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
You are a helpful AI assistant for python code generation<|eot_id|><|start_header_id|>user<|end_header_id|>
 
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return user_input

def math_500_llama_prompt(query):
    user_input = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
You are a helpful AI assistant for solving math problems<|eot_id|><|start_header_id|>user<|end_header_id|>
 
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return user_input

def sudoku_llama_prompt(query):
    user_input = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
 
You are a helpful AI assistant for playing sudoku game<|eot_id|><|start_header_id|>user<|end_header_id|>
 
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    return user_input

def llama_prompt(query, task):
    if task == 'humaneval':
        return humaneval_llama_prompt(query)
    elif task =='mbpp':
        return mbpp_llama_prompt(query)
    elif task =='math500':
        return math_500_llama_prompt(query)
    elif task =='countdown':
        return countdown_llama_prompt(query)
    elif task =='sudoku':
        return sudoku_llama_prompt(query)
    else:
        raise NotImplementedError(f"Mode {task} not implemented.")