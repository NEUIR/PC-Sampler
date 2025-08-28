import json
import os

def load_json_or_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} not found")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            f.seek(0)
            data = json.load(f)
            if isinstance(data, list):
                result = {}
                current_key = 1
                for item in data:
                    result[current_key] = item
                    current_key += 1
                return result
            else:
                return data 
        except json.JSONDecodeError:
            pass
        result = {}
        f.seek(0)
        current_key = 1    
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    for item in parsed:
                        result[current_key] = item
                        current_key += 1
                else:
                    result[current_key] = parsed
                    current_key += 1
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing error on line {line_num}: {e}")
    return result