import json
import os
import math
import time
from typing import List, Dict, Counter, Tuple, Optional
from collections import Counter
from queue import Queue
import concurrent.futures
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
except ImportError:
    print("Error: transformers library is not installed.")
    print("Please run 'pip install transformers' to install it.")
    exit()

# --- Smart Chunk Configuration ---
# Multiplier for tasks per CPU core. For example, 4 means 4 tasks per core to keep CPUs busy
CHUNKS_PER_WORKER = 4
# Minimum and maximum chunk sizes to prevent extreme cases with very small or large files
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 20000

def process_chunk(
    lines_chunk: List[str], 
    tokenizer: PreTrainedTokenizer, 
    keys_to_process: List[str], 
    pos_queue: Queue
) -> Tuple[Counter[int], int]:
    """
    Process a chunk of lines from the JSONL file
    
    Args:
        lines_chunk: List of lines from the JSONL file
        tokenizer: Hugging Face tokenizer to use for encoding
        keys_to_process: List of JSON keys to extract text from
        pos_queue: Queue to manage progress bar positions
        
    Returns:
        Tuple containing:
            - Counter with token counts for this chunk
            - Total number of tokens in this chunk
    """
    position = pos_queue.get()
    try:
        local_token_counts = Counter()
        local_total_tokens = 0
        
        # Progress bar for this worker
        pbar = tqdm(
            lines_chunk, 
            desc=f'Worker {position:02d}', 
            position=position,
            leave=False,
            dynamic_ncols=True
        )
        
        for line in pbar:
            try:
                # Parse JSON object
                data_obj = json.loads(line)
                
                # Combine text from specified keys
                combined_text = "".join(
                    f"{data_obj[key]} " for key in keys_to_process 
                    if key in data_obj and isinstance(data_obj[key], str)
                )
                
                if combined_text:
                    # Encode text to token IDs
                    token_ids = tokenizer.encode(combined_text, add_special_tokens=False)
                    local_token_counts.update(token_ids)
                    local_total_tokens += len(token_ids)
                    
            except json.JSONDecodeError:
                pbar.write("Warning: Could not decode JSON line")
            except Exception as e:
                pbar.write(f"Warning: Error processing line - {str(e)}")
                
    finally:
        pos_queue.put(position)
        
    return local_token_counts, local_total_tokens

def calculate_token_frequencies() -> None:
    """
    Interactive script to calculate token frequencies in JSONL datasets with parallel processing.
    Automatically adjusts chunk sizes for efficient processing and supports resuming from previous runs.
    """
    print("--- Token Frequency Calculator (Smart Chunking Version) ---")

    # --- 1. Get User Inputs ---
    model_name = input("Enter Hugging Face model name (e.g., 'bert-base-uncased'): ")
    jsonl_file_path = input("Enter full path to the dataset JSONL file: ")
    keys_to_process_str = input("Enter keys to process, separated by commas (e.g., 'prompt,response'): ")
    output_json_path = input("Enter path to save results JSON file (will accumulate if exists): ")
    
    default_workers = os.cpu_count() or 4
    num_workers_str = input(f"Enter number of workers (default: {default_workers}): ")
    
    try:
        num_workers = int(num_workers_str) if num_workers_str else default_workers
    except ValueError:
        print(f"Invalid input, using default number of workers: {default_workers}")
        num_workers = default_workers

    # --- 2. Load Tokenizer ---
    try:
        print(f"\nLoading tokenizer: '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"\nError: Failed to load tokenizer '{model_name}'. Please check the model name.\nDetails: {e}")
        return

    # --- 3. Initialize or Load Historical Statistics ---
    total_tokens: int = 0
    token_counts: Counter[int] = Counter()
    
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                existing_data: Dict = json.load(f)
                
                if "num_token" in existing_data and "token_frequency_dict" in existing_data:
                    total_tokens = existing_data.get("num_token", 0)
                    raw_counts = existing_data.get("token_frequency_dict", {})
                    token_counts = Counter({int(k): v for k, v in raw_counts.items()})
                    print(f"\nSuccessfully loaded historical data. Current total tokens: {total_tokens:,}, Token types: {len(token_counts):,}")
                else:
                    print(f"\nWarning: '{output_json_path}' has incorrect format. Starting fresh.")
                    
        except json.JSONDecodeError:
            print(f"\nWarning: Could not parse '{output_json_path}'. Starting fresh.")
        except Exception as e:
            print(f"\nWarning: Could not read historical file '{output_json_path}'. Starting fresh. Error: {e}")
    else:
        print("\nNo historical statistics found. Starting new calculation.")
    
    # --- 4. Smart Chunk Sizing and Parallel Processing ---
    keys_to_process = [key.strip() for key in keys_to_process_str.split(',')]
    
    try:
        # 4.1: Pre-calculate total number of lines
        print("\nCounting total lines to determine optimal chunk size...")
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in tqdm(f, desc="Counting lines"))
        
        if total_lines == 0:
            print("\nError: The input file is empty. No processing needed.")
            return
            
        # 4.2: Calculate optimal chunk size
        target_num_chunks = num_workers * CHUNKS_PER_WORKER
        calculated_chunk_size = math.ceil(total_lines / target_num_chunks)
        chunk_size = max(MIN_CHUNK_SIZE, min(calculated_chunk_size, MAX_CHUNK_SIZE))

        print(f"Total lines in file: {total_lines:,}")
        print(f"Using {num_workers} workers. Automatically calculated chunk size: {chunk_size} lines.")
        time.sleep(2)  # Pause to let user read the information

        # Prepare queue for progress bar positions
        pos_queue = Queue()
        for i in range(num_workers):
            pos_queue.put(i + 1)

        # 4.3: Process file in chunks using parallel workers
        with open(jsonl_file_path, 'r', encoding='utf-8') as f, \
             concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            
            futures = []
            current_chunk: List[str] = []
            
            for line in f:
                current_chunk.append(line)
                if len(current_chunk) == chunk_size:
                    futures.append(executor.submit(
                        process_chunk, 
                        current_chunk, 
                        tokenizer, 
                        keys_to_process, 
                        pos_queue
                    ))
                    current_chunk = []
            
            # Process remaining lines in the last chunk
            if current_chunk:
                futures.append(executor.submit(
                    process_chunk, 
                    current_chunk, 
                    tokenizer, 
                    keys_to_process, 
                    pos_queue
                ))

            print(f"\nAll chunks ({len(futures)}) submitted for processing...")
            
            # Main progress bar for overall progress
            main_pbar = tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures), 
                desc="Overall Progress",
                position=0,
                dynamic_ncols=True
            )

            for future in main_pbar:
                try:
                    local_counts, local_total = future.result()
                    token_counts.update(local_counts)
                    total_tokens += local_total
                except Exception as e:
                    main_pbar.write(f"Error processing chunk: {str(e)}")

    except FileNotFoundError:
        print(f"\nError: Dataset file '{jsonl_file_path}' not found. Please check the path.")
        return
    except Exception as e:
        print(f"\nUnexpected error during processing.\nDetails: {str(e)}")
        return

    # --- 5. Save Updated Results ---
    print("\nProcessing complete. Preparing to save results...")
    final_output: Dict[str, object] = {
        "num_token": total_tokens,
        "token_frequency_dict": {str(token_id): count for token_id, count in token_counts.items()},
        "metadata": {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_used": model_name,
            "processed_keys": keys_to_process
        }
    }
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        
        print(f"\n🎉 Results successfully saved to: '{output_json_path}'")
        print(f"Final total tokens: {total_tokens:,}")
        print(f"Final token types: {len(token_counts):,}")
        
    except Exception as e:
        print(f"\nError: Failed to save output file.\nDetails: {str(e)}")

if __name__ == '__main__':
    calculate_token_frequencies()
