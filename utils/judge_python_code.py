import os
import subprocess
import concurrent.futures
import sys
from pathlib import Path
import argparse

def run_python_file(file_path):
    try:
        compile(open(file_path, 'rb').read(), file_path, 'exec')
        result = subprocess.run(
            [sys.executable, file_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stderr:
            return file_path, "RuntimeError", result.stderr.strip()
        else:
            return file_path, "Success", result.stdout.strip()
    except SyntaxError as e:
        return file_path, "SyntaxError", str(e)
    except subprocess.TimeoutExpired:
        return file_path, "Timeout", "Execution timed out"
    except Exception as e:
        return file_path, "Error", str(e)

def main():
    parser = argparse.ArgumentParser(description='Run Python files in a folder and save results.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing Python files')
    parser.add_argument('--output_path', type=str, help='Path to the TXT file to save results')
    args = parser.parse_args()

    py_files = [str(p) for p in Path(args.folder_path).rglob("*.py")]
    total_files = len(py_files)

    if not py_files:
        print("No Python files found!")
        return

    print(f"Found {total_files} Python files, starting execution...")

    success_count = 0
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        future_to_file = {executor.submit(run_python_file, f): f for f in py_files}

        for future in concurrent.futures.as_completed(future_to_file):
            file_path, status, message = future.result()
            results.append((file_path, status, message))

            if status == "Success":
                success_count += 1

            print(f"{file_path}: {status}")

    accuracy = (success_count / total_files) * 100 if total_files > 0 else 0

    with open(args.output_path, 'w', encoding='utf-8') as f:
        f.write("=== Execution Results Summary ===\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Successful files: {success_count}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("=== Detailed Results ===\n")
        for file_path, status, message in results:
            f.write(f"File: {file_path}\n")
            f.write(f"Status: {status}\n")
            if message:
                f.write(f"Message: {message}\n")
            f.write("-" * 40 + "\n")

    print(f"Execution completed! Results saved to {args.output_path}")
    print(f"Overall accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()