#!/usr/bin/env python3
"""
This script runs the file list extraction pipeline:
1. Combines original instructions with file search instructions.
2. Generates model responses using the combined instructions (can be automated or manual).
3. Extracts file list JSON from model responses.
"""

import os
import argparse
import subprocess
from pathlib import Path
import sys

def run_command(command):
    """
    Executes a command and returns the result.
    
    Args:
        command: List containing the command to execute
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description='Run file extraction pipeline')
    parser.add_argument('--rex-bench-dir', type=str, required=True, 
                        help='Path to rex-bench directory')
    parser.add_argument('--output-dir', type=str, default='file_extraction_output',
                        help='Base output directory')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('--project', type=str, default=None,
                        help='Specific project to process (if not specified, process all projects)')
    parser.add_argument('--response-dir', type=str, default=None,
                        help='Directory containing model responses (if already generated)')
    parser.add_argument('--auto-generate', action='store_true',
                        help='Automatically generate model responses using API')
    parser.add_argument('--api', type=str, default='claude',
                        choices=['claude', 'openai', 'gemini', 'together'],
                        help='API to use for auto-generation (claude, openai, gemini, or together)')
    parser.add_argument('--instruction-type', type=str, default='default',
                        help='Type of instruction file to use (default, hints, more_detailed_hints)')
    
    args = parser.parse_args()
    
    # Set up base directories
    base_dir = Path(args.output_dir)
    
    # 모델 이름에서 슬래시(/) 제거하여 디렉토리 이름으로 사용 가능하게 함
    safe_model_name = args.model.replace('/', '_')
    
    # deepseek 모델인 경우 'deepseek' 디렉토리 사용
    if 'deepseek' in args.model.lower() or 'together' in args.model.lower():
        model_dir_name = 'deepseek'
    else:
        model_dir_name = safe_model_name
    
    combined_dir = base_dir / "combined_instructions"
    response_dir = Path(args.response_dir) if args.response_dir else base_dir / "model_responses"
    file_lists_dir = base_dir / "file_lists" / model_dir_name
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(response_dir, exist_ok=True)
    os.makedirs(file_lists_dir, exist_ok=True)
    
    # Script paths
    script_dir = Path(__file__).parent
    combine_script = script_dir / "combine_instructions.py"
    generate_script = script_dir / "generate_model_responses.py"
    extract_script = script_dir / "extract_model_responses.py"
    
    # 1. Combine instructions
    print("=== 1. Combining instructions ===")
    combine_cmd = [
        "python", str(combine_script),
        "--rex-bench-dir", args.rex_bench_dir,
        "--output-dir", str(combined_dir),
        "--model", args.model,
        "--instruction-type", args.instruction_type
    ]
    
    if args.project:
        combine_cmd.extend(["--project", args.project])
    
    return_code, stdout, stderr = run_command(combine_cmd)
    if return_code != 0:
        print(f"Error combining instructions: {stderr}")
        return
    
    print(stdout)
    
    # 2. Generate model responses (automated or manual)
    if args.auto_generate:
        print("\n=== 2. Generating model responses automatically ===")
        # Check if required environment variables are set
        if args.api == "claude" and "ANTHROPIC_API_KEY" not in os.environ:
            print("Error: ANTHROPIC_API_KEY environment variable is not set")
            print("Please set it with: export ANTHROPIC_API_KEY=your_api_key")
            return
        elif args.api == "openai" and "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable is not set")
            print("Please set it with: export OPENAI_API_KEY=your_api_key")
            return
        elif args.api == "gemini" and "GEMINI_API_KEY" not in os.environ:
            print("Error: GEMINI_API_KEY environment variable is not set")
            print("Please set it with: export GEMINI_API_KEY=your_api_key")
            return
        elif args.api == "together" and "TOGETHER_API_KEY" not in os.environ:
            print("Error: TOGETHER_API_KEY environment variable is not set")
            print("Please set it with: export TOGETHER_API_KEY=your_api_key")
            return
            
        generate_cmd = [
            "python", str(generate_script),
            "--instructions-dir", str(combined_dir),
            "--output-dir", str(response_dir),
            "--model", args.model,
            "--api", args.api,
            "--instruction-type", args.instruction_type
        ]
        
        return_code, stdout, stderr = run_command(generate_cmd)
        if return_code != 0:
            print(f"Error generating model responses: {stderr}")
            return
        
        print(stdout)
        print(f"Model responses have been generated and saved to {response_dir}")
        
        # Set response_dir for the next step
        response_dir_to_use = response_dir
    else:
        print("\n=== 2. Generate model responses (manual step) ===")
        print(f"Please use the combined instructions in {combined_dir} to generate responses with {args.model}.")
        print(f"Save the responses in a directory and provide the path with --response-dir in the next run.")
        
        # If response directory is not provided, stop here
        if not args.response_dir:
            print("\nTo extract file lists from model responses, run this script again with --response-dir or --auto-generate.")
            return
        
        response_dir_to_use = Path(args.response_dir)
    
    # 3. Extract file lists from model responses
    print("\n=== 3. Extracting file lists from model responses ===")
    extract_cmd = [
        "python", str(extract_script),
        "--response-dir", str(response_dir_to_use),
        "--output-dir", str(file_lists_dir),
        "--model", args.model,
        "--instruction-type", args.instruction_type
    ]
    
    return_code, stdout, stderr = run_command(extract_cmd)
    if return_code != 0:
        print(f"Error extracting file lists: {stderr}")
        return
    
    print(stdout)
    print(f"\nFile lists have been extracted to {file_lists_dir}")
    print("These files can now be used by benchmark.py for model-specific file selection.")

if __name__ == "__main__":
    main()
