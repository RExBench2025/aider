#!/usr/bin/env python3
"""
This script extracts file list JSON from model responses and saves it.
"""

import os
import re
import json
import argparse
from pathlib import Path

def extract_json_from_response(response_text):
    """
    Extracts file list in JSON format from model response.
    
    Args:
        response_text: The model's response text
    
    Returns:
        Extracted JSON object or None
    """
    # Find JSON code blocks (in ```json ... ``` format)
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, response_text)
    
    if not matches:
        # Find in regular code blocks (in ``` ... ``` format)
        json_pattern = r'```\s*({\s*"files"[\s\S]*?})\s*```'
        matches = re.findall(json_pattern, response_text)
    
    if not matches:
        # Find JSON format directly without code blocks
        json_pattern = r'({[\s\S]*?"files"\s*:\s*\[[\s\S]*?\]\s*})'
        matches = re.findall(json_pattern, response_text)
    
    for match in matches:
        try:
            # Try parsing JSON
            json_obj = json.loads(match)
            # Check if there's a 'files' key
            if "files" in json_obj:
                return json_obj
        except json.JSONDecodeError:
            continue
    
    return None

def process_response_file(response_file, output_dir, model_name, instruction_type=None):
    """
    Processes a response file to extract file list JSON.
    
    Args:
        response_file: Path to the file containing the model's response
        output_dir: Directory to save the extracted JSON
        model_name: Model name
        instruction_type: Type of instruction file used (default, hints, more_detailed_hints)
    
    Returns:
        Path to the extracted JSON file or None
    """
    # Extract project name from filename (simpler approach)
    file_stem = response_file.stem
    parts = file_stem.split("_")
    base_name = parts[0]
    
    known_instruction_types = ["default", "hints", "more_detailed_hints"]
    
    known_projects = [
        "checkeval", "cogs", "entity-tracking-multimodal", "explain-then-translate",
        "implicit-instructions", "mission-impossible", "othello", "re-reading",
        "reasoning-or-reciting", "tree-of-thoughts", "varierr-nli", "winodict"
    ]
    
    # Try to match exact project name
    matched_project = None
    for project in known_projects:
        if base_name.startswith(f"{project}-") or base_name == project:
            matched_project = project
            break
    
    if matched_project:
        # Check if instruction type exists after project name
        if instruction_type and base_name == f"{matched_project}-{instruction_type}":
            project_name = matched_project
        elif instruction_type and base_name.startswith(f"{matched_project}-{instruction_type}-"):
            project_name = matched_project
        else:
            # If instruction type doesn't exist or has a different format
            for instr_type in known_instruction_types:
                if base_name == f"{matched_project}-{instr_type}":
                    project_name = matched_project
                    break
            else:
                # If instruction type wasn't found, just use the matched project
                project_name = matched_project
    else:
        # Use existing logic (if no matching project found)
        if instruction_type and f"-{instruction_type}" in base_name:
            # Remove explicitly specified instruction_type
            project_name = base_name.replace(f"-{instruction_type}", "")
        else:
            # Check for all known instruction types
            found_type = False
            for instr_type in known_instruction_types:
                if base_name.endswith(f"-{instr_type}"):
                    # Remove instruction type from the end
                    project_name = base_name[:-(len(instr_type)+1)]  # +1 for the hyphen
                    found_type = True
                    break
            
            if not found_type:
                project_name = base_name
    
    with open(response_file, 'r', encoding='utf-8') as f:
        response_text = f.read()
    
    json_obj = extract_json_from_response(response_text)
    if not json_obj:
        print(f"Warning: No valid JSON found in {response_file}")
        return None
    
    if model_name == "o1":
        model_short_name = "o1"
    elif "claude" in model_name.lower():
        model_short_name = "claude"
    elif "gemini" in model_name.lower():
        model_short_name = "gemini"
    elif "deepseek" in model_name.lower() or "together" in model_name.lower():
        model_short_name = "deepseek"
    else:
        model_short_name = model_name.replace("/", "_").replace(":", "_")
    
    if str(output_dir).endswith('file_lists'):
        model_output_dir = output_dir / model_short_name
    elif 'benchmark' in str(output_dir):
        # If it's in benchmark directory, use direct path to avoid duplication
        model_output_dir = Path(str(output_dir).split('benchmark')[0]) / "benchmark/file_lists" / model_short_name
    else:
        # Otherwise, create the full path
        model_output_dir = output_dir / "file_lists" / model_short_name
    
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Output file in model-specific directory with instruction type
    # Use the instruction_type provided as parameter
    if not instruction_type:
        instruction_type = "default"
    
    output_file = model_output_dir / f"{project_name}-{instruction_type}.json"
    
    if output_file.exists():
        print(f"File already exists: {output_file}. Skipping...")
        return output_file
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Create the nested structure that benchmark.py expects - only use model-specific key
        json_data = {
            model_short_name: {
                project_name: {
                    "files": json_obj.get("files", [])
                }
            }
            # No default key as per user request
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted file list saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Extract file list JSON from model responses')
    parser.add_argument('--response-dir', type=str, required=True, 
                        help='Directory containing model responses')
    parser.add_argument('--output-dir', type=str, default='file_lists',
                        help='Output directory for extracted JSON files')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name that generated the responses')
    parser.add_argument('--instruction-type', type=str, default='default',
                        help='Type of instruction file used (default, hints, more_detailed_hints)')
    
    args = parser.parse_args()
    
    response_dir = Path(args.response_dir)
    output_dir = Path(args.output_dir)
    
    if not response_dir.exists() or not response_dir.is_dir():
        print(f"Error: {response_dir} does not exist or is not a directory")
        return
    
    # Replace slashes in model name with underscores for safe filename
    safe_model_name = args.model.replace('/', '_').replace('\\', '_')
    model_prefixes = {
        'claude': ['anthropic', 'claude'],
        'openai': ['openai', 'gpt', 'o1'],
        'gemini': ['gemini', 'google'],
        'together': ['together', 'deepseek']
    }
    
    current_model_prefixes = []
    for key, prefixes in model_prefixes.items():
        if any(prefix in args.model.lower() for prefix in prefixes):
            current_model_prefixes = prefixes
            break
    
    if not current_model_prefixes:
        current_model_prefixes = [safe_model_name.lower()]
    
    print(f"Using model prefixes for filtering: {current_model_prefixes}")
    
    processed_count = 0
    skipped_count = 0
    
    response_files = list(response_dir.glob("*.md"))
    
    for prefix in current_model_prefixes:
        model_specific_dir = response_dir / prefix
        if model_specific_dir.exists() and model_specific_dir.is_dir():
            response_files.extend(model_specific_dir.glob("*.md"))
    
    model_dir = response_dir / safe_model_name
    if model_dir.exists() and model_dir.is_dir():
        response_files.extend(model_dir.glob("*.md"))
    
    for response_file in response_files:
        file_name = response_file.name.lower()
        
        if any(prefix in file_name for prefix in current_model_prefixes) or safe_model_name.lower() in file_name:
            print(f"Processing matching file: {response_file.name}...")
            if process_response_file(response_file, output_dir, args.model, args.instruction_type):
                processed_count += 1
        elif any(prefix in file_name for prefix in [p for prefixes in model_prefixes.values() for p in prefixes if p not in current_model_prefixes]):
            print(f"Skipping file for different model: {response_file.name}")
            skipped_count += 1
        else:
            print(f"Processing generic file: {response_file.name}...")
            if process_response_file(response_file, output_dir, args.model, args.instruction_type):
                processed_count += 1
    
    print(f"Processed {processed_count} response files, skipped {skipped_count} files for other models")

if __name__ == "__main__":
    main()
