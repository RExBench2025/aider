#!/usr/bin/env python3
"""
This script combines existing instructions.md files with file search instructions
to create the final instructions to be provided to the model.
"""

import os
import argparse
import json
import subprocess
from pathlib import Path
import shutil

# File search instruction template
FILE_SEARCH_INSTRUCTION_TEMPLATE = """
## File Search Instructions

Based on the task description above, please list exactly all file paths that need to be reviewed and modified to complete this task. 

IMPORTANT: Your file list MUST include at least one existing file from the repository structure shown above. Do not list only files that need to be created.

For each file, clearly indicate whether it is an existing file to be modified or a new file to be created. Provide file paths relative to the repository root directory, and briefly explain the purpose of each file. Please provide the file list in the following JSON format:

```json
{
  "files": [
    {
      "path": "file path",
      "purpose": "brief description of this file's purpose",
      "exists": true/false (indicate whether this is an existing file or a new file to create)
    }
  ]
}
```
"""

def get_directory_structure(project_dir, max_depth=2):
    """
    Get the directory structure of a project using the tree command.
    
    Args:
        project_dir: Path to the project directory
        max_depth: Maximum depth for the tree command
        
    Returns:
        String containing the directory structure
    """
    try:
        # Check if tree command is available
        result = subprocess.run(['which', 'tree'], capture_output=True, text=True)
        if result.returncode == 0:
            # Use tree command
            cmd = ['tree', '-L', str(max_depth), str(project_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout
        else:
            # Fallback to find command if tree is not available
            cmd = ['find', str(project_dir), '-type', 'd', '-maxdepth', str(max_depth)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return "Directory structure:\n" + result.stdout
    except Exception as e:
        print(f"Warning: Could not get directory structure: {e}")
        return ""

def combine_instructions(original_instruction_path, output_path, model_name=None, project_dir=None, instruction_type="default"):
    """
    Combines existing instruction file with file search instructions to create a new file.
    
    Args:
        original_instruction_path: Path to the original instruction file
        output_path: Path to save the combined instructions
        model_name: Model name (to add to the filename)
        project_dir: Path to the project directory (for directory structure)
        instruction_type: Type of instruction file (default, hints, more_detailed_hints)
    """
    # Read the original instruction file
    with open(original_instruction_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Get directory structure if project_dir is provided
    repo_structure = ""
    if project_dir:
        # Get the src directory if it exists
        src_dir = project_dir / "src"
        if src_dir.exists() and src_dir.is_dir():
            repo_structure = "\n\n## Repository Structure\n\n```\n" + get_directory_structure(src_dir) + "\n```"
        else:
            repo_structure = "\n\n## Repository Structure\n\n```\n" + get_directory_structure(project_dir) + "\n```"
    
    # Add file search instruction with repository structure
    combined_content = original_content + repo_structure + "\n\n" + FILE_SEARCH_INSTRUCTION_TEMPLATE
    
    # Save the combined instruction
    # Replace slashes in model name with underscores to avoid directory path issues
    safe_model_name = model_name.replace('/', '_').replace('\\', '_') if model_name else ""
    # Include the instruction type in the filename
    model_suffix = f"_{instruction_type}_{safe_model_name}" if safe_model_name else f"_{instruction_type}"
    output_file = output_path / f"instructions{model_suffix}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_content)
    
    print(f"Combined instruction saved to {output_file}")
    return output_file

def process_project(project_dir, output_dir, model_name=None, instruction_type="default"):
    """
    Processes the instruction file in the project directory.
    
    Args:
        project_dir: Path to the project directory
        output_dir: Directory to save the combined instructions
        model_name: Model name (to add to the filename)
        instruction_type: Type of instruction file to use (default, hints, more_detailed_hints)
        
    Returns:
        Boolean indicating if the project was processed successfully
    """
    # Extract project name
    project_name = project_dir.name
    
    # Find the appropriate instruction file based on instruction_type
    instruction_filename = f"instructions_{instruction_type}.md"
    instructions_file = project_dir / ".docs" / instruction_filename
    
    # Check if the instruction file exists
    if not instructions_file.exists():
        print(f"Error: {instruction_filename} not found in {project_dir}")
        return None
    
    # Create output directory
    project_output_dir = output_dir / project_name
    os.makedirs(project_output_dir, exist_ok=True)
    
    # Combine instructions with directory structure
    return combine_instructions(instructions_file, project_output_dir, model_name, project_dir, instruction_type)

def main():
    parser = argparse.ArgumentParser(description='Combine original instructions with file search instructions')
    parser.add_argument('--rex-bench-dir', type=str, required=True, 
                        help='Path to rex-bench directory')
    parser.add_argument('--output-dir', type=str, default='combined_instructions',
                        help='Output directory for combined instructions')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name to add to the output filename')
    parser.add_argument('--project', type=str, default=None,
                        help='Specific project to process (if not specified, process all projects)')
    parser.add_argument('--instruction-type', type=str, default='default',
                        help='Type of instruction file to use (default, hints, more_detailed_hints)')
    
    args = parser.parse_args()
    
    rex_bench_dir = Path(args.rex_bench_dir)
    output_dir = Path(args.output_dir)
    
    if not rex_bench_dir.exists() or not rex_bench_dir.is_dir():
        print(f"Error: {rex_bench_dir} does not exist or is not a directory")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process specific project or all projects
    if args.project:
        project_dir = rex_bench_dir / args.project
        if not project_dir.exists() or not project_dir.is_dir():
            print(f"Error: Project {args.project} not found in {rex_bench_dir}")
            return
        
        process_project(project_dir, output_dir, args.model, args.instruction_type)
    else:
        # Find all project directories in the rex-bench directory
        processed_count = 0
        for item in rex_bench_dir.iterdir():
            if item.is_dir() and (item / ".docs").exists():
                print(f"Processing {item.name}...")
                if process_project(item, output_dir, args.model, args.instruction_type):
                    processed_count += 1
        
        print(f"Processed {processed_count} projects")

if __name__ == "__main__":
    main()
