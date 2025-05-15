#!/usr/bin/env python
import os
import shutil
from pathlib import Path
import git
import typer

app = typer.Typer()

def setup_test_directory(base_dir: Path, test_name: str, prompt: str, repo_url: str, branch: str = "main"):
    """
    Set up a test directory with prompt and original code
    """
    test_dir = base_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save prompt
    with open(test_dir / "prompt.md", "w") as f:
        f.write(prompt)
    
    # Clone original repo
    original_dir = test_dir / "original"
    if original_dir.exists():
        shutil.rmtree(original_dir)
    
    repo = git.Repo.clone_from(repo_url, original_dir)
    if branch != "main":
        repo.git.checkout(branch)

    # Create directory for expected results
    expected_dir = test_dir / "expected"
    expected_dir.mkdir(exist_ok=True)

@app.command()
def create_dataset(
    dataset_dir: str = typer.Argument(..., help="Directory to create dataset in"),
    prompts_file: str = typer.Argument(..., help="File containing prompts (JSON or JSONL)"),
    repo_url: str = typer.Argument(..., help="URL of the repository to test"),
):
    """
    Create a new dataset from prompts and repository
    """
    import json
    
    base_dir = Path(dataset_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    with open(prompts_file) as f:
        if prompts_file.endswith('.jsonl'):
            prompts = [json.loads(line) for line in f]
        else:
            prompts = json.load(f)
    
    # Create test directories
    for i, prompt_data in enumerate(prompts):
        test_name = f"test_{i:03d}"
        if isinstance(prompt_data, dict):
            prompt = prompt_data['prompt']
            branch = prompt_data.get('branch', 'main')
        else:
            prompt = prompt_data
            branch = 'main'
            
        setup_test_directory(base_dir, test_name, prompt, repo_url, branch)
    
    print(f"Created dataset with {len(prompts)} tests in {base_dir}")

if __name__ == "__main__":
    app()
