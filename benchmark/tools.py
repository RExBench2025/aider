from typing import Optional, List
from pathlib import Path
import os

class BenchmarkTool:
    """Tool for running Aider benchmark tests using benchmark.py"""
    
    def func(self, command: str, *args, **kwargs) -> str:
        """
        Run benchmark.py commands.
        Example commands:
        - python benchmark.py dir1 dir2 --model gpt-4
        - python benchmark.py --stats dir1
        - python benchmark.py --diffs dir1 dir2
        """
        try:
            import subprocess
            result = subprocess.run(
                ["python", "benchmark.py"] + command.split(),
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent)
            )
            return result.stdout if result.stdout else result.stderr
        except Exception as e:
            return f"Error running benchmark: {str(e)}"

class FileReadTool:
    """Tool for reading file contents"""
    
    def func(self, file_path: str) -> str:
        """Read and return the contents of a file"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

class RelatedFilesTool:
    """Tool for finding related files"""
    
    def func(self, file_path: str) -> List[str]:
        """Find files related to the given file"""
        try:
            directory = os.path.dirname(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            related_files = []
            
            for root, _, files in os.walk(directory):
                for file in files:
                    if base_name in file:
                        related_files.append(os.path.join(root, file))
            
            return related_files
        except Exception as e:
            return [f"Error finding related files: {str(e)}"]
