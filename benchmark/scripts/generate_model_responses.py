#!/usr/bin/env python3
"""
This script automatically generates model responses for combined instructions
using the specified model API.
"""

import os
import argparse
import json
import time
import requests
import traceback
from pathlib import Path
import anthropic
import google.generativeai as genai
from google import genai as google_genai
from google.genai import types

try:
    from together import Together
except ImportError:
    Together = None

def generate_claude_response(instruction_content, model_name="claude-3-7-sonnet-20250219"):
    """
    Generate a response using Claude API.
    
    Args:
        instruction_content: The instruction content to send to the model
        model_name: Claude model name to use
    
    Returns:
        The model's response text
    """
    # Initialize Claude client
    # You need to set ANTHROPIC_API_KEY environment variable
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    
    # Remove 'anthropic/' prefix if present
    if model_name.startswith("anthropic/"):
        model_name = model_name.replace("anthropic/", "")
    
    client = anthropic.Anthropic()
    
    # Create a message
    message = client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0.7,
        system="You are a helpful AI assistant that analyzes code repositories and identifies important files.",
        messages=[
            {"role": "user", "content": instruction_content}
        ]
    )
    
    # Return the response content
    return message.content[0].text

def generate_openai_response(instruction_content, model_name="o1"):
    """
    Generate a response using OpenAI API.
    
    Args:
        instruction_content: The instruction content to send to the model
        model_name: OpenAI model name to use
    
    Returns:
        The model's response text
    """
    # You need to set OPENAI_API_KEY environment variable
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    import openai
    client = openai.OpenAI()
    
    # Create chat completion parameters
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant that analyzes code repositories and identifies important files."},
            {"role": "user", "content": instruction_content}
        ]
    }
    
    # Add temperature parameter only if not using o1 or o4 models
    if "o1" not in model_name and "o4" not in model_name:
        params["temperature"] = 0.7
    
    # Create a chat completion
    response = client.chat.completions.create(**params)
    
    # Return the response content
    return response.choices[0].message.content

def generate_gemini_response(instruction_content, model_name="gemini-2.5-pro-preview-03-25", max_retries=3, timeout=60):
    """
    Generate a response using Google Gemini API.
    
    Args:
        instruction_content: The instruction content to send to the model
        model_name: Gemini model name to use
        max_retries: Maximum number of retries on failure
        timeout: Timeout in seconds for the API call
    
    Returns:
        The model's response text
    """
    # You need to set GEMINI_API_KEY environment variable
    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    # Remove 'gemini/' prefix if present
    if model_name.startswith("gemini/"):
        model_name = model_name.replace("gemini/", "")
    
    # Initialize the Gemini client with the API key - test_gemini.py와 동일한 방식 사용
    client = google_genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    # Add system prompt
    system_prompt = "You are a helpful AI assistant that analyzes code repositories and identifies important files."
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n{instruction_content}"
            )
            
            # Return the response content
            return response.text
        
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt * 1 second
                wait_time = (2 ** attempt) * 1
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed. Last error: {str(e)}")
                print(traceback.format_exc())
                raise

def generate_together_response(instruction_content, model_name="deepseek-ai/DeepSeek-R1", max_retries=3, timeout=60):
    """
    Generate a response using Together API.
    
    Args:
        instruction_content: The instruction content to send to the model
        model_name: Together model name to use
        max_retries: Maximum number of retries on failure
        timeout: Timeout in seconds for the API call
    
    Returns:
        The model's response text
    """
    # You need to set TOGETHER_API_KEY environment variable
    if "TOGETHER_API_KEY" not in os.environ:
        raise ValueError("TOGETHER_API_KEY environment variable is not set")
    
    # Remove 'together/' prefix if present
    if model_name.startswith("together/"):
        model_name = model_name.replace("together/", "")
    
    # Initialize the Together client
    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            # Generate response
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes code repositories and identifies important files."},
                    {"role": "user", "content": instruction_content}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            # Return the response content
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def process_instruction_file(instruction_file, output_dir, model_name, api_type="claude", instruction_type="default"):
    """
    Process an instruction file to generate a model response.
    
    Args:
        instruction_file: Path to the instruction file
        output_dir: Directory to save the model response
        model_name: Model name to use
        api_type: API type to use (claude, openai, or gemini)
        instruction_type: Type of instruction file (default, hints, more_detailed_hints)
    
    Returns:
        Path to the generated response file
    """
    # Read the instruction file
    with open(instruction_file, 'r', encoding='utf-8') as f:
        instruction_content = f.read()
    
    # Generate response based on API type
    try:
        print(f"Generating response for {instruction_file} using {api_type} API with model {model_name}...")
        start_time = time.time()
        
        if api_type.lower() == "claude":
            response_text = generate_claude_response(instruction_content, model_name)
        elif api_type.lower() == "openai":
            response_text = generate_openai_response(instruction_content, model_name)
        elif api_type.lower() == "gemini":
            response_text = generate_gemini_response(instruction_content, model_name, max_retries=3, timeout=120)
        elif api_type.lower() == "together":
            response_text = generate_together_response(instruction_content, model_name, max_retries=3, timeout=120)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
        
        elapsed_time = time.time() - start_time
        print(f"Response generated in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error generating response for {instruction_file}: {e}")
        print(traceback.format_exc())
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save response to file
    project_name = instruction_file.parent.name
    # Replace slashes in model name with underscores for file naming
    safe_model_name = model_name.replace('/', '_')
    output_file = output_dir / f"{project_name}-{instruction_type}_{safe_model_name}.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response_text)
    
    print(f"Response saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate model responses for combined instructions')
    parser.add_argument('--instructions-dir', type=str, required=True, 
                        help='Directory containing combined instruction files')
    parser.add_argument('--output-dir', type=str, default='model_responses',
                        help='Output directory for model responses')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to use')
    parser.add_argument('--api', type=str, default='claude',
                        choices=['claude', 'openai', 'gemini', 'together'],
                        help='API to use (claude, openai, gemini, or together)')
    parser.add_argument('--instruction-type', type=str, default='default',
                        help='Type of instruction file to use (default, hints, more_detailed_hints)')
    parser.add_argument('--delay', type=int, default=1,
                        help='Delay between API calls in seconds')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for API calls')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout in seconds for API calls')
    
    args = parser.parse_args()
    
    instructions_dir = Path(args.instructions_dir)
    output_dir = Path(args.output_dir)
    
    if not instructions_dir.exists() or not instructions_dir.is_dir():
        print(f"Error: {instructions_dir} does not exist or is not a directory")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    safe_model_name = args.model.replace('/', '_')

    api_prefixes = {
        'claude': ['anthropic', 'claude'],
        'openai': ['openai', 'gpt', 'o1', 'o4'],
        'gemini': ['gemini', 'google'],
        'together': ['together', 'deepseek']
    }
    
    current_api_prefixes = api_prefixes.get(args.api.lower(), [])
    
    processed_count = 0
    for instruction_file in instructions_dir.glob("**/instructions*.md"):
        file_name = instruction_file.name
        
        if any(prefix in file_name.lower() for prefix in current_api_prefixes) or '_' + safe_model_name.lower() in file_name.lower():
            print(f"Processing matching file: {instruction_file}...")
            if process_instruction_file(instruction_file, output_dir, args.model, args.api, args.instruction_type):
                processed_count += 1
                # Add delay to avoid rate limiting
                if processed_count > 0 and args.delay > 0:
                    time.sleep(args.delay)
        elif not any(prefix in file_name.lower() for prefix in [p for prefixes in api_prefixes.values() for p in prefixes]):
            print(f"Processing generic file: {instruction_file}...")
            if process_instruction_file(instruction_file, output_dir, args.model, args.api, args.instruction_type):
                processed_count += 1
                # Add delay to avoid rate limiting
                if processed_count > 0 and args.delay > 0:
                    time.sleep(args.delay)
        else:
            print(f"Skipping file for different model: {instruction_file}")
    
    print(f"Processed {processed_count} instruction files")

if __name__ == "__main__":
    main()
