#!/bin/bash -l

# Claude 3.7 Sonnet
chmod +x run_benchmark_claude.sh
./run_benchmark_claude.sh

# OpenAI o1
chmod +x run_benchmark_o1.sh
./run_benchmark_o1.sh

#OpenAI o4-mini
chmod +x run_benchmark_o4-mini.sh
./run_benchmark_o4-mini.sh

# DeepSeek R1
chmod +x run_benchmark_deepseek.sh
./run_benchmark_deepseek.sh