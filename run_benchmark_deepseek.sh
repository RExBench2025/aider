#!/bin/bash -l
MODEL=together_ai/deepseek-ai/DeepSeek-R1
NAME=deepseek-rex-bench
EXEC_DIR=rex-bench

# List of projects to benchmark - comment out as needed to run specific projects only
PROJECTS=(
    "checkeval"
    "cogs"
    "entity-tracking-multimodal"
    "explain-then-translate"
    "implicit-instructions"
    "mission-impossible"
    "othello"
    "re-reading"
    "reasoning-or-reciting"
    "tree-of-thoughts"
    "varierr-nli"
    "winodict"
)

# Define instruction types without prefixes (instructions_ prefix will be added in file path)
INSTRUCTION_TYPES=("nohints")

# Run file extraction pipeline (if needed)
run_file_extraction() {
    local instruction_type=$1
    
    echo "Running file extraction pipeline for instruction type: $instruction_type..."

    # Use Together API key for DeepSeek model
    export TOGETHER_API_KEY==YOUR_API_KEY_HERE

    # Set output directory to benchmark directory so that benchmark.py can find it
    local output_dir="./benchmark"
    mkdir -p "$output_dir/file_lists/deepseek"
    
    # Remove 'instructions_' prefix from instruction_type if present
    local instr_type=${instruction_type#instructions_}
    if [ "$instr_type" == "$instruction_type" ]; then
        instr_type="nohints"
    fi
    
    echo "Using instruction type: $instr_type for file extraction"
    
    # Run file extraction pipeline - specify instruction-type
    python ./benchmark/scripts/run_file_extraction_pipeline.py \
        --rex-bench-dir ./tmp.benchmarks/$EXEC_DIR \
        --model "$MODEL" \
        --auto-generate \
        --api together \
        --output-dir "$output_dir" \
        --instruction-type "$instr_type"
    
    # Wait a moment to ensure files are written
    sleep 5
}

# Function to generate file lists for each instruction type and run
generate_file_lists() {
    echo "Generating file lists for all instruction types and runs..."
    
    # Create benchmark/file_lists/deepseek directory if it doesn't exist
    mkdir -p "./benchmark/file_lists/deepseek"
    
    # Define instruction types
    local instruction_types=("nohints")
    
    local total_file_count=0
    
    for instruction_type in "${instruction_types[@]}"; do
        echo "\n======================================="
        echo "Processing instruction type: $instruction_type"
        echo "=======================================\n"
        
        # Run file extraction pipeline for this instruction type
        local instruction_file="instructions_${instruction_type}"
        run_file_extraction "$instruction_file"
        
        # Process each run
        for i in {1..3}; do
            echo "\n---------------------------------------"
            echo "Generating run $i for instruction type: $instruction_type"
            echo "---------------------------------------\n"
            
            local run_file_count=0
            local missing_projects=()
            
            # Process each project
            for project in "${PROJECTS[@]}"; do
                # Create test file (including run number)
                test_file="./benchmark/file_lists/deepseek/test-${project}-${instruction_type}_${i}.json"
                
                # 파일 찾기 - 하이픈이 포함된 프로젝트 이름도 처리
                # First try with exact name
                generated_file="./benchmark/file_lists/deepseek/${project}-${instruction_type}.json"
                
                # If file not found, search all JSON files in the directory
                if [ ! -f "$generated_file" ]; then
                    # Find files that start with project name (handle projects with hyphens)
                    found_file=$(find ./benchmark/file_lists/deepseek/ -name "${project}*-${instruction_type}.json" -type f | head -n 1)
                    if [ -n "$found_file" ]; then
                        generated_file="$found_file"
                        echo "Found alternative file for project $project: $generated_file"
                    fi
                fi
                
                if [ -f "$generated_file" ]; then
                    # Create test file (including run number)
                    cp "$generated_file" "$test_file"
                    echo "Created test file for run $i: $test_file"
                    run_file_count=$((run_file_count + 1))
                else
                    echo "Warning: Generated file for project $project with instruction type $instruction_type not found for run $i."
                    missing_projects+=("$project")
                fi
            done
            
            if [ ${#missing_projects[@]} -gt 0 ]; then
                echo "Missing test files for projects: ${missing_projects[*]}"
                echo "Running file extraction pipeline for ${instruction_file} (run $i)..."
                run_file_extraction "$instruction_file"
            fi
            
            echo "Processed $run_file_count files for $instruction_type (run $i)"
            total_file_count=$((total_file_count + run_file_count))
        done
    done
    
    echo "\nFile list generation complete. Total files processed: $total_file_count"
}

# Install jq (needed for JSON parsing in file lists)
if ! command -v jq &> /dev/null; then
    echo "Installing jq for JSON parsing..."
    conda install -y -c conda-forge jq
fi

# Options for performing diff after benchmark execution
GENERATE_DIFF=false
RUN_NAME=""

if [ "$1" = "--diff" ]; then
    # Only perform diff for already executed benchmarks
    if [ -z "$2" ]; then
        echo "Error: Please provide the run name for diff. Example: ./run_benchmark.sh --diff test-rex-bench_1"
        exit 1
    fi
    echo "Diff generation will be processed separately."
    exit 0
fi

if [ "$2" = "--with-diff" ]; then
    # Perform diff after benchmark execution
    GENERATE_DIFF=true
fi

echo "Generated file list: "
echo "This file list contains important files for the model to focus on."
echo "The benchmark.py script will automatically use these file lists."
echo "If file lists are not properly loaded, the benchmark will be skipped."

# API rate limit related settings
MAX_RETRIES=5       # Maximum retry count
RETRY_DELAY=60      # Wait time before retry (seconds)
BETWEEN_RUNS_DELAY=120  # Wait time between benchmark runs (seconds)
BETWEEN_PROJECTS_DELAY=300  # Wait time between projects (seconds)

# Create folder name with current date and time
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
RESULTS_DIR="./results-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
echo "Results will be saved to $RESULTS_DIR"

# Prepare execution
generate_file_lists

# Instruction type loop - outermost loop
for instruction_type in "${INSTRUCTION_TYPES[@]}"; do
    # Remove 'instructions_' prefix if present for display and filename construction
    instr_type=${instruction_type#instructions_}
    if [ "$instr_type" == "$instruction_type" ]; then
        instr_file="$instruction_type.md"
    else
        instr_file="$instruction_type"
        instr_type="$instr_file"
    fi
    
    echo "\n============================================"
    echo "Processing instruction type: $instr_type"
    echo "============================================\n"
    
    # Create results directory for this instruction
    INSTRUCTION_RESULTS_DIR="$RESULTS_DIR/$instr_type"
    mkdir -p "$INSTRUCTION_RESULTS_DIR"
    
    # Set environment variables for instruction
    # 환경변수 설정 - benchmark.py가 이 값을 사용하여 instruction 파일을 로드합니다
    export CURRENT_INSTRUCTION_NAME="${instr_type}"
    echo "Setting CURRENT_INSTRUCTION_NAME environment variable to: $CURRENT_INSTRUCTION_NAME"
    echo "Instructions will be loaded from: .docs/instructions_${instr_type}.md 파일"
    
    # Run loop - middle loop
    for run_number in {1..3}; do
        RUN_NAME="${NAME}-${instr_type}_${run_number}"
        # Variable to share one benchmark directory for each run
        BENCHMARK_DIR=""
        echo "\n------------------------------------------"
        echo "Running experiment $run_number of 3 for $instr_type"
        echo "------------------------------------------\n"
        
        # Project loop - innermost loop
        for project in "${PROJECTS[@]}"; do
            echo "\n------------------------------------------"
            echo "Processing project: $project for instruction: $instr_type (run $run_number)"
            echo "------------------------------------------\n"
            
            # Project directory path
            project_dir="./tmp.benchmarks/$EXEC_DIR/$project"
            docs_dir="$project_dir/.docs"
            
            # Check if project directory exists
            if [ ! -d "$project_dir" ]; then
                echo "Warning: Project directory $project_dir not found. Skipping."
                continue
            fi
            
            # 해당 instruction 파일이 존재하는지 확인합니다
            instruction_file="$docs_dir/instructions_${instr_type}.md"
            if [ ! -f "$instruction_file" ]; then
                echo "Warning: Instruction file $instruction_file not found for project $project. Skipping."
                continue
            else
                echo "Found instruction file: $instruction_file for project: $project"
            fi
        
            # Set environment variable for allowed projects - 현재 프로젝트만 포함
            export ALLOWED_PROJECTS="$project"
            echo "Setting ALLOWED_PROJECTS environment variable to: $ALLOWED_PROJECTS"
            
            # Find the correct file list for this project, instruction type, and run number
            test_file="./benchmark/file_lists/deepseek/test-${project}-${instr_type}_${run_number}.json"
            
            # Check if test file exists
            if [ ! -f "$test_file" ]; then
                echo "Warning: Test file not found: $test_file. Skipping this run."
                continue
            fi
            
            echo "Using test file: $test_file"
            
            # Add retry logic
            retry_count=0
            success=false
            
            while [ $retry_count -lt $MAX_RETRIES ] && [ "$success" = false ]; do
                echo "Attempt $(($retry_count + 1)) of $MAX_RETRIES for project: $project (run $run_number)"
            
                # Run benchmark with the correct test file
                if [ -z "$BENCHMARK_DIR" ]; then
                    # 첫 번째 프로젝트: --new 플래그 사용하여 새 디렉토리 생성
                    echo "Running first project of run $run_number with --new flag"
                    python ./benchmark/benchmark.py main "$RUN_NAME" \
                        --model $MODEL \
                        --edit-format diff \
                        --exercises-dir $EXEC_DIR \
                        --file-list "$test_file" \
                        --new
                    
                    # 생성된 디렉토리 이름 찾기
                    BENCHMARK_DIR=$(ls -d tmp.benchmarks/*--${RUN_NAME} 2>/dev/null | sort -r | head -n 1)
                    echo "Created benchmark directory: $BENCHMARK_DIR"
                else
                    # 나머지 프로젝트: 저장된 디렉토리 이름 사용
                    echo "Using existing benchmark directory: $BENCHMARK_DIR"
                    python ./benchmark/benchmark.py main "$BENCHMARK_DIR" \
                        --model $MODEL \
                        --edit-format diff \
                        --exercises-dir $EXEC_DIR \
                        --file-list "$test_file"
                fi
                
                # Check execution result
                BENCHMARK_RESULT=$?
                if [ $BENCHMARK_RESULT -eq 0 ]; then
                    success=true
                    echo "Benchmark completed successfully for project: $project (run $run_number)"
                    
                    echo "Benchmark completed for $RUN_NAME - results being saved..."
                    
                    # 디렉토리 이름 이미 알고 있으므로 직접 사용
                    RESULT_DIR="$BENCHMARK_DIR"
                    if [ -n "$RESULT_DIR" ]; then
                        echo "Copying results from $RESULT_DIR to $INSTRUCTION_RESULTS_DIR/run_${run_number}/${project}"
                        mkdir -p "$INSTRUCTION_RESULTS_DIR/run_${run_number}/${project}"
                        cp -r $RESULT_DIR/* "$INSTRUCTION_RESULTS_DIR/run_${run_number}/${project}/"
                        echo "Results successfully copied to $INSTRUCTION_RESULTS_DIR/run_${run_number}/${project}"
                    else
                        echo "Warning: Result directory not found for project: $project"
                    fi
                elif [ $BENCHMARK_RESULT -eq 2 ]; then
                    # When file list loading fails (exit code 2)
                    echo "ERROR: Failed to load file lists properly for project: $project. Skipping this benchmark."
                    echo "Please check the file lists in ./benchmark/file_lists/ directory."
                    break
                else
                    retry_count=$((retry_count + 1))
                    if [ $retry_count -lt $MAX_RETRIES ]; then
                        echo "API error or rate limit exceeded. Waiting for $RETRY_DELAY seconds before retrying..."
                        sleep $RETRY_DELAY
                    else
                        echo "Maximum retry attempts reached. Moving to the next run."
                    fi
                fi
            done
        done
        
        # Summarize results for this run and instruction type
        echo "\n------------------------------------------"
        echo "Summary of results for run $run_number with instruction type $instr_type:"
        for project in "${PROJECTS[@]}"; do
            project_result_dir="$INSTRUCTION_RESULTS_DIR/run_${run_number}/${project}"
            if [ -d "$project_result_dir" ]; then
                echo "  - Project $project: Results available"
            else
                echo "  - Project $project: No results available"
            fi
        done
        
        # 모든 프로젝트가 완료되었음을 표시
        echo "All projects completed for run: $run_number with instruction type: $instr_type"
        
        # Wait before starting the next run (중간 루프)
        if [ $run_number -lt 3 ]; then
            echo "Waiting for $BETWEEN_RUNS_DELAY seconds before starting the next run..."
            sleep $BETWEEN_RUNS_DELAY
        fi
    done
    
    # Wait before starting the next instruction type (외부 루프)
    if [ "$instruction_type" != "${INSTRUCTION_TYPES[-1]}" ]; then
        echo "Waiting for $BETWEEN_PROJECTS_DELAY seconds before starting the next instruction type..."
        sleep $BETWEEN_PROJECTS_DELAY
    fi
done


echo "All benchmarks completed. Results saved to $RESULTS_DIR"
