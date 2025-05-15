#!/usr/bin/env python
import datetime
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from json.decoder import JSONDecodeError
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import git
import lox
import pandas as pd
import prompts
import typer
from dotenv import load_dotenv
from plots import plot_refactoring
from rich.console import Console

from aider import models
from aider.coders import Coder
from aider.dump import dump  # noqa: F401
from aider.io import InputOutput

BENCHMARK_DNAME = Path(os.environ.get("AIDER_BENCHMARK_DIR", "tmp.benchmarks"))

EXERCISES_DIR_DEFAULT = "rex-bench"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


NUM_TESTS = (89, 133)

load_dotenv(override=True)


def find_latest_benchmark_dir():
    benchmark_dirs = [d for d in BENCHMARK_DNAME.iterdir() if d.is_dir()]
    if not benchmark_dirs:
        print("Error: No benchmark directories found under tmp.benchmarks.")
        sys.exit(1)

    # Get current time and 24 hours ago
    now = datetime.datetime.now()
    day_ago = now - datetime.timedelta(days=1)

    # Filter directories by name pattern YYYY-MM-DD-HH-MM-SS--
    recent_dirs = []
    for d in benchmark_dirs:
        try:
            # Extract datetime from directory name
            date_str = d.name[:19]  # Takes YYYY-MM-DD-HH-MM-SS
            dir_date = datetime.datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
            if dir_date >= day_ago:
                recent_dirs.append(d)
        except ValueError:
            # Skip directories that don't match the expected format
            continue

    if not recent_dirs:
        print("Error: No benchmark directories found from the last 24 hours.")
        sys.exit(1)

    # Find directory with most recently modified .md file
    latest_dir = None
    latest_time = 0

    for d in recent_dirs:
        # Look for .md files in subdirectories
        for md_file in d.glob("*/.*.md"):
            if md_file.is_file():
                mtime = md_file.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_dir = d

    if not latest_dir:
        print("Error: No .md files found in recent benchmark directories.")
        sys.exit(1)

    print(f"Using the most recently updated benchmark directory: {latest_dir.name}")
    return latest_dir


def show_stats(dirnames, graphs):
    raw_rows = []
    for dirname in dirnames:
        row = summarize_results(dirname)
        raw_rows.append(row)

    # return

    seen = dict()
    rows = []
    for row in raw_rows:
        if not row:
            continue

        if row.completed_tests not in NUM_TESTS:
            print(f"Warning: {row.dir_name} is incomplete: {row.completed_tests}")

        kind = (row.model, row.edit_format)
        if kind in seen:
            dump(row.dir_name)
            dump(seen[kind])
            return

        seen[kind] = row.dir_name
        rows.append(vars(row))

    repeat_hi = repeat_lo = repeat_avg = None  # noqa: F841

    df = pd.DataFrame.from_records(rows)

    # dump(df)
    if graphs:
        # plot_timing(df)
        # plot_outcomes(df, repeats, repeat_hi, repeat_lo, repeat_avg)
        # plot_outcomes_claude(df)
        plot_refactoring(df)


def resolve_dirname(dirname, use_single_prior, make_new):
    if len(dirname.parts) > 1:
        return dirname

    priors = list(BENCHMARK_DNAME.glob(f"*--{dirname}"))
    if len(priors) == 1 and use_single_prior:
        dirname = priors[0].name
        print(f"Using pre-existing {dirname}")
    elif len(priors):
        if not make_new:
            print(f"Prior runs of {dirname} exist, use --new or name one explicitly")
            print()
            for prior in priors:
                print(prior)
            return

    if not re.match(r"\d\d\d\d-\d\d-\d\d-", str(dirname)):
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S--")
        dirname = now + dirname.name

    dirname = BENCHMARK_DNAME / dirname
    return dirname


@app.command()
def main(
    dirnames: Optional[List[str]] = typer.Argument(None, help="Directory names"),
    graphs: bool = typer.Option(False, "--graphs", help="Generate graphs"),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="Model name"),
    edit_format: str = typer.Option(None, "--edit-format", "-e", help="Edit format"),
    editor_model: str = typer.Option(None, "--editor-model", help="Editor model name"),
    editor_edit_format: str = typer.Option(None, "--editor-edit-format", help="Editor edit format"),
    replay: str = typer.Option(
        None,
        "--replay",
        help="Replay previous .aider.chat.history.md responses from previous benchmark run",
    ),
    max_apply_update_errors: int = typer.Option(
        3,
        "--max-apply-update-errors",
        help="Maximum number of apply update errors before stopping the test",
    ),
    keywords: str = typer.Option(
        None, "--keywords", "-k", help="Only run tests that contain keywords (comma sep)"
    ),
    clean: bool = typer.Option(
        False, "--clean", "-c", help="Discard the existing testdir and make a clean copy"
    ),
    cont: bool = typer.Option(False, "--cont", help="Continue the (single) matching testdir"),
    make_new: bool = typer.Option(False, "--new", "-n", help="Make a new dated testdir"),
    no_unit_tests: bool = typer.Option(False, "--no-unit-tests", help="Do not run unit tests"),
    no_aider: bool = typer.Option(False, "--no-aider", help="Do not run aider"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    stats_only: bool = typer.Option(
        False, "--stats", "-s", help="Do not run tests, just collect stats on completed tests"
    ),
    diffs_only: bool = typer.Option(False, "--diffs", help="Just diff the provided stats dirs"),
    tries: int = typer.Option(2, "--tries", "-r", help="Number of tries for running tests"),
    threads: int = typer.Option(1, "--threads", "-t", help="Number of threads to run in parallel"),
    num_tests: int = typer.Option(-1, "--num-tests", "-n", help="Number of tests to run"),
    num_ctx: Optional[int] = typer.Option(
        None, "--num-ctx", help="Override model context window size"
    ),
    exercises_dir: str = typer.Option(
        EXERCISES_DIR_DEFAULT, "--exercises-dir", help="Directory with exercise files"
    ),
    file_list: Optional[str] = typer.Option(
        None, "--file-list", help="Path to JSON file containing list of files to include"
    ),
):
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha[:7]
    if repo.is_dirty():
        commit_hash += "-dirty"

    if stats_only and not dirnames:
        latest_dir = find_latest_benchmark_dir()
        dirnames = [str(latest_dir)]

    if dirnames is None:
        dirnames = []

    if len(dirnames) > 1 and not (stats_only or diffs_only):
        print("Only provide 1 dirname unless running with --stats or --diffs")
        return 1

    updated_dirnames = []
    for dirname in dirnames:
        dirname = Path(dirname)
        dirname = resolve_dirname(dirname, stats_only or cont, make_new)
        if not dirname:
            return 1
        updated_dirnames.append(dirname)

    if stats_only:
        return show_stats(updated_dirnames, graphs)

    if diffs_only:
        return show_diffs(updated_dirnames)

    assert len(updated_dirnames) == 1, updated_dirnames
    dirname = updated_dirnames[0]

    if "AIDER_DOCKER" not in os.environ:
        print("Warning: benchmarking runs unvetted code from GPT, run in a docker container")
        return

    assert BENCHMARK_DNAME.exists() and BENCHMARK_DNAME.is_dir(), BENCHMARK_DNAME
    original_dname = BENCHMARK_DNAME / exercises_dir
    assert original_dname.exists() and original_dname.is_dir(), original_dname

    if clean and dirname.exists():
        print("Cleaning up and replacing", dirname)
        dir_files = set(fn.name for fn in dirname.glob("*"))
        original_files = set(fn.name for fn in original_dname.glob("*"))
        if dir_files != original_files:
            print("ERROR: will not delete dir that does not look like original tests", dirname)
            return

        dest = dirname.parent / "OLD" / dirname.name
        if dest.exists():
            old_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            dest = dirname.parent / "OLD" / (old_now + dirname.name)

        dirname.rename(dest)

    if not dirname.exists():
        print(f"Copying {original_dname} -> {dirname} ...")
        shutil.copytree(original_dname, dirname)
        print("...done")

    test_dnames = sorted(os.listdir(dirname))
    
    # Only process projects defined in environment variable
    allowed_projects = os.environ.get('ALLOWED_PROJECTS', '')
    if allowed_projects:
        allowed_projects = allowed_projects.split(',')
        print(f"Filtering projects to only include: {allowed_projects}")
        test_dnames = [dn for dn in test_dnames if dn in allowed_projects]
        print(f"Filtered projects: {test_dnames}")

    if keywords:
        keywords = keywords.split(",")
        test_dnames = [dn for dn in test_dnames for keyword in keywords if keyword in dn]

    # random.shuffle 제거 - 프로젝트 순서 유지
    if num_tests > 0:
        test_dnames = test_dnames[:num_tests]

    if threads == 1:
        all_results = []
        for testname in test_dnames:
            results = run_test(
                original_dname,
                dirname / testname,
                model,
                edit_format,
                tries,
                no_unit_tests,
                no_aider,
                verbose,
                commit_hash,
                replay,
                max_apply_update_errors,
                editor_model,
                editor_edit_format,
                num_ctx,
                file_list,
            )

            all_results.append(results)
            summarize_results(dirname)
    else:
        run_test_threaded = lox.thread(threads)(run_test)
        for testname in test_dnames:
            run_test_threaded.scatter(
                original_dname,
                dirname / testname,
                model,
                edit_format,
                tries,
                no_unit_tests,
                no_aider,
                verbose,
                commit_hash,
                replay,
                max_apply_update_errors,
                editor_model,
                editor_edit_format,
                num_ctx,
                file_list,
            )
        all_results = run_test_threaded.gather(tqdm=True)

    print()
    print()
    print()
    
    # 스레드 결과에서 None이 있는지 확인 (오류 발생 여부)
    if all_results and None in all_results:
        print("\nError: At least one test returned None, indicating failure!")
        print("Returning error code 1 to trigger retry logic")
        return 1
        
    results = summarize_results(dirname)
    
    # 컨텍스트 윈도우 소진이 발생하는 경우만 오류 코드 반환
    if results.exhausted_context_windows > 0:
        print("\nError: Context window exhausted during execution!")
        print("Returning error code 1 to trigger retry logic")
        return 1
        
    return 0


def show_diffs(dirnames):
    dirnames = sorted(dirnames)

    all_results = dict((dirname, load_results(dirname)) for dirname in dirnames)
    testcases = set()
    for results in all_results.values():
        testcases.update(result["testcase"] for result in results)

    testcases = sorted(testcases)

    unchanged = set()

    for testcase in testcases:
        all_outcomes = []
        for dirname in dirnames:
            results = all_results[dirname]
            result = [r for r in results if r["testcase"] == testcase][0]

            outcomes = tuple(result["tests_outcomes"])
            all_outcomes.append(True in outcomes)

        if len(set(all_outcomes)) == 1:
            unchanged.add(testcase)
            continue

        print()
        print(testcase)
        for outcome, dirname in zip(all_outcomes, dirnames):
            print(outcome, f"{dirname}/{testcase}/.aider.chat.history.md")

    changed = set(testcases) - unchanged
    print()
    print("changed:", len(changed), ",".join(sorted(changed)))
    print()
    print("unchanged:", len(unchanged), ",".join(sorted(unchanged)))


def load_results(dirname):
    dirname = Path(dirname)
    all_results = [json.loads(fname.read_text()) for fname in dirname.glob("*/.aider.results.json")]
    return all_results


def summarize_results(dirname):
    all_results = load_results(dirname)

    res = SimpleNamespace()
    res.total_tests = len(list(Path(dirname).glob("*")))

    try:
        tries = max(len(results.get("tests_outcomes", [])) for results in all_results if results)
    except ValueError:
        tries = 0

    res.dir_name = str(dirname)

    passed_tests = [0] * tries

    res.completed_tests = 0
    res.duration = 0
    res.cost = 0
    res.error_outputs = 0
    res.user_asks = 0
    res.test_timeouts = 0
    res.exhausted_context_windows = 0
    res.num_malformed_responses = 0
    res.num_with_malformed_responses = 0
    res.syntax_errors = 0
    res.indentation_errors = 0
    res.lazy_comments = 0

    variants = defaultdict(set)

    for results in all_results:
        if not results:
            continue

        res.completed_tests += 1
        tests_outcomes = results.get("tests_outcomes", [])
        passed = tests_outcomes and tests_outcomes[-1]
        if passed:
            for i in range(len(tests_outcomes) - 1, tries):
                passed_tests[i] += 1

        res.cost += results.get("cost", 0)
        res.duration += results.get("duration", 0)
        res.test_timeouts += results.get("test_timeouts", 0)

        res.error_outputs += results.get("num_error_outputs", 0)
        res.user_asks += results.get("num_user_asks", 0)
        res.exhausted_context_windows += results.get("num_exhausted_context_windows", 0)
        res.num_malformed_responses += results.get("num_malformed_responses", 0)
        if results.get("num_malformed_responses"):
            res.num_with_malformed_responses += 1
        res.lazy_comments += results.get("lazy_comments", 0)

        res.syntax_errors += results.get("syntax_errors", 0)
        res.indentation_errors += results.get("indentation_errors", 0)

        for key in "model edit_format commit_hash editor_model editor_edit_format".split():
            val = results.get(key)
            if val:
                variants[key].add(val)

    if not res.completed_tests:
        return

    # if res.completed_tests < 133:
    #    return

    console = Console(highlight=False)
    console.rule(title=str(dirname))

    commit_hashes = variants["commit_hash"]
    versions = get_versions(commit_hashes)
    date = dirname.name[:10]

    def show(stat, red="red"):
        val = getattr(res, stat)
        style = red if val else None
        console.print(f"  {stat}: {val}", style=style)

    percents = dict()
    for i in range(tries):
        pass_rate = 100 * passed_tests[i] / res.completed_tests
        percents[i] = pass_rate
        setattr(res, f"pass_rate_{i + 1}", f"{pass_rate:.1f}")

    print(f"- dirname: {dirname.name}")
    style = None if res.completed_tests in NUM_TESTS else "red"
    console.print(f"  test_cases: {res.completed_tests}", style=style)
    for key, val in variants.items():
        if len(val) > 1:
            style = "red"
        else:
            style = None
        val = ", ".join(map(str, val))
        setattr(res, key, val)
        console.print(f"  {key}: {val}", style=style)

    for i in range(tries):
        print(f"  pass_rate_{i + 1}: {percents[i]:.1f}")

    pct_well_formed = 1.0 - res.num_with_malformed_responses / res.completed_tests
    print(f"  percent_cases_well_formed: {pct_well_formed * 100:.1f}")

    show("error_outputs")
    show("num_malformed_responses")
    show("num_with_malformed_responses")
    show("user_asks")
    show("lazy_comments")
    show("syntax_errors")
    show("indentation_errors")
    show("exhausted_context_windows")
    show("test_timeouts")

    a_model = set(variants["model"]).pop()
    command = f"aider --model {a_model}"
    print(f"  command: {command}")

    print(f"  date: {date}")
    print("  versions:", ",".join(versions))

    res.avg_duration = res.duration / res.completed_tests
    print(f"  seconds_per_case: {res.avg_duration:.1f}")

    print(f"  total_cost: {res.cost:.4f}")

    res.avg_cost = res.cost / res.completed_tests

    projected_cost = res.avg_cost * res.total_tests

    print()
    print(
        f"costs: ${res.avg_cost:.4f}/test-case, ${res.cost:.2f} total,"
        f" ${projected_cost:.2f} projected"
    )

    console.rule()

    # print(json.dumps(vars(res), indent=4, sort_keys=True))
    return res


def get_versions(commit_hashes):
    versions = set()
    for hsh in commit_hashes:
        if not hsh:
            continue
        hsh = hsh.split("-")[0]
        try:
            version = subprocess.check_output(
                ["git", "show", f"{hsh}:aider/__init__.py"], universal_newlines=True
            )
            version = re.search(r'__version__ = "(.*)"', version).group(1)
            versions.add(version)
        except subprocess.CalledProcessError:
            pass
    return versions


def get_replayed_content(replay_dname, test_dname):
    replay_dname = Path(replay_dname)
    test_dname = Path(test_dname)
    dump(replay_dname, test_dname)

    test_name = test_dname.name
    replay_fname = replay_dname / test_name / ".aider.chat.history.md"
    dump(replay_fname)

    res = replay_fname.read_text()
    return res

    res = res.splitlines(keepends=True)
    res = [line for line in res if not line.startswith("> ") and not line.startswith("#### ")]
    return "".join(res)


def format_markdown_content(content, **kwargs):
    """Format markdown content by replacing placeholders with values.
    Example: {file_name} in markdown will be replaced with kwargs['file_name']
    """
    try:
        return content.format(**kwargs)
    except KeyError as e:
        print(f"Warning: Missing placeholder value for {e}")
        return content


def load_meta_links(test_dir):
    """Load meta links from JSON file if it exists."""
    meta_links_file = test_dir / ".docs/meta_links.json"
    if meta_links_file.exists():
        try:
            with open(meta_links_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse {meta_links_file}")
    return {}


def run_test(original_dname, testdir, *args, **kwargs):
    try:
        return run_test_real(original_dname, testdir, *args, **kwargs)
    except Exception as err:
        print("=" * 40)
        print("Test failed")
        print(err)
        traceback.print_exc()

        testdir = Path(testdir)
        results_fname = testdir / ".aider.results.json"
        results_fname.write_text(json.dumps(dict(exception=str(err))))


def run_test_real(
    original_dname,
    testdir,
    model_name,
    edit_format,
    tries,
    no_unit_tests,
    no_aider,
    verbose,
    commit_hash,
    replay,
    max_apply_update_errors,
    editor_model,
    editor_edit_format,
    num_ctx=None,
    file_list=None,
):
    """Run a single test with the specified parameters."""
    
    io = InputOutput()
    
    if not os.path.isdir(testdir):
        print("Not a dir:", testdir)
        return

    testdir = Path(testdir)

    history_fname = testdir / ".aider.chat.history.md"

    results_fname = testdir / ".aider.results.json"
    if results_fname.exists():
        try:
            res = json.loads(results_fname.read_text())
            return res
        except JSONDecodeError:
            print(f"{results_fname} failed to parse, skipping")
            return

    fnames = []
    EXCLUDE_PATTERNS = ["example", ".meta", "human_eval", ".csv",".json"]
    
    instr_name = os.environ.get('CURRENT_INSTRUCTION_NAME', '')
    if not instr_name:
        print("ERROR: CURRENT_INSTRUCTION_NAME environment variable not set")
        return 2
    
    model_dir = None
    if model_name:
        if "o1" in model_name:
            model_dir = "o1"
        elif "claude" in model_name or "anthropic" in model_name:
            model_dir = "claude"
        elif "gemini" in model_name:
            model_dir = "gemini"
        elif "deepseek" in model_name or "together" in model_name:
            model_dir = "deepseek"
    
    file_list_paths = []
    
    if model_dir:
        file_list_paths.append(Path(__file__).parent / "file_lists" / model_dir / f"{testdir.name}.json")
        file_list_paths.append(Path(__file__).parent / "file_lists" / model_dir / f"{testdir.name}-{instr_name}.json")
    
    file_list_paths.append(Path(__file__).parent / "file_lists" / f"{testdir.name}-{instr_name}.json")
    file_list_paths.append(Path(__file__).parent / "file_lists" / f"{testdir.name}.json")
    
    file_list_json = None
    if file_list:
        file_list_path = Path(file_list)
        if file_list_path.exists():
            file_list_json = file_list_path
            print(f"Using file list from command line argument: {file_list_json}")
        else:
            print(f"WARNING: File list specified in command line does not exist: {file_list}")
    
    if not file_list_json:
        for path in file_list_paths:
            if path.exists():
                file_list_json = path
                print(f"Found file list at: {file_list_json}")
                break
    
    if not file_list_json or not file_list_json.exists():
        print(f"ERROR: File list JSON not found in any of these locations:")
        for path in file_list_paths:
            print(f"  - {path}")
        print(f"Make sure the file exists for project {testdir.name} and instruction {instr_name}")
        return 2
    
    # Debugging: Output file list path
    print(f"Looking for file list at: {file_list_json}")
    print(f"File exists: {file_list_json.exists()}")
    
    # Extract current model name - try full model name too
    model_keys_to_try = []
    if model_name:
        # Full model name
        model_keys_to_try.append(model_name)
        # First part split by slash
        if '/' in model_name:
            model_keys_to_try.append(model_name.split('/')[0])
        # First part split by hyphen
        if '-' in model_name:
            model_keys_to_try.append(model_name.split('-')[0])
    # Add model-specific keys
    if model_name and ("claude" in model_name.lower() or "anthropic" in model_name.lower()):
        model_keys_to_try.append("claude")
    elif model_name and "o1" in model_name.lower():
        model_keys_to_try.append("o1")
    elif model_name and "gemini" in model_name.lower():
        model_keys_to_try.append("gemini")
    elif model_name and ("deepseek" in model_name.lower() or "together" in model_name.lower()):
        model_keys_to_try.append("together_ai/deepseek-ai/DeepSeek-R1")
        model_keys_to_try.append("deepseek")
    # Add default value
    model_keys_to_try.append("default")
    
    print(f"Model keys to try: {model_keys_to_try}")
    
    # Use file list from JSON if it exists
    if file_list_json.exists():
        try:
            with open(file_list_json, 'r', encoding='utf-8') as f:
                file_content = f.read()
                print(f"File content preview: {file_content[:200]}...")
                all_model_data = json.loads(file_content)
            
            # Debugging: Check model keys
            print(f"Available model keys: {list(all_model_data.keys())}")
            
            # Try multiple model keys sequentially
            found_model_key = None
            for model_key in model_keys_to_try:
                if model_key in all_model_data:
                    found_model_key = model_key
                    print(f"Found matching model key: {found_model_key}")
                    break
            
            if found_model_key:
                model_data = all_model_data[found_model_key]
                found_project = testdir.name
                
                # 모델 키를 못읽으면 첫번째 키를 가져오도록 함. 
                if found_project not in model_data and model_data:
                    found_project = list(model_data.keys())[0]
                    print(f"Project {testdir.name} not found in model data, using {found_project} instead")
                
                if found_project:
                    project_data = model_data.get(found_project, {})
                    print(f"Using file list for model: {found_model_key}, project: {found_project}")
                    
                    # 중요 파일 목록 추출
                    if "files" in project_data:
                        important_files = [f["path"] for f in project_data["files"]]
                        print(f"Found {len(important_files)} important files")
            else:
                # 모델이 없으면 default 모델 사용
                if "default" in all_model_data:
                    model_data = all_model_data["default"]
                    project_data = model_data.get(testdir.name, {})
                    print(f"Model {current_model} not found, using default model file list")
                else:
                    # default도 없으면 처음 모델 사용
                    first_model = next(iter(all_model_data.keys()))
                    model_data = all_model_data[first_model]
                    project_data = model_data.get(testdir.name, {})
                    print(f"Using file list for first available model: {first_model}")
                
            important_files = []
            file_list_str = ""
            raw_file_paths = []
            
            for file_info in project_data.get("files", []):
                file_path = file_info.get("path", "")
                purpose = file_info.get("purpose", "")
                if file_path:
                    src_file_path = f"src/{file_path}"
                    raw_file_paths.append(src_file_path)
                    file_list_str += f"{src_file_path} - {purpose}\n"
            
            for src_file_path in raw_file_paths:
                if not any(x in src_file_path for x in EXCLUDE_PATTERNS):
                    if not src_file_path.startswith('/'):
                        if src_file_path.startswith('./'):
                            src_file_path = src_file_path[2:]
                        src_file_path = str(testdir / src_file_path)
                    
                    file_obj = Path(src_file_path)
                    file_exists = file_obj.exists() and file_obj.is_file()
                    
                    if not file_exists:
                        for file_info in project_data.get("files", []):
                            if file_info.get("path", "") == src_file_path.replace("src/", ""):
                                exists_flag = file_info.get("exists", True)  # 기본값은 True
                                if not exists_flag:
                                    print(f"File marked as non-existent in JSON: {src_file_path}")
                                    # 디렉토리가 없으면 생성
                                    file_obj.parent.mkdir(parents=True, exist_ok=True)
                                    # 빈 파일 생성
                                    file_obj.touch()
                                    file_exists = True
                                    print(f"Created empty file: {file_obj}")
                                break
                    
                    if file_exists:
                        important_files.append(file_obj)
                    else:
                        print(f"Adding non-existent file to important_files: {file_obj}")
                        file_obj.parent.mkdir(parents=True, exist_ok=True)
                        file_obj.touch()
                        important_files.append(file_obj)
            
            for file_obj in important_files:
                fnames.append(file_obj)
                try:
                    relative_path = file_obj.relative_to(testdir)
                    original_fname = original_dname / testdir.name / relative_path
                    if original_fname.exists():
                        shutil.copy(original_fname, file_obj)
                except ValueError:
                    pass
            
            print(f"Loaded {len(important_files)} important files from {file_list_json}")
        except Exception as e:
            print(f"Error loading file list from {file_list_json}: {e}")
    
    file_list_loaded = 'file_list_str' in locals() and file_list_str
    
    if not file_list_loaded:
        print(f"ERROR: File list JSON at {file_list_json} failed to load properly.")
        print("Stopping benchmark execution as requested.")
        return 2
        
    if len(important_files) == 0:
        print("WARNING: No important files were found or created. Continuing anyway.")

    docs_dir = original_dname / testdir.name / ".docs"
    if docs_dir.exists():
        dst_docs_dir = testdir / ".docs"
        os.makedirs(dst_docs_dir, exist_ok=True)
        for doc_file in docs_dir.glob("*"):
            if doc_file.is_file():
                shutil.copy(doc_file, dst_docs_dir / doc_file.name)

    original_file_list = " ".join(fname.name for fname in fnames)
    
    if 'file_list_str' in locals() and file_list_str:
        file_list = file_list_str
        print("Using file list from JSON:", file_list)
        print(f"Loaded {len(important_files)} files from JSON file list")
    else:
        file_list = original_file_list
        print("Using original file list:", file_list)
        print(f"Loaded {len(fnames)} files using default method")
    
    # prompt  
    instructions = f"{prompts.use_system_prompt}\n"

    # Add meta instruction from the docs directory
    benchmark_meta = docs_dir / "meta_instruction.md"
    if benchmark_meta.exists():
        meta_content = benchmark_meta.read_text()
        
        # Load meta links from JSON
        meta_links = load_meta_links(docs_dir)
        
        format_vars = {
            'file_list': file_list,
            'testdir': testdir.name,
            'model': model_name,
            **meta_links  # Include all links from JSON file
        }
        meta_content = format_markdown_content(meta_content, **format_vars)
        instructions += meta_content + "\n\n"

    introduction = testdir / ".docs/introduction.md"
    if introduction.exists():
        instructions += introduction.read_text()
    
    # 동적으로 instruction 파일 로드 - 환경변수에 따라 적절한 파일 선택
    instr_name = os.environ.get('CURRENT_INSTRUCTION_NAME', 'default')
    instruction_file = testdir / f".docs/instructions_{instr_name}.md"
    
    # instruction 파일 로드
    print(f"Loading instruction file: {instruction_file}")
    instructions += instruction_file.read_text()
    
    instructions_append = testdir / ".docs/instructions.append.md"
    if instructions_append.exists():
        instructions += instructions_append.read_text()

    instructions += prompts.instructions_addendum.format(file_list=file_list)

    io = InputOutput(
        pretty=True,
        yes=True,
        chat_history_file=history_fname,
    )

    # weak_model_name = model_name
    weak_model_name = None

    main_model = models.Model(
        model_name,
        weak_model=weak_model_name,
        editor_model=editor_model,
        editor_edit_format=editor_edit_format,
    )

    if num_ctx:
        if not main_model.extra_params:
            main_model.extra_params = {}
        main_model.extra_params["num_ctx"] = num_ctx
    edit_format = edit_format or main_model.edit_format

    dump(main_model)
    dump(edit_format)
    show_fnames = ",".join(map(str, fnames))
    print("fnames:", show_fnames)

    coder = Coder.create(
        main_model,
        edit_format,
        io,
        fnames=fnames,
        use_git=False,
        stream=False,
        verbose=verbose,
        auto_lint=True,  # disabled for code-in-json experiments
        cache_prompts=True,
        suggest_shell_commands=False,
    )
    coder.max_apply_update_errors = max_apply_update_errors
    coder.temperature = 0.7 
    coder.show_announcements()
    # coder.get_repo_map()

    timeouts = 0

    syntax_errors = 0
    indentation_errors = 0
    lazy_comments = 0

    dur = 0
    test_outcomes = []
    for i in range(tries):
        start = time.time()
        if no_aider:
            pass
        elif replay:
            response = get_replayed_content(replay, testdir)
            coder.partial_response_content = response

            show = response.splitlines(keepends=True)
            show = [">> " + line for line in show]
            io.append_chat_history("".join(show))

            coder.apply_updates()
        else:
            response = coder.run(with_message=instructions, preproc=False)
        dur += time.time() - start

        if not no_aider:
            pat = r"^[+]? *[#].* [.][.][.] "
            # Count the number of lines that match pat in response
            dump(response)
            lazy_comments += len(re.findall(pat, response, re.MULTILINE))
            dump(lazy_comments)

        if coder.last_keyboard_interrupt:
            raise KeyboardInterrupt

        if no_unit_tests:
            break

        try:
            errors = run_unit_tests(testdir, history_fname)
        except subprocess.TimeoutExpired:
            errors = "Tests timed out!"
            timeouts += 1

        if errors:
            test_outcomes.append(False)
        else:
            test_outcomes.append(True)
            break

        if replay:
            io.append_chat_history(errors)

        errors = errors.splitlines()

        syntax_errors += sum(1 for line in errors if line.startswith("SyntaxError"))
        indentation_errors += sum(1 for line in errors if line.startswith("IndentationError"))

        print(errors[-1])
        errors = errors[:50]
        errors = "\n".join(errors)
        instructions = errors
        instructions += prompts.test_failures.format(file_list=file_list)

    results = dict(
        testdir=str(testdir),
        testcase=testdir.name,
        model=main_model.name,
        edit_format=edit_format,
        tests_outcomes=test_outcomes,
        cost=coder.total_cost,
        duration=dur,
        test_timeouts=timeouts,
        commit_hash=commit_hash,
        num_error_outputs=io.num_error_outputs,
        num_user_asks=io.num_user_asks,
        num_exhausted_context_windows=coder.num_exhausted_context_windows,
        num_malformed_responses=coder.num_malformed_responses,
        syntax_errors=syntax_errors,
        indentation_errors=indentation_errors,
        lazy_comments=lazy_comments,  # Add the count of pattern matches to the results
        chat_hashes=list(
            zip(
                coder.chat_completion_call_hashes,
                coder.chat_completion_response_hashes,
            )
        ),
    )

    if edit_format == "architect":
        results["editor_model"] = main_model.editor_model.name if main_model.editor_model else None
        results["editor_edit_format"] = main_model.editor_edit_format
    dump(results)

    results_fname.write_text(json.dumps(results, indent=4))

    # context window 소진 체크 (오직 이 경우에만 오류 반환)
    if coder.num_exhausted_context_windows > 0:
        print("\nError: Context window exhausted during execution!")
        print(f"Returning error code 1 for {testdir.name}\n")
        return None
    
    return results


def run_unit_tests(testdir, history_fname):
    command = [
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        str(testdir),
        "-t",
        str(testdir),
        "-p",
        "*_test.py",
    ]
    print(" ".join(command))

    timeout = 60

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )

    success = result.returncode == 0
    res = result.stdout
    res = cleanup_test_output(res, testdir)

    with history_fname.open("a") as fh:
        fh.write(f"```\n{res}\n```")

    if not success:
        print(f"Tests failed: {testdir}")
        return res


def cleanup_test_output(output, testdir):
    # remove timing info, to avoid randomizing the response to GPT
    res = re.sub(
        r"^Ran \d+ tests in \d+\.\d+s$",
        "",
        output,
        flags=re.MULTILINE,
    )
    res = re.sub(
        r"^====*$",
        "====",
        res,
        flags=re.MULTILINE,
    )
    res = re.sub(
        r"^----*$",
        "----",
        res,
        flags=re.MULTILINE,
    )

    res = res.replace(str(testdir), str(testdir.name))
    return res


@app.command()
def read(
    filepath: str = typer.Argument(..., help="Path to the file to read"),
    show_links: bool = typer.Option(False, "--links", "-l", help="Show meta links if reading meta_instruction.md"),
):
    """Read and process file content using aider's InputOutput functionality"""
    io = InputOutput()
    try:
        if filepath.endswith(".md"):
            content = io.read_file(filepath)
        elif filepath.endswith(".json"):
            with open(filepath, 'r') as f:
                content = json.load(f)
        else:
            print(f"Unsupported file type: {filepath}")
            return

        if not content:
            print(f"Could not read content from {filepath}")
            return

        # If it's meta_instruction.md and show_links is True, try to read corresponding meta_links.json
        if show_links and filepath.endswith("meta_instruction.md"):
            json_path = str(filepath).replace("meta_instruction.md", "meta_links.json")
            try:
                with open(json_path, 'r') as f:
                    links = json.load(f)
                print("\nMeta Links:")
                print(json.dumps(links, indent=2))
                print("\nContent:")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"\nWarning: Could not load meta links: {e}\n")
        
        print(content)
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    app()
