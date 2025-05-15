#!/usr/bin/env python
import os
import tempfile
import multiprocessing

# 프로젝트 내 임시 디렉토리 경로 설정
temp_base_dir = os.path.join('/projectnb/tin-lab/yukyung/aider', 'tmp')
os.makedirs(temp_base_dir, exist_ok=True)

# 환경 변수 설정
os.environ['TMPDIR'] = temp_base_dir
tempfile.tempdir = temp_base_dir
multiprocessing.set_start_method('spawn')  # 안전한 멀티프로세싱 시작 방법

import json
import random
import subprocess
import sys
from pathlib import Path

import lox
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model, register_litellm_models

from dump import dump
from tests import run_tests
from utils import get_full_dataset, get_lite_dataset, get_devin_instance_ids, get_plausible, load_predictions, pick_winner

REPOS_DNAME = Path("repos")
CHAT_LOGS_DNAME = Path("chat-logs")
PREDS_DNAME = Path("predictions")


def diff_versus_commit(git_dname, commit):
    """
    Take a diff of `git_dname` current contents versus the `commit`.
    """
    diff_cmd = f"git -C {git_dname} diff {commit}"
    diff_output = subprocess.check_output(diff_cmd.split()).decode()
    return diff_output


def files_in_patch(patch):
    """
    Extract the list of modified files from a unified diff patch string.
    """
    files = []
    for line in patch.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fname = line.split("/", 1)[1]
            if fname not in files:
                files.append(fname)
    return files


def checkout_repo(git_tempdir, entry):
    """
    Clone the SWE Bench entry's git `repo` into `dname` at the `base_commit`.
    Make a tempdir if no `dname` provided.
    """
    github_url = "https://github.com/"
    repo_url = github_url + entry["repo"]
    commit = entry["base_commit"]

    print(repo_url, commit)

    checkout_repo_url_commit(git_tempdir, repo_url, commit)


def checkout_repo_url_commit(repo_dname, url, commit):
    """
    Clone the git `url` into `dname` at `commit`.
    Check a local cache of the bare repo to avoid pulling from github every time.
    """
    repo_name = url.split("/")[-1].split(".")[0]
    repo_name += ".git"

    REPOS_DNAME.mkdir(exist_ok=True)
    bare_repo = REPOS_DNAME / repo_name

    if not bare_repo.exists():
        cmd = f"git clone --bare {url} {bare_repo}"
        subprocess.run(cmd.split(), check=True)

    cmd = f"git clone {bare_repo} {repo_dname}"
    subprocess.run(cmd.split(), check=True)

    cmd = f"git -c advice.detachedHead=false -C {repo_dname} checkout {commit}"
    subprocess.run(cmd.split(), check=True)


def show_problems(dataset):
    """
    Print out all the instance_id and problem_descriptions.
    """
    for inst, entry in dataset.items():
        problem = entry["problem_statement"].splitlines()[0]
        print(f"{inst}: {problem}")


def run_pre_existing_tests(entry, git_dname):
    """
    Given the current contents of the `git_dname`, run the tests that
    were present in the entry's `repo` at the time of the
    `base_commit` or which have been added into the repo since.
    """
    model_patch = diff_versus_commit(git_dname, entry["base_commit"])
    passed, output = run_tests(
        entry,
        model_patch=model_patch,
        use_test_patch=False,
    )
    if passed is None:
        return

    if passed:
        return

    output = output.split(">>>>> Applied Patch (test)")[-1]

    return output


def get_coder(model_name, git_dname, chat_history_file, fnames, max_apply_update_errors, verbose):
    """
    Get an instance of aider to work with the given LLM `model_name` on the code in `git_dname`.
    """
    io = InputOutput(
        pretty=True,
        yes=True,
        chat_history_file=chat_history_file,
    )

    main_model = Model(
        model_name,
        weak_model=None,  # Assuming no weak model is used
        editor_model=None,  # Assuming no separate editor model is used
        editor_edit_format=None  # Assuming no separate editor edit format is used
    )

    coder = Coder.create(
        main_model,
        edit_format=None,  # Assuming default edit format
        io=io,
        fnames=fnames,
        use_git=False,
        stream=False,
        verbose=verbose,
        cache_prompts=True,
        suggest_shell_commands=False,
    )
    coder.max_apply_update_errors = max_apply_update_errors
    coder.show_announcements()

    return coder


def process_one_instance(entry, num_tries, models, temperature, model_name_or_path, out_dname):
    """
    Process one `entry` from SWE Bench using the LLM `models` at the
    given `temperature`.  Set `model_name_or_path` in the result json.
    Store the result json and the chat log into `out_dname`.
    """
    instance_id = entry["instance_id"]
    base_commit = entry["base_commit"]

    print("=" * 60)
    dump(instance_id)
    print("=" * 60)
    problem_statement = entry["problem_statement"]
    print(problem_statement)

    oracle = False
    gold_files = files_in_patch(entry["patch"])
    if oracle:
        oracle_files = gold_files
    else:
        oracle_files = None

    chat_history_file = out_dname / (instance_id + ".md")

    if chat_history_file.exists():
        chat_history_file.unlink()

    results = []
    cost = 0
    winner = None

    for attempt in range(1, num_tries + 1):
        for model in models:
            dump(attempt, model)

            with tempfile.TemporaryDirectory(dir="/projectnb/tin-lab/yukyung/aider-swe-bench/tmp") as git_tempdir:
                dump(git_tempdir)
                checkout_repo(git_tempdir, entry)

                fnames = [Path(git_tempdir) / f for f in gold_files]

                print(f"Debug: model={model}, git_tempdir={git_tempdir}, chat_history_file={chat_history_file}")
                coder = get_coder(
                    model,
                    git_tempdir,
                    chat_history_file,
                    fnames,
                    max_apply_update_errors=3,  # Example value, adjust as needed
                    verbose=True  # Example value, adjust as needed
                )

                dump(instance_id)
                dump(gold_files)

                message = """Below is a real GitHub issue from a popular GitHub repository.
The issue was filed some time ago.
The repo has been checked out at the commit that existed at the moment the issue was filed.
If you are already familiar with this repo, be cautious!
You are working with an old version of the repo!
Filenames, directory names, file contents, etc may be different than what you're used to.

Propose changes to update the repo to fix the problem below.

#"""
                message += problem_statement
                try:
                    coder.run(with_message=message, preproc=False)
                except Exception as coder_err:
                    dump(coder_err)
                    continue

                added_files = coder.get_inchat_relative_files()

                if not added_files:
                    message = """You haven't named any files in this repo.
Remember, this repo is checked out at quite an old commit.
So the file layout and contents may be unfamiliar.

Tell me: which 3-5 files from this repo should I look at to solve the problem?
"""
                    coder.run(with_message=message, preproc=False)

                dump(instance_id)
                dump(gold_files)
                dump(added_files)

                cost += coder.total_cost

                model_patch = diff_versus_commit(git_tempdir, base_commit)
                dump(model_patch)

            result = dict(
                instance_id=instance_id,
                model_name_or_path=model_name_or_path,
                model_patch=model_patch,
                model=model,
                temperature=temperature,
                cost=coder.total_cost,
                added_files=added_files,
                gold_files=gold_files,
                edited_files=files_in_patch(model_patch),
                edit_outcome=coder.edit_outcome,
                lint_outcome=coder.lint_outcome,
                test_outcome=coder.test_outcome,
            )
            result["try"] = attempt
            results.append(result)

            dump(result)

            if model_patch and coder.edit_outcome and coder.lint_outcome and coder.test_outcome:
                winner = result
                break

        if winner:
            break

    if not winner:
        winner = pick_winner(results)

    if not winner:
        result = dict(
            instance_id=instance_id,
            model_name_or_path=model_name_or_path,
            model_patch=None,
        )

    dump(winner)
    if not winner:
        return

    print("\n\nFinal diff:\n")
    print(winner["model_patch"])

    winner = dict(winner)

    winner.update(
        dict(
            tries=attempt,
            all_results=results,
            cost=cost,
        )
    )

    out_fname = out_dname / (instance_id + ".json")
    out_fname.write_text(json.dumps(winner, indent=4))


def process_instances(prefix, dataset, models, num_tries, temperature, threads, prior_dnames, just_devin_570):
    """
    prefix - Prefix used in front of the dirname in predictions/.
    dataset - The subset of the SWE Bench dataset to process.
    models - List of models to use to try and find plausible solutions.
    num_tries - Number of attempts to make using each model.
    temperature - Temp to use during chat completions.
    threads - How many problems to attempt concurrently.
    prior_dnames - Names of predictions/ dirnames from previous runs.
                   If they contain a plausible solution for an instance,
                   don't continue looking.
    """
    models_slug = "--".join(model.replace("/", "-") for model in models)
    model_name_or_path = "aider--" + models_slug
    models_slug = prefix + "--" + models_slug

    dump(models)
    dump(temperature)

    out_dname = PREDS_DNAME / models_slug
    if not out_dname.exists():
        out_dname.mkdir()

    dump(out_dname)

    done_preds = load_predictions([out_dname], just_devin_570)
    done_instances = set(done_preds.keys())
    dump(len(done_instances))

    dump(prior_dnames)
    prior_preds = load_predictions(prior_dnames, just_devin_570)
    dump(len(prior_preds))

    plausible_instances = get_plausible(prior_preds)
    dump(len(plausible_instances))

    if prior_preds:
        all_instances = set(prior_preds.keys())
    else:
        all_instances = set(dataset.keys())

    remaining_instances = set(all_instances)
    remaining_instances -= done_instances
    remaining_instances -= plausible_instances

    remaining_instances = list(remaining_instances)
    random.shuffle(remaining_instances)

    dump(sorted(remaining_instances))
    dump(len(remaining_instances))

    print()
    print("press enter...")
    input()

    if not CHAT_LOGS_DNAME.exists():
        CHAT_LOGS_DNAME.mkdir()

    chat_history_dname = CHAT_LOGS_DNAME / models_slug
    chat_history_dname.mkdir(exist_ok=True)

    if threads > 1:
        process_one_instance_lox = lox.process(threads)(process_one_instance)
        process_one_instance_func = process_one_instance_lox.scatter
        gather = process_one_instance_lox.gather
    else:
        process_one_instance_func = process_one_instance

    for instance_id in remaining_instances:
        if instance_id in done_instances:
            print("skipping", instance_id)
            continue

        process_one_instance_func(
            dataset[instance_id],
            num_tries,
            models,
            temperature,
            model_name_or_path,
            out_dname,
        )

        print("#" * 60)

    if threads > 1:
        gather()


def main():
    models_json = Path(".aider.models.json")
    if models_json.exists():
        print(f"Registering {models_json}")
        register_litellm_models([str(models_json)])

    prefix = "lite025"

    models = ["gpt-4o"]

    num_tries = 1

    temperature = 0

    dataset = get_lite_dataset()

    just_devin_570 = False

    process_instances(
        prefix,
        dataset,
        models,
        num_tries,
        temperature,
        threads=1,
        prior_dnames=[],
        just_devin_570=just_devin_570,
    )


if __name__ == "__main__":
    status = main()
    sys.exit(status)
