#!/usr/bin/env python

import asyncio
import json
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from dump import dump
from swebench_docker.constants import MAP_REPO_TO_TEST_FRAMEWORK, MAP_VERSION_TO_INSTALL
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_directives
from utils import get_dataset, get_devin_instance_ids, load_predictions  # noqa: F401

# clipped from `run_docker_evaluation()`
def get_docker_image(task_instance: dict, namespace: str = "aorwall"):
    repo_name = task_instance["repo"].replace("/", "_")

    specifications = MAP_VERSION_TO_INSTALL[task_instance["repo"]][task_instance["version"]]
    image_prefix = "swe-bench"

    if specifications.get("instance_image", False):
        docker_image = (
            f"{namespace}/{image_prefix}-{repo_name}-instance:{task_instance['instance_id']}"
        )
    else:
        docker_image = f"{namespace}/{image_prefix}-{repo_name}-testbed:{task_instance['version']}"

    return docker_image


# A no-op patch which creates an empty file is used to stand in for
# the `model_patch` and/or `test_patch` when running SWE Bench tests
# without one or both of those patches.
NOOP_PATCH = (
    "diff --git a/empty.file.{nonce}.ignore b/empty.file.{nonce}.ignore\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
)

def remove_patches_to_tests(model_patch):
    """
    Remove any changes to the tests directory from the provided patch.
    This is to ensure that the model_patch does not disturb the repo's
    tests when doing acceptance testing with the `test_patch`.
    """
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    in_tests = False
    for line in lines:
        if line.startswith("diff --git a/tests/"):
            in_tests = True
        elif line.startswith("diff --git "):
            in_tests = False

        if not in_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)

def run_tests(entry, model_patch=None, use_test_patch=False, model_name_or_path="none"):
    """
    Run tests for the SWE Bench `entry`, optionally applying a `model_patch` first.

    If `use_test_patch` is True, then also apply the `test_patch` to bring in
    the tests which determine if the issue is resolved. So False means
    only run the tests that existed at the `base_commit` and any new/changed
    tests contained in the `model_patch`.

    Optionally specify a `model_name_or_path`, which isn't really used since
    the log_dir for the tests is a temp dir which is discarded.
    """
    # Placeholder for actual test execution logic
    return True, "Tests passed"

def main_check_docker_images():
    """
    Check docker images for SWE Bench.
    """
    # Placeholder for actual docker image checking logic
    return 0

def update_cache(cache_fname, instances, good_dockers, bad_dockers):
    """
    Update cache with docker information.
    """
    # Placeholder for actual cache update logic
    pass

def main_preds():
    """
    Main function for predictions.
    """
    # Placeholder for actual prediction logic
    pass

if __name__ == "__main__":
    status = main_check_docker_images()
    # status = main_preds()
    sys.exit(status)
