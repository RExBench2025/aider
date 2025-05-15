import os
cache_dir = "/projectnb/tin-lab/cache/huggingface"
os.environ["HF_HOME"] = cache_dir
import datetime
import json
import shutil
from pathlib import Path

from datasets import load_dataset

from dump import dump  # noqa: F401

FULL_DATASET = "princeton-nlp/SWE-bench"
FULL_DATASET_FNAME = FULL_DATASET.replace("/", "--") + ".json"

LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
LITE_DATASET_FNAME = LITE_DATASET.replace("/", "--") + ".json"

def dump_dataset(dataset, fname):
    """
    Save the dataset to json.
    """
    entries = list(dataset)
    for entry in entries:
        entry["FAIL_TO_PASS"] = json.loads(entry["FAIL_TO_PASS"])
        entry["PASS_TO_PASS"] = json.loads(entry["PASS_TO_PASS"])

    with open(fname, "w") as f:
        json.dump(entries, f, indent=4)

def get_full_dataset():
    return get_dataset(FULL_DATASET, FULL_DATASET_FNAME)

def get_lite_dataset():
    return get_dataset(LITE_DATASET, LITE_DATASET_FNAME)

def get_dataset(dataset, fname):
    """
    Load the `DATASET` from hugging face, and turn it into a dict
    keyed on `instance_id`.
    Cache the dict locally in a json file.
    """

    fname = Path(fname)
    if fname.exists():
        dataset = json.loads(fname.read_text())
    else:
        dataset = load_dataset(dataset)
        dump_dataset(dataset, fname)

    return {entry["instance_id"]: entry for entry in dataset}

def load_predictions(paths, devin_only=False):
    """
    Load all predictions from the given paths.
    """
    predictions = {}
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for pred_file in path.glob("*.json"):
                with open(pred_file) as f:
                    preds = json.load(f)
                    predictions.update(preds)
        elif path.is_file():
            with open(path) as f:
                preds = json.load(f)
                predictions.update(preds)

    if devin_only:
        devin_ids = get_devin_instance_ids()
        predictions = {k: v for k, v in predictions.items() if k in devin_ids}

    return predictions

def is_plausible(pred):
    """
    Determine if a prediction is plausible.
    """
    return pred.get("edit_outcome") and pred.get("lint_outcome") and pred.get("test_outcome")

def get_plausible(preds):
    """
    Filter predictions to only include plausible ones.
    """
    return {k: v for k, v in preds.items() if is_plausible(v)}

def check_criteria(pred, criteria):
    """
    Check if a prediction meets given criteria.
    """
    return all(pred.get(criterion) for criterion in criteria)

def pick_winner(results):
    """
    Pick the best result based on criteria.
    """
    for result in results:
        if is_plausible(result):
            return result
    return None

def get_devin_instance_ids():
    """
    Get instance ids for devin.
    """
    return ["devin-001", "devin-002"]
