"""Build train/val parquet files of rl_app.py invoke payloads from the OfficeBench tasks on S3.

Prerequisite: upload the OfficeBench tasks to S3 with
examples/strands_officebench_agent/preprocess.py (see that example's README):

    cd examples/strands_officebench_agent
    python preprocess.py --officebench_dir /path/to/OfficeBench --s3_bucket <your-bucket>

This script reads the prefix that preprocess.py writes (s3://<your-bucket>/officebench/
by default). Each output row has two columns:

    payload: the exact ACR invoke payload rl_app.py consumes (models.InvocationRequest:
             task_uri, testbed_uri). Trainers forward it verbatim.
    prompt:  chat-format [{"role": "user", "content": <task instruction>}]. The payload
             carries S3 URIs rather than prompt text, so a trainer that needs a chat
             prompt column (verl's PayloadDataset: dataloader + length filtering) cannot
             synthesize one from it. The agent itself never reads this column.

Usage (from the repo root; benchmark, models, and utils are imported from
examples/strands_officebench_agent, so that directory must be on PYTHONPATH):
    PYTHONPATH=examples/strands_officebench_agent python \
        src/agentcore_rl_toolkit/backends/verl/examples/office_bench_agent/preprocess_officebench.py \
        --s3_input s3://<your-bucket>/officebench/ --out_dir ~/data/officebench
"""

import argparse
import concurrent.futures
import json
import os
import random
from collections import defaultdict

import boto3
import pandas as pd
from benchmark import list_all_subtasks
from botocore.exceptions import ClientError
from models import InvocationRequest, TaskConfig
from utils import parse_s3_uri


def split_task_ids(entries: list[dict], val_fraction: float, seed: int) -> set[str]:
    """Return the task_ids held out for validation.

    Splits by task dir, not subtask: subtasks of a task share one testbed and are
    near-duplicates, so a per-subtask split leaks. Stratified by category (the first
    component of the task_id) so both splits cover 1/2/3-app tasks.

    Deterministic for a given set of task_ids: categories are visited in sorted order
    and each category's ids are sorted before the seeded shuffle. The split does depend
    on which tasks exist, so a partial upload (preprocess.py --limit/--category) yields
    a different split than the full 300.
    """
    by_category = defaultdict(set)
    for e in entries:
        by_category[e["task_id"].split("-")[0]].add(e["task_id"])
    rng = random.Random(seed)
    val_tasks = set()
    for category in sorted(by_category):
        task_ids = sorted(by_category[category])
        rng.shuffle(task_ids)
        val_tasks.update(task_ids[: round(len(task_ids) * val_fraction)])
    return val_tasks


def testbed_exists(s3, uri: str) -> bool:
    bucket, key = parse_s3_uri(uri)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def load_task(s3, uri: str) -> TaskConfig:
    bucket, key = parse_s3_uri(uri)
    return TaskConfig(**json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s3_input", required=True, help="Prefix written by preprocess.py")
    parser.add_argument("--out_dir", default=os.path.expanduser("~/data/officebench"))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    s3 = boto3.client("s3")
    entries = list_all_subtasks(args.s3_input)
    if not entries:
        raise SystemExit(f"No subtasks under {args.s3_input} — run preprocess.py against it first.")
    print(f"Found {len(entries)} subtasks")

    testbeds = sorted({e["testbed_uri"] for e in entries})
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        has_testbed = dict(zip(testbeds, pool.map(lambda uri: testbed_exists(s3, uri), testbeds), strict=True))
        tasks = list(pool.map(lambda e: load_task(s3, e["task_uri"]), entries))

    val_tasks = split_task_ids(entries, args.val_fraction, args.seed)

    rows = {"train": [], "val": []}
    for e, task in zip(entries, tasks, strict=True):
        payload = InvocationRequest(
            task_uri=e["task_uri"],
            # Resolve missing testbeds here, so the runtime never HEADs a missing key.
            testbed_uri=e["testbed_uri"] if has_testbed[e["testbed_uri"]] else None,
        ).model_dump()
        split = "val" if e["task_id"] in val_tasks else "train"
        rows[split].append({"payload": payload, "prompt": [{"role": "user", "content": task.task}]})

    num_task_dirs = len({e["task_id"] for e in entries})
    task_dir_counts = {"train": num_task_dirs - len(val_tasks), "val": len(val_tasks)}

    os.makedirs(args.out_dir, exist_ok=True)
    for split, split_rows in rows.items():
        dst = os.path.join(args.out_dir, f"officebench_agent_{split}.parquet")
        pd.DataFrame(split_rows, columns=["payload", "prompt"]).to_parquet(dst, index=False)
        print(f"{split}: {len(split_rows)} rows ({task_dir_counts[split]} task dirs) -> {dst}")


if __name__ == "__main__":
    main()
