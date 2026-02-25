#!/usr/bin/env python3
import argparse
import json
import math
from typing import Dict, List


DEFAULT_TASKS = [
    "enem_greedy",
    "enem_2022_greedy",
    "enem_full_2022_greedy",
    "enem_full_2023_greedy",
    "enem_full_2024_greedy",
    "logiqa_greedy",
    "balanced_copa_greedy",
    "arc_challenge_greedy_pt",
]


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        fail(f"{name} must be in [0,1], got {value}")


def get_task_metrics(data: Dict, task: str, prompt_mode: str) -> Dict:
    try:
        return data["results"][task][prompt_mode]
    except KeyError as exc:
        fail(f"Missing metrics block for task={task}, prompt_mode={prompt_mode}: {exc}")


def check_task(metrics: Dict, task: str, pass_k: int, tol: float) -> None:
    if "acc" not in metrics:
        fail(f"{task}: missing acc")
    if "unknown_pred" not in metrics:
        fail(f"{task}: missing unknown_pred")
    if "num_examples" not in metrics:
        fail(f"{task}: missing num_examples")

    check_probability(f"{task}.acc", float(metrics["acc"]))
    check_probability(f"{task}.unknown_pred", float(metrics["unknown_pred"]))

    pass_key = f"acc_pass@{pass_k}"
    unknown_pass_key = f"unknown_pred_pass@{pass_k}"
    for key in [pass_key, unknown_pass_key, "c_mean", "n", "k"]:
        if key not in metrics:
            fail(f"{task}: missing {key}")

    acc_pass = float(metrics[pass_key])
    unknown_pass = float(metrics[unknown_pass_key])
    c_mean = float(metrics["c_mean"])
    n = float(metrics["n"])
    k = float(metrics["k"])

    check_probability(f"{task}.{pass_key}", acc_pass)
    check_probability(f"{task}.{unknown_pass_key}", unknown_pass)

    if n < 1:
        fail(f"{task}: n must be >= 1, got {n}")
    if k < 1:
        fail(f"{task}: k must be >= 1, got {k}")
    if n < k:
        fail(f"{task}: n must be >= k, got n={n}, k={k}")
    if int(round(k)) != pass_k:
        fail(f"{task}: metric k={k} does not match expected pass_k={pass_k}")

    if c_mean < 0.0 or c_mean > n + 1e-12:
        fail(f"{task}: c_mean must be in [0,n], got c_mean={c_mean}, n={n}")

    if pass_k == 1:
        expected = c_mean / n
        if not math.isclose(acc_pass, expected, abs_tol=tol, rel_tol=0.0):
            fail(
                f"{task}: acc_pass@1 mismatch: got {acc_pass}, expected c_mean/n={expected} (tol={tol})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity checks for pass@k output JSON."
    )
    parser.add_argument("--results_path", required=True, help="Path to output JSON.")
    parser.add_argument(
        "--prompt_mode",
        default="dynamic-random",
        help="Prompt mode key inside results JSON.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated tasks to validate.",
    )
    parser.add_argument(
        "--pass_k",
        type=int,
        required=True,
        help="Expected pass_k to validate metric key acc_pass@k.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for pass@1 identity check (acc_pass@1 == c_mean/n).",
    )
    args = parser.parse_args()

    if args.pass_k < 1:
        fail("--pass_k must be >= 1")

    with open(args.results_path, "r") as f:
        data = json.load(f)

    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        fail("No tasks provided")

    for task in tasks:
        metrics = get_task_metrics(data, task, args.prompt_mode)
        check_task(metrics, task, args.pass_k, args.tol)

    print("PASS: sanity checks succeeded")
    print(f"Validated tasks: {', '.join(tasks)}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"pass_k: {args.pass_k}")


if __name__ == "__main__":
    main()
