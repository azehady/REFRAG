#!/usr/bin/env python3
"""
Run multiple evaluations with random sampling and compute aggregate statistics.
"""
import json
import time
import random
import argparse
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Dict


def run_evaluation(
    test_json: str,
    samples: int,
    seed: int,
    enc: str,
    dec: str,
    refrag_load: str,
    p: float,
    run_id: int,
    output_dir: str = "runs/multieval"
) -> Dict:
    """Run a single evaluation with a specific random seed."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/run_{run_id}_seed{seed}.json"

    # Create temporary test file with random samples
    random.seed(seed)
    with open(test_json, 'r') as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    sampled = random.sample(all_data, min(samples, len(all_data)))
    temp_file = f"/tmp/eval_sample_{seed}.jsonl"
    with open(temp_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    cmd = [
        "uv", "run", "python", "eval/evaluate_comparison.py",
        "--test_json", temp_file,
        "--max_samples", str(samples),
        "--skip_refrag_v2",
        "--enc", enc,
        "--dec", dec,
        "--refrag_load", refrag_load,
        "--p", str(p),
        "--output", output_file
    ]

    env = {
        **os.environ,
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "TOKENIZERS_PARALLELISM": "false"
    }

    print(f"\n{'='*60}")
    print(f"RUN {run_id} (seed={seed})")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Create log file for this run
    log_file = f"{output_dir}/run_{run_id}_seed{seed}.log"

    t0 = time.time()
    with open(log_file, 'w') as lf:
        # Stream output to both console and log file
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env, bufsize=1)
        for line in process.stdout:
            print(line, end='')
            lf.write(line)
            lf.flush()
        process.wait()
    elapsed = time.time() - t0

    # Read results
    if Path(output_file).exists():
        with open(output_file, 'r') as f:
            return json.load(f)

    return {"error": f"Output file not created. Check log: {log_file}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="data/rag_eval_test.jsonl")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--enc", default="roberta-base")
    parser.add_argument("--dec", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--refrag_load", default="runs/refrag_v1_fixed_5Dec2025_2_17pm_policy")
    parser.add_argument("--p", type=float, default=0.25)
    args = parser.parse_args()

    # Generate random seeds
    random.seed(42)
    seeds = [random.randint(1, 10000) for _ in range(args.runs)]

    rag_accuracies = []
    refrag_accuracies = []
    all_results = []

    for run_id, seed in enumerate(seeds, 1):
        result = run_evaluation(
            args.test_json, args.samples, seed,
            args.enc, args.dec, args.refrag_load, args.p, run_id
        )

        if "error" not in result:
            summary = result.get("summary", {})
            rag_acc = summary.get("rag", {}).get("accuracy", 0) * 100
            refrag_acc = summary.get("refrag_v1", {}).get("accuracy", 0) * 100

            rag_accuracies.append(rag_acc)
            refrag_accuracies.append(refrag_acc)
            all_results.append({
                "run": run_id,
                "seed": seed,
                "rag_accuracy": rag_acc,
                "refrag_accuracy": refrag_acc,
                "rag_avg_time": summary.get("rag", {}).get("avg_time_per_query_sec", 0),
                "refrag_avg_time": summary.get("refrag_v1", {}).get("avg_time_per_query_sec", 0),
                "rag_throughput": summary.get("rag", {}).get("avg_throughput", 0),
                "refrag_throughput": summary.get("refrag_v1", {}).get("avg_throughput", 0),
            })

            print(f"\nRun {run_id} Results:")
            print(f"  RAG Accuracy:    {rag_acc:.1f}%")
            print(f"  REFRAG Accuracy: {refrag_acc:.1f}%")

    # Print summary
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)

    print(f"\n{'Run':<6} {'RAG Acc':<12} {'REFRAG Acc':<12} {'RAG Time':<12} {'REFRAG Time':<12}")
    print("-"*54)
    for r in all_results:
        print(f"{r['run']:<6} {r['rag_accuracy']:.1f}%{'':<7} {r['refrag_accuracy']:.1f}%{'':<7} "
              f"{r['rag_avg_time']:.2f}s{'':<7} {r['refrag_avg_time']:.2f}s")

    print("-"*54)

    if rag_accuracies and refrag_accuracies:
        avg_rag = sum(rag_accuracies) / len(rag_accuracies)
        avg_refrag = sum(refrag_accuracies) / len(refrag_accuracies)

        min_rag = min(rag_accuracies)
        max_rag = max(rag_accuracies)
        min_refrag = min(refrag_accuracies)
        max_refrag = max(refrag_accuracies)

        print(f"\n{'Metric':<20} {'RAG':<15} {'REFRAG v1':<15}")
        print("-"*50)
        print(f"{'Average Accuracy':<20} {avg_rag:.1f}%{'':<10} {avg_refrag:.1f}%")
        print(f"{'Min Accuracy':<20} {min_rag:.1f}%{'':<10} {min_refrag:.1f}%")
        print(f"{'Max Accuracy':<20} {max_rag:.1f}%{'':<10} {max_refrag:.1f}%")

        # Calculate std dev
        if len(rag_accuracies) > 1:
            import statistics
            std_rag = statistics.stdev(rag_accuracies)
            std_refrag = statistics.stdev(refrag_accuracies)
            print(f"{'Std Dev':<20} {std_rag:.1f}%{'':<10} {std_refrag:.1f}%")

        avg_rag_time = sum(r['rag_avg_time'] for r in all_results) / len(all_results)
        avg_refrag_time = sum(r['refrag_avg_time'] for r in all_results) / len(all_results)
        avg_rag_throughput = sum(r['rag_throughput'] for r in all_results) / len(all_results)
        avg_refrag_throughput = sum(r['refrag_throughput'] for r in all_results) / len(all_results)

        print(f"{'Avg Time (s)':<20} {avg_rag_time:.2f}{'':<11} {avg_refrag_time:.2f}")
        print(f"{'Avg Throughput':<20} {avg_rag_throughput:.1f}{'':<11} {avg_refrag_throughput:.1f}")

    # Save aggregate results
    aggregate = {
        "settings": {
            "samples_per_run": args.samples,
            "num_runs": args.runs,
            "p": args.p,
            "seeds": seeds
        },
        "runs": all_results,
        "aggregate": {
            "rag_avg_accuracy": sum(rag_accuracies) / len(rag_accuracies) if rag_accuracies else 0,
            "refrag_avg_accuracy": sum(refrag_accuracies) / len(refrag_accuracies) if refrag_accuracies else 0,
            "rag_min_accuracy": min(rag_accuracies) if rag_accuracies else 0,
            "rag_max_accuracy": max(rag_accuracies) if rag_accuracies else 0,
            "refrag_min_accuracy": min(refrag_accuracies) if refrag_accuracies else 0,
            "refrag_max_accuracy": max(refrag_accuracies) if refrag_accuracies else 0,
        }
    }

    output_file = "runs/multi_eval_aggregate.json"
    with open(output_file, 'w') as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nAggregate results saved to {output_file}")


if __name__ == "__main__":
    main()
