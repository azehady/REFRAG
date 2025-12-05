#!/usr/bin/env python3
"""
Compare RAG vs REFRAG v1 vs REFRAG v2 performance on the same test set.

This script runs all three implementations on the same questions and compares:
- Accuracy (exact match)
- Time-to-first-token (TTFT)
- Throughput (tokens/sec)
- Total generation time

Usage:
    python eval/evaluate_comparison.py --test_json data/rag_eval_test.jsonl --max_samples 20

    # With all models
    python eval/evaluate_comparison.py \
        --test_json data/rag_eval_test.jsonl \
        --rag_index runs/rag_index \
        --refrag_index runs/index \
        --refrag_load runs/policy_aligned \
        --refrag_v2_load runs/refrag_v2_cpt \
        --max_samples 20
"""
import json
import time
import argparse
import subprocess
import sys
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import os
from pathlib import Path
from datetime import datetime


@dataclass
class ModelResult:
    """Result from a single model query."""
    answer: str = ""
    correct: bool = False
    time_sec: float = 0.0
    ttft_ms: float = 0.0
    throughput: float = 0.0
    error: str = ""
    raw_output: Dict = field(default_factory=dict)


@dataclass
class ModelStats:
    """Aggregate statistics for a model."""
    name: str
    correct: int = 0
    total: int = 0
    total_time_sec: float = 0.0
    total_ttft_ms: float = 0.0
    total_throughput: float = 0.0
    results: List[Dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    @property
    def avg_time_sec(self) -> float:
        return self.total_time_sec / self.total if self.total > 0 else 0

    @property
    def avg_ttft_ms(self) -> float:
        return self.total_ttft_ms / self.total if self.total > 0 else 0

    @property
    def avg_throughput(self) -> float:
        return self.total_throughput / self.total if self.total > 0 else 0


def run_command(cmd: List[str], timeout: int = 180) -> Tuple[str, str, int]:
    """Run a command and return stdout, stderr, returncode."""
    env = {
        **os.environ,
        "TOKENIZERS_PARALLELISM": "false",
        "KMP_DUPLICATE_LIB_OK": "TRUE"
    }
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1
    except Exception as e:
        return "", str(e), -1


def parse_json_output(output: str) -> Dict:
    """Extract JSON from command output."""
    start = output.find('{')
    end = output.rfind('}') + 1
    if start >= 0 and end > start:
        try:
            return json.loads(output[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def run_rag_query(
    question: str,
    index_dir: str,
    topk: int,
    dec_model: str,
    max_new: int
) -> ModelResult:
    """Run a single RAG query and return results."""
    cmd = [
        "uv", "run", "python", "src/rag.py", "generate",
        "--index_dir", index_dir,
        "--question", question,
        "--topk", str(topk),
        "--dec", dec_model,
        "--max_new", str(max_new),
        "--temperature", "0.0"
    ]

    t0 = time.time()
    stdout, stderr, rc = run_command(cmd)
    elapsed = time.time() - t0

    result = ModelResult(time_sec=elapsed)

    if rc != 0:
        result.error = stderr[:200] if stderr else "Unknown error"
        return result

    parsed = parse_json_output(stdout)
    result.answer = parsed.get("answer", "")
    result.throughput = parsed.get("throughput_tok_per_sec", 0)
    result.ttft_ms = parsed.get("ttft_ms", elapsed * 1000)
    result.raw_output = parsed

    return result


def run_refrag_v1_query(
    question: str,
    index_dir: str,
    load_dir: str,
    topk: int,
    enc_model: str,
    dec_model: str,
    embed_model: str,
    k: int,
    p: float,
    max_new: int
) -> ModelResult:
    """Run a single REFRAG v1 query and return results."""
    cmd = [
        "uv", "run", "python", "src/refrag.py", "generate",
        "--index_dir", index_dir,
        "--load_dir", load_dir,
        "--question", question,
        "--topk", str(topk),
        "--enc", enc_model,
        "--dec", dec_model,
        "--embed_model", embed_model,
        "--k", str(k),
        "--p", str(p),
        "--max_new", str(max_new),
        "--temperature", "0.0"
    ]

    t0 = time.time()
    stdout, stderr, rc = run_command(cmd)
    elapsed = time.time() - t0

    result = ModelResult(time_sec=elapsed)

    if rc != 0:
        result.error = stderr[:200] if stderr else "Unknown error"
        return result

    parsed = parse_json_output(stdout)
    result.answer = parsed.get("answer", "")
    result.throughput = parsed.get("throughput_tok_per_sec", 0)
    result.ttft_ms = parsed.get("ttft_ms", elapsed * 1000)
    result.raw_output = parsed

    return result


def run_refrag_v2_query(
    question: str,
    index_dir: str,
    load_dir: str,
    topk: int,
    encoder: str,
    decoder: str,
    k: int,
    max_tokens: int,
    expand_fraction: float = 0.1,
    no_policy: bool = False
) -> ModelResult:
    """Run a single REFRAG v2 query and return results."""
    cmd = [
        "uv", "run", "python", "src/refrag_v2.py", "generate",
        "--index-dir", index_dir,
        "--load-dir", load_dir,
        "--question", question,
        "--topk", str(topk),
        "--encoder", encoder,
        "--decoder", decoder,
        "--k", str(k),
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
        "--expand-fraction", str(expand_fraction),
    ]

    if no_policy:
        cmd.append("--no-policy")

    t0 = time.time()
    stdout, stderr, rc = run_command(cmd, timeout=300)  # Longer timeout for v2
    elapsed = time.time() - t0

    result = ModelResult(time_sec=elapsed)

    if rc != 0:
        result.error = stderr[:200] if stderr else "Unknown error"
        return result

    parsed = parse_json_output(stdout)
    result.answer = parsed.get("answer", "")
    result.throughput = parsed.get("throughput_tok_per_sec", 0)
    result.ttft_ms = parsed.get("ttft_ms", elapsed * 1000)
    result.raw_output = parsed

    return result


def check_answer(predicted: str, expected: List[str]) -> bool:
    """Check if any expected answer is in the predicted answer."""
    if not predicted:
        return False
    predicted_lower = predicted.lower().strip()
    for exp in expected:
        if exp.lower() in predicted_lower:
            return True
    return False


def check_model_available(load_dir: str) -> bool:
    """Check if model checkpoint exists."""
    if not load_dir:
        return False
    path = Path(load_dir)
    if not path.exists():
        return False
    # Check for any checkpoint files
    checkpoint_patterns = ["*.pt", "*.safetensors", "*.bin"]
    for pattern in checkpoint_patterns:
        if list(path.glob(pattern)):
            return True
    return False


def print_summary_table(stats_list: List[ModelStats]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<15} {'Accuracy':>12} {'Avg Time':>12} {'Avg TTFT':>12} {'Throughput':>12}")
    print("-" * 63)

    for stats in stats_list:
        if stats.total > 0:
            print(f"{stats.name:<15} "
                  f"{stats.accuracy*100:>10.1f}% "
                  f"{stats.avg_time_sec:>10.2f}s "
                  f"{stats.avg_ttft_ms:>10.1f}ms "
                  f"{stats.avg_throughput:>10.1f}")

    print("-" * 63)

    # Relative comparison
    if len(stats_list) >= 2 and stats_list[0].total > 0:
        baseline = stats_list[0]  # RAG is baseline
        print(f"\nRelative to {baseline.name}:")
        for stats in stats_list[1:]:
            if stats.total > 0 and baseline.avg_time_sec > 0:
                acc_diff = (stats.accuracy - baseline.accuracy) * 100
                speedup = baseline.avg_time_sec / stats.avg_time_sec if stats.avg_time_sec > 0 else 0
                ttft_speedup = baseline.avg_ttft_ms / stats.avg_ttft_ms if stats.avg_ttft_ms > 0 else 0

                acc_str = f"+{acc_diff:.1f}%" if acc_diff >= 0 else f"{acc_diff:.1f}%"
                print(f"  {stats.name}: Accuracy {acc_str}, "
                      f"Time {speedup:.1f}x, TTFT {ttft_speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RAG vs REFRAG v1 vs REFRAG v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Test data
    parser.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    parser.add_argument("--max_samples", type=int, default=20, help="Max samples to evaluate")
    parser.add_argument("--output", type=str, default="runs/comparison_results.json")

    # RAG settings
    parser.add_argument("--rag_index", type=str, default="runs/rag_index")
    parser.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")

    # REFRAG v1 settings
    parser.add_argument("--refrag_index", type=str, default="runs/index")
    parser.add_argument("--refrag_load", type=str, default="runs/policy_aligned")
    parser.add_argument("--enc", type=str, default="roberta-base")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--k", type=int, default=16, help="Chunk size for REFRAG")
    parser.add_argument("--p", type=float, default=0.25, help="Expansion probability for v1")

    # REFRAG v2 settings
    parser.add_argument("--refrag_v2_index", type=str, default="runs/index")
    parser.add_argument("--refrag_v2_load", type=str, default="runs/refrag_v2_cpt")
    parser.add_argument("--v2_encoder", type=str, default="roberta-large")
    parser.add_argument("--v2_decoder", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--v2_k", type=int, default=16, help="Chunk size for v2")
    parser.add_argument("--v2_expand_fraction", type=float, default=0.1)
    parser.add_argument("--v2_no_policy", action="store_true")

    # Common settings
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--max_new", type=int, default=64)

    # Control which models to run
    parser.add_argument("--skip_rag", action="store_true", help="Skip RAG evaluation")
    parser.add_argument("--skip_refrag_v1", action="store_true", help="Skip REFRAG v1 evaluation")
    parser.add_argument("--skip_refrag_v2", action="store_true", help="Skip REFRAG v2 evaluation")

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_json}")
    test_data = []
    with open(args.test_json, 'r') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    if args.max_samples and args.max_samples < len(test_data):
        import random
        random.seed(42)
        random.shuffle(test_data)
        test_data = test_data[:args.max_samples]

    print(f"Evaluating on {len(test_data)} samples")

    # Check which models are available
    models_to_run = []

    if not args.skip_rag:
        if Path(args.rag_index).exists():
            models_to_run.append("RAG")
            print(f"✓ RAG index found: {args.rag_index}")
        else:
            print(f"✗ RAG index not found: {args.rag_index}")

    if not args.skip_refrag_v1:
        if check_model_available(args.refrag_load):
            models_to_run.append("REFRAG_v1")
            print(f"✓ REFRAG v1 checkpoint found: {args.refrag_load}")
        else:
            print(f"✗ REFRAG v1 checkpoint not found: {args.refrag_load}")

    if not args.skip_refrag_v2:
        if check_model_available(args.refrag_v2_load):
            models_to_run.append("REFRAG_v2")
            print(f"✓ REFRAG v2 checkpoint found: {args.refrag_v2_load}")
        else:
            print(f"✗ REFRAG v2 checkpoint not found: {args.refrag_v2_load}")
            print(f"  To train v2: just v2-recon && just v2-cpt")

    if not models_to_run:
        print("\n❌ No models available to evaluate!")
        sys.exit(1)

    print(f"\nRunning evaluation for: {', '.join(models_to_run)}")
    print("=" * 80)

    # Initialize stats
    stats = {
        "RAG": ModelStats("RAG"),
        "REFRAG_v1": ModelStats("REFRAG_v1"),
        "REFRAG_v2": ModelStats("REFRAG_v2"),
    }

    # Run evaluation
    for i, item in enumerate(test_data):
        question = item["question"]
        expected = item.get("answers", [])

        print(f"\n[{i+1}/{len(test_data)}] Q: {question}")
        print(f"Expected: {expected}")

        # RAG
        if "RAG" in models_to_run:
            result = run_rag_query(
                question, args.rag_index, args.topk, args.dec, args.max_new
            )
            result.correct = check_answer(result.answer, expected)

            stats["RAG"].total += 1
            stats["RAG"].total_time_sec += result.time_sec
            stats["RAG"].total_ttft_ms += result.ttft_ms
            stats["RAG"].total_throughput += result.throughput
            if result.correct:
                stats["RAG"].correct += 1

            stats["RAG"].results.append({
                "question": question,
                "expected": expected,
                "answer": result.answer,
                "correct": result.correct,
                "time_sec": result.time_sec,
                "ttft_ms": result.ttft_ms,
                "throughput": result.throughput,
                "error": result.error
            })

            status = "✓" if result.correct else "✗"
            answer_preview = result.answer[:80] if result.answer else result.error[:80]
            print(f"  RAG: {answer_preview}... ({status})")

        # REFRAG v1
        if "REFRAG_v1" in models_to_run:
            result = run_refrag_v1_query(
                question, args.refrag_index, args.refrag_load, args.topk,
                args.enc, args.dec, args.embed_model, args.k, args.p, args.max_new
            )
            result.correct = check_answer(result.answer, expected)

            stats["REFRAG_v1"].total += 1
            stats["REFRAG_v1"].total_time_sec += result.time_sec
            stats["REFRAG_v1"].total_ttft_ms += result.ttft_ms
            stats["REFRAG_v1"].total_throughput += result.throughput
            if result.correct:
                stats["REFRAG_v1"].correct += 1

            stats["REFRAG_v1"].results.append({
                "question": question,
                "expected": expected,
                "answer": result.answer,
                "correct": result.correct,
                "time_sec": result.time_sec,
                "ttft_ms": result.ttft_ms,
                "throughput": result.throughput,
                "error": result.error
            })

            status = "✓" if result.correct else "✗"
            answer_preview = result.answer[:80] if result.answer else result.error[:80]
            print(f"  REFRAG v1: {answer_preview}... ({status})")

        # REFRAG v2
        if "REFRAG_v2" in models_to_run:
            result = run_refrag_v2_query(
                question, args.refrag_v2_index, args.refrag_v2_load, args.topk,
                args.v2_encoder, args.v2_decoder, args.v2_k, args.max_new,
                args.v2_expand_fraction, args.v2_no_policy
            )
            result.correct = check_answer(result.answer, expected)

            stats["REFRAG_v2"].total += 1
            stats["REFRAG_v2"].total_time_sec += result.time_sec
            stats["REFRAG_v2"].total_ttft_ms += result.ttft_ms
            stats["REFRAG_v2"].total_throughput += result.throughput
            if result.correct:
                stats["REFRAG_v2"].correct += 1

            stats["REFRAG_v2"].results.append({
                "question": question,
                "expected": expected,
                "answer": result.answer,
                "correct": result.correct,
                "time_sec": result.time_sec,
                "ttft_ms": result.ttft_ms,
                "throughput": result.throughput,
                "error": result.error
            })

            status = "✓" if result.correct else "✗"
            answer_preview = result.answer[:80] if result.answer else result.error[:80]
            print(f"  REFRAG v2: {answer_preview}... ({status})")

    # Print summary
    active_stats = [stats[m] for m in ["RAG", "REFRAG_v1", "REFRAG_v2"] if m in models_to_run]
    print_summary_table(active_stats)

    # Build summary dict
    summary = {
        "timestamp": datetime.now().isoformat(),
        "test_file": args.test_json,
        "total_samples": len(test_data),
        "models_evaluated": models_to_run,
    }

    for model_name in models_to_run:
        s = stats[model_name]
        summary[model_name.lower()] = {
            "correct": s.correct,
            "total": s.total,
            "accuracy": s.accuracy,
            "total_time_sec": s.total_time_sec,
            "avg_time_per_query_sec": s.avg_time_sec,
            "avg_ttft_ms": s.avg_ttft_ms,
            "avg_throughput": s.avg_throughput,
        }

    # Save results
    full_results = {
        "summary": summary,
    }

    for model_name in models_to_run:
        full_results[f"{model_name.lower()}_results"] = stats[model_name].results

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n📊 Detailed results saved to {args.output}")

    # Return exit code based on whether REFRAG v2 beats v1
    if "REFRAG_v2" in models_to_run and "REFRAG_v1" in models_to_run:
        if stats["REFRAG_v2"].accuracy > stats["REFRAG_v1"].accuracy:
            print("\n🎉 REFRAG v2 outperforms REFRAG v1!")
            return 0
        elif stats["REFRAG_v2"].accuracy == stats["REFRAG_v1"].accuracy:
            print("\n📈 REFRAG v2 matches REFRAG v1 accuracy")
            return 0
        else:
            print("\n⚠️  REFRAG v2 underperforms REFRAG v1")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
