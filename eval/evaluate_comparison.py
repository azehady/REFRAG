#!/usr/bin/env python3
"""
Compare RAG vs REFRAG performance on the same test set.
"""
import json
import time
import argparse
import subprocess
import sys
from typing import List, Dict
import os

def run_rag_query(question: str, index_dir: str, topk: int, dec_model: str, max_new: int) -> Dict:
    """Run a single RAG query and return results"""
    cmd = [
        "uv", "run", "python", "src/rag.py", "generate",
        "--index_dir", index_dir,
        "--question", question,
        "--topk", str(topk),
        "--dec", dec_model,
        "--max_new", str(max_new),
        "--temperature", "0.0"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        # Parse JSON from output
        output = result.stdout
        # Find JSON in output
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(output[start:end])
    except Exception as e:
        return {"error": str(e), "answer": ""}

    return {"answer": "", "error": "Failed to parse output"}


def run_refrag_query(question: str, index_dir: str, load_dir: str, topk: int,
                     enc_model: str, dec_model: str, embed_model: str,
                     k: int, p: float, max_new: int) -> Dict:
    """Run a single REFRAG query and return results"""
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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE",
                                     "TOKENIZERS_PARALLELISM": "false"})
        output = result.stdout
        start = output.find('{')
        end = output.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(output[start:end])
    except Exception as e:
        return {"error": str(e), "answer": ""}

    return {"answer": "", "error": "Failed to parse output"}


def check_answer(predicted: str, expected: List[str]) -> bool:
    """Check if any expected answer is in the predicted answer"""
    predicted_lower = predicted.lower().strip()
    for exp in expected:
        if exp.lower() in predicted_lower:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Compare RAG vs REFRAG")
    parser.add_argument("--test_json", type=str, required=True, help="Test JSONL file")
    parser.add_argument("--rag_index", type=str, default="runs/rag_index")
    parser.add_argument("--refrag_index", type=str, default="runs/index")
    parser.add_argument("--refrag_load", type=str, default="runs/policy_aligned")
    parser.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--enc", type=str, default="roberta-base")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.25)
    parser.add_argument("--max_new", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--output", type=str, default="runs/comparison_results.json")
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
    print("=" * 60)

    # Results
    rag_results = []
    refrag_results = []

    rag_correct = 0
    refrag_correct = 0
    rag_total_time = 0
    refrag_total_time = 0

    for i, item in enumerate(test_data):
        question = item["question"]
        expected = item.get("answers", [])

        print(f"\n[{i+1}/{len(test_data)}] Q: {question}")
        print(f"Expected: {expected}")

        # Run RAG
        t0 = time.time()
        rag_out = run_rag_query(
            question, args.rag_index, args.topk, args.dec, args.max_new
        )
        rag_time = time.time() - t0
        rag_answer = rag_out.get("answer", "")
        rag_is_correct = check_answer(rag_answer, expected)
        if rag_is_correct:
            rag_correct += 1
        rag_total_time += rag_time

        # Run REFRAG
        t0 = time.time()
        refrag_out = run_refrag_query(
            question, args.refrag_index, args.refrag_load, args.topk,
            args.enc, args.dec, args.embed_model, args.k, args.p, args.max_new
        )
        refrag_time = time.time() - t0
        refrag_answer = refrag_out.get("answer", "")
        refrag_is_correct = check_answer(refrag_answer, expected)
        if refrag_is_correct:
            refrag_correct += 1
        refrag_total_time += refrag_time

        print(f"RAG Answer: {rag_answer[:100]}... ({'✓' if rag_is_correct else '✗'})")
        print(f"REFRAG Answer: {refrag_answer[:100]}... ({'✓' if refrag_is_correct else '✗'})")

        rag_results.append({
            "question": question,
            "expected": expected,
            "answer": rag_answer,
            "correct": rag_is_correct,
            "time_sec": rag_time,
            "throughput": rag_out.get("throughput_tok_per_sec", 0)
        })

        refrag_results.append({
            "question": question,
            "expected": expected,
            "answer": refrag_answer,
            "correct": refrag_is_correct,
            "time_sec": refrag_time,
            "throughput": refrag_out.get("throughput_tok_per_sec", 0)
        })

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    n = len(test_data)

    summary = {
        "total_samples": n,
        "rag": {
            "correct": rag_correct,
            "accuracy": rag_correct / n if n > 0 else 0,
            "total_time_sec": rag_total_time,
            "avg_time_per_query_sec": rag_total_time / n if n > 0 else 0,
            "avg_throughput": sum(r.get("throughput", 0) for r in rag_results) / n if n > 0 else 0
        },
        "refrag": {
            "correct": refrag_correct,
            "accuracy": refrag_correct / n if n > 0 else 0,
            "total_time_sec": refrag_total_time,
            "avg_time_per_query_sec": refrag_total_time / n if n > 0 else 0,
            "avg_throughput": sum(r.get("throughput", 0) for r in refrag_results) / n if n > 0 else 0
        }
    }

    print(f"\nRAG:")
    print(f"  Accuracy: {summary['rag']['accuracy']*100:.1f}% ({rag_correct}/{n})")
    print(f"  Avg Time: {summary['rag']['avg_time_per_query_sec']:.2f}s")
    print(f"  Avg Throughput: {summary['rag']['avg_throughput']:.1f} tok/s")

    print(f"\nREFRAG:")
    print(f"  Accuracy: {summary['refrag']['accuracy']*100:.1f}% ({refrag_correct}/{n})")
    print(f"  Avg Time: {summary['refrag']['avg_time_per_query_sec']:.2f}s")
    print(f"  Avg Throughput: {summary['refrag']['avg_throughput']:.1f} tok/s")

    # Save results
    full_results = {
        "summary": summary,
        "rag_results": rag_results,
        "refrag_results": refrag_results
    }

    with open(args.output, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
