#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard RAG (Retrieval-Augmented Generation) — Baseline implementation for comparison with REFRAG

This script implements a standard RAG pipeline using:
  - Qdrant as the vector database
  - Same embedding models as REFRAG for fair comparison
  - Same LLM decoder for generation
  - Matching CLI interface with refrag.py

USAGE (examples):
  # 0) Install deps
  #    uv pip install torch transformers qdrant-client sentence-transformers

  # 1) Build Qdrant index from corpus
  #    uv run python rag.py index --corpus data/corpus_large.txt --index_dir runs/rag_index --embed_model BAAI/bge-small-en-v1.5

  # 2) Generate answer
  #    uv run python rag.py generate --index_dir runs/rag_index --question "Which river flows through City_20?" --topk 4

  # 3) Evaluate on test set
  #    uv run python rag.py evaluate --index_dir runs/rag_index --test_json data/rag_train.jsonl --topk 4
"""

import os
import sys
import json
import time
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ----------------------------
# MLflow (optional)
# ----------------------------

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


def ensure_mlflow():
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow not installed. Run: pip install mlflow")


class MLflowTracker:
    """MLflow experiment tracking wrapper for RAG."""

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "RAG_Baseline",
        run_name: Optional[str] = None
    ):
        ensure_mlflow()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = None

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, params: Optional[Dict] = None):
        """Start MLflow run and log parameters."""
        self.run = mlflow.start_run(run_name=self.run_name)
        if params:
            mlflow.log_params(params)
        print(f"[MLflow] Started run: {self.run_name}")
        return self.run

    def log_params(self, params: Dict):
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        mlflow.log_artifact(local_path, artifact_path)

    def end_run(self):
        """End MLflow run."""
        if self.run:
            mlflow.end_run()
            print("[MLflow] Ended run")


# ----------------------------
# Qdrant Vector DB
# ----------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None


def ensure_qdrant():
    if QdrantClient is None:
        raise RuntimeError(
            "Qdrant client is not installed. Install with `uv pip install qdrant-client`"
        )


# ----------------------------
# Utilities
# ----------------------------

def now_device():
    """Prefer CUDA, then MPS (Apple Silicon), then CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_everything(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Passage Encoder
# ----------------------------

class PassageEncoder:
    """Passage encoder using sentence transformers or HF models"""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device=None):
        self.device = device or now_device()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        self.out_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Encode texts to embeddings using CLS pooling"""
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)

        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Encoding", total=len(texts) // batch_size + 1)
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.encoder(**inputs)
            # CLS pooling
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        return self.encode([query])[0]


# ----------------------------
# Qdrant Index Manager
# ----------------------------

class QdrantIndex:
    """Manages Qdrant vector database for RAG"""

    COLLECTION_NAME = "rag_passages"

    def __init__(self, index_dir: str):
        ensure_qdrant()
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Use local persistent storage
        self.db_path = self.index_dir / "qdrant_db"
        self.client = QdrantClient(path=str(self.db_path))

        # Store texts separately for retrieval
        self.texts_path = self.index_dir / "texts.json"
        self.texts: List[str] = []

    def build_index(self, texts: List[str], embeddings: np.ndarray):
        """Build Qdrant index from texts and their embeddings"""
        self.texts = texts
        dim = embeddings.shape[1]

        # Recreate collection
        if self.client.collection_exists(self.COLLECTION_NAME):
            self.client.delete_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

        # Add points in batches
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            points = [
                PointStruct(
                    id=j,
                    vector=embeddings[j].tolist(),
                    payload={"text": texts[j], "idx": j}
                )
                for j in range(i, end_idx)
            ]
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

        # Save texts for later retrieval
        with open(self.texts_path, 'w') as f:
            json.dump(texts, f)

        print(f"[index] Built Qdrant index with {len(texts)} passages, dim={dim}")

    def load(self):
        """Load existing index"""
        if self.texts_path.exists():
            with open(self.texts_path, 'r') as f:
                self.texts = json.load(f)
        return self

    def search(self, query_embedding: np.ndarray, topk: int = 4) -> Tuple[List[str], List[float]]:
        """Search for similar passages"""
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=topk
        ).points

        passages = [hit.payload["text"] for hit in results]
        scores = [hit.score for hit in results]

        return passages, scores


# ----------------------------
# RAG Generator
# ----------------------------

class RAGGenerator:
    """Standard RAG generator using retrieved passages"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B",
        device=None,
        max_ctx_tokens: int = 2048
    ):
        self.device = device or now_device()
        self.model_name = model_name
        self.max_ctx_tokens = max_ctx_tokens

        print(f"[RAG] Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map="auto" if self.device.type == 'cuda' else None
        )

        if self.device.type != 'cuda':
            self.model = self.model.to(self.device)

        self.model.eval()

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, question: str, passages: List[str]) -> str:
        """Build RAG prompt with retrieved passages"""
        context = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])

        prompt = f"""Use the following passages to answer the question. Be concise and accurate.

Passages:
{context}

Question: {question}

Answer:"""
        return prompt

    @torch.no_grad()
    def generate(
        self,
        question: str,
        passages: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> Dict:
        """Generate answer using retrieved passages"""

        prompt = self.build_prompt(question, passages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ctx_tokens
        ).to(self.device)

        input_len = inputs.input_ids.shape[1]

        # Timing
        start_time = time.perf_counter()

        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(inputs.input_ids, **gen_kwargs)

        end_time = time.perf_counter()

        # Decode
        generated_ids = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Metrics
        num_tokens = len(generated_ids)
        total_time = end_time - start_time

        return {
            "answer": answer,
            "TTFT_sec": total_time / max(num_tokens, 1),  # Approximate
            "total_time_sec": total_time,
            "throughput_tok_per_sec": num_tokens / total_time if total_time > 0 else 0,
            "num_tokens": num_tokens,
            "input_tokens": input_len
        }


# ----------------------------
# CLI Commands
# ----------------------------

def cmd_index(args):
    """Build Qdrant index from corpus"""
    seed_everything()

    # Load corpus
    print(f"[index] Loading corpus from {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"[index] Loaded {len(texts)} passages")

    # Encode
    print(f"[index] Encoding with {args.embed_model}")
    encoder = PassageEncoder(args.embed_model)
    embeddings = encoder.encode(texts, show_progress=True)

    # Build index
    index = QdrantIndex(args.index_dir)
    index.build_index(texts, embeddings)

    print(f"[index] Done! Index saved to {args.index_dir}")


def cmd_generate(args):
    """Generate answer for a question"""
    seed_everything()

    # Load index
    print(f"[generate] Loading index from {args.index_dir}")
    index = QdrantIndex(args.index_dir).load()

    # Encode query
    encoder = PassageEncoder(args.embed_model)
    query_emb = encoder.encode_query(args.question)

    # Retrieve
    passages, scores = index.search(query_emb, topk=args.topk)

    # Generate
    generator = RAGGenerator(
        model_name=args.dec,
        max_ctx_tokens=args.ctx_max
    )

    result = generator.generate(
        question=args.question,
        passages=passages,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Output
    output = {
        "question": args.question,
        "passages": passages,
        "scores": scores,
        **result
    }

    print(json.dumps(output, indent=2))


def cmd_evaluate(args):
    """Evaluate RAG on a test set"""
    seed_everything()

    # Setup MLflow tracking if enabled
    tracker = None
    if args.use_mlflow:
        tracker = MLflowTracker(
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
            run_name=f"eval_{args.run_name}"
        )
        tracker.start_run(params={
            "model": args.dec,
            "embed_model": args.embed_model,
            "topk": args.topk,
            "ctx_max": args.ctx_max,
            "max_new": args.max_new,
            "max_samples": args.max_samples or "all"
        })

    try:
        # Load test data
        print(f"[evaluate] Loading test data from {args.test_json}")
        test_data = []
        with open(args.test_json, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))

        # Limit samples if specified
        if args.max_samples and args.max_samples < len(test_data):
            import random
            random.shuffle(test_data)
            test_data = test_data[:args.max_samples]

        print(f"[evaluate] Evaluating on {len(test_data)} samples")

        # Load index
        index = QdrantIndex(args.index_dir).load()
        encoder = PassageEncoder(args.embed_model)
        generator = RAGGenerator(model_name=args.dec, max_ctx_tokens=args.ctx_max)

        # Metrics
        results = []
        correct = 0
        total_time = 0
        total_tokens = 0

        for i, item in enumerate(test_data):
            question = item["question"]
            expected_answers = item.get("answers", [])

            # Retrieve
            query_emb = encoder.encode_query(question)
            passages, scores = index.search(query_emb, topk=args.topk)

            # Generate
            result = generator.generate(
                question=question,
                passages=passages,
                max_new_tokens=args.max_new,
                temperature=0.0  # Deterministic for eval
            )

            # Check accuracy (simple exact match)
            answer = result["answer"].strip()
            is_correct = any(
                exp.lower() in answer.lower()
                for exp in expected_answers
            ) if expected_answers else None

            if is_correct:
                correct += 1

            total_time += result["total_time_sec"]
            total_tokens += result["num_tokens"]

            results.append({
                "id": item.get("id", i),
                "question": question,
                "expected": expected_answers,
                "answer": answer,
                "correct": is_correct,
                "time_sec": result["total_time_sec"]
            })

            if (i + 1) % 10 == 0:
                current_acc = correct / (i + 1)
                print(f"[evaluate] Progress: {i+1}/{len(test_data)} | Accuracy: {current_acc:.2%}")

                # Log intermediate metrics to MLflow
                if tracker:
                    tracker.log_metrics({
                        "running_accuracy": current_acc,
                        "samples_processed": i + 1
                    }, step=i + 1)

        # Summary
        accuracy = correct / len(test_data) if test_data else 0
        avg_time = total_time / len(test_data) if test_data else 0
        throughput = total_tokens / total_time if total_time > 0 else 0

        summary = {
            "total_samples": len(test_data),
            "correct": correct,
            "accuracy": accuracy,
            "avg_time_per_question_sec": avg_time,
            "total_time_sec": total_time,
            "throughput_tok_per_sec": throughput
        }

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(json.dumps(summary, indent=2))

        # Log final metrics to MLflow
        if tracker:
            tracker.log_metrics({
                "accuracy": accuracy,
                "avg_time_per_question_sec": avg_time,
                "total_time_sec": total_time,
                "throughput_tok_per_sec": throughput,
                "total_samples": len(test_data),
                "correct": correct
            })

        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"summary": summary, "results": results}, f, indent=2)
            print(f"\n[evaluate] Detailed results saved to {args.output}")

            # Log results file as artifact
            if tracker:
                tracker.log_artifact(args.output)

    finally:
        if tracker:
            tracker.end_run()


# ----------------------------
# Argparse
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Standard RAG baseline (Qdrant + HuggingFace) for comparison with REFRAG"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    sp = sub.add_parser("index", help="Build Qdrant index from corpus")
    sp.add_argument("--corpus", type=str, required=True, help="Text file, one passage per line")
    sp.add_argument("--index_dir", type=str, required=True, help="Output directory for Qdrant index")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.set_defaults(func=cmd_index)

    # generate
    sp = sub.add_parser("generate", help="RAG generate with retrieval")
    sp.add_argument("--index_dir", type=str, required=True, help="Directory containing Qdrant index")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B", help="Decoder/LLM model")
    sp.add_argument("--question", type=str, required=True)
    sp.add_argument("--topk", type=int, default=4, help="Number of passages to retrieve")
    sp.add_argument("--ctx_max", type=int, default=2048, help="Max context tokens")
    sp.add_argument("--max_new", type=int, default=128, help="Max new tokens to generate")
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--top_p", type=float, default=1.0)
    sp.set_defaults(func=cmd_generate)

    # evaluate
    sp = sub.add_parser("evaluate", help="Evaluate RAG on test set")
    sp.add_argument("--index_dir", type=str, required=True)
    sp.add_argument("--test_json", type=str, required=True, help="JSONL with {'question':..., 'answers':...}")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--topk", type=int, default=4)
    sp.add_argument("--ctx_max", type=int, default=2048)
    sp.add_argument("--max_new", type=int, default=128)
    sp.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    sp.add_argument("--output", type=str, default=None, help="Output file for detailed results")
    # MLflow arguments
    sp.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    sp.add_argument("--mlflow-uri", type=str, default="mlruns", help="MLflow tracking URI")
    sp.add_argument("--experiment", type=str, default="RAG_Baseline", help="MLflow experiment name")
    sp.add_argument("--run-name", type=str, default="run", help="MLflow run name")
    sp.set_defaults(func=cmd_evaluate)

    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
