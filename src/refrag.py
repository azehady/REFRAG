#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REFRAG-style RAG (compress → sense/select → expand) — Single-file reference implementation

This script reconstructs a REFRAG-style retrieval-augmented generation pipeline based on the
first 11 pages of the provided paper (compress context with encoder-produced chunk embeddings,
project those to decoder token space, selectively re-expand informative chunks, and decode).
It includes:
  - FAISS-based retrieval index (build + search)
  - Encoder-side chunk embeddings (CLS pooling) + projection to decoder embedding dimension
  - Selective expansion via a tiny policy net (REINFORCE) with a strong heuristic fallback
  - Continual pretraining (CPT) curricula: reconstruction → next-paragraph prediction
  - Generation with TTFT/TTIT/throughput measurements
  - Full CLI with subcommands

USAGE (examples):
  # 0) Install deps (adjust CUDA wheel index if needed)
  #    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  #    pip install transformers==4.43.3 accelerate datasets sentencepiece faiss-cpu sacrebleu numpy

  # 1) Build a local FAISS index from a text corpus (1 doc per line)
  #    python refrag.py index --corpus data/wiki_lines.txt --index_dir runs/index --embed_model BAAI/bge-small-en-v1.5

  # 2) Continual pretraining (CPT) phase A: Reconstruction curriculum (freeze decoder)
  #    python refrag.py cpt_recon --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-1B

  # 3) Continual pretraining (CPT) phase B: Next-paragraph prediction curriculum (unfreeze decoder)
  #    python refrag.py cpt_next --train_json data/cpt_train.jsonl --enc roberta-base --dec meta-llama/Llama-3.2-1B

  # 4) Optional: train the RL policy that decides selective expansion (REINFORCE, reward=-PPL)
  #    python refrag.py train_policy --rag_json data/rag_train.jsonl --index_dir runs/index --topk 8

  # 5) RAG generate (with compression rate k and policy-driven expansion fraction p)
  #    python refrag.py generate --index_dir runs/index --question "Who discovered penicillin?" --topk 8 --k 16 --p 0.25

Data formats:
  - cpt_* expects JSONL with fields:
      {"id":"...", "tokens":"<long text>", "split":{"s":2048,"o":256}}
  - rag_* expects JSONL with fields:
      {"id":"...", "question":"...", "answers":["..."]}  # answers optional
  - index corpus: plain text file with one passage per line (≤ ~200 words).

Notes:
  - Default model IDs use Hugging Face Hub; for offline use, point to local directories.
  - For long contexts, PyTorch 2.1+ torch.compile may improve speed.
  - This implementation is designed for clarity + completeness; tune as needed.

Author: Matthew Combatti - Simulanics Technologies
"""
import os, sys, json, math, time, random, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

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
    """MLflow experiment tracking wrapper for REFRAG."""

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "REFRAG",
        run_name: Optional[str] = None
    ):
        ensure_mlflow()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name or f"refrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    def log_model(self, model_dir: str, artifact_path: str = "model"):
        """Log model directory as artifact."""
        mlflow.log_artifacts(model_dir, artifact_path)

    def end_run(self):
        """End MLflow run."""
        if self.run:
            mlflow.end_run()
            print("[MLflow] Ended run")


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_device():
    # Prefer CUDA (includes ROCm builds), then Apple MPS, then CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def ensure_faiss():
    if faiss is None:
        raise RuntimeError(
            "FAISS is not installed. Install with `pip install faiss-cpu` (or faiss-gpu)."
        )


# ----------------------------
# Retrieval (FAISS + encoder)
# ----------------------------

class PassageEncoder(nn.Module):
    """Passage encoder that returns a fixed vector per passage using CLS pooling."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device=None):
        super().__init__()
        self.device = device or now_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.out_dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode_passages(self, texts: List[str], bs: int = 32) -> np.ndarray:
        self.encoder.eval()
        if not texts:
            return np.zeros((0, self.out_dim), dtype=np.float32)
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            toks = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = self.encoder(**toks).last_hidden_state
            emb = out[:, 0, :]  # CLS
            emb = F.normalize(emb, dim=-1)
            vecs.append(emb.detach().cpu().float().numpy())
        return np.concatenate(vecs, axis=0)

    @torch.no_grad()
    def encode_query(self, text: str) -> np.ndarray:
        v = self.encode_passages([text], bs=1)
        return v[0] if len(v) else np.zeros((self.out_dim,), dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray, index_path: str):
    ensure_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ≈ cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str):
    ensure_faiss()
    return faiss.read_index(index_path)


def search_index(index, query_vec: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    ensure_faiss()
    q = query_vec.astype(np.float32)[None, :]
    faiss.normalize_L2(q)
    D, I = index.search(q, topk)
    return D[0], I[0]


# ----------------------------
# REFRAG Core
# ----------------------------

@dataclass
class REFRAGConfig:
    encoder_name: str = "roberta-base"
    decoder_name: str = "meta-llama/Llama-3.2-3B"
    chunk_len_tokens: int = 64     # k
    max_q_tokens: int = 256
    max_ctx_tokens: int = 2048     # s (pre-chunked, before compression)
    max_out_tokens: int = 256      # o
    selective_p: float = 0.25      # fraction cap for expansions
    policy_hidden: int = 256
    lr: float = 2e-5
    wd: float = 0.0
    grad_clip: float = 1.0
    fp16: bool = True
    seed: int = 1337


class ChunkEncoder(nn.Module):
    """Encoder that returns one vector per text chunk via CLS pooling."""
    def __init__(self, name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModel.from_pretrained(name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, texts: List[str], device=None) -> torch.Tensor:
        device = device or next(self.model.parameters()).device
        if len(texts) == 0:
            return torch.zeros((0, self.out_dim), device=device)
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        h = self.model(**toks).last_hidden_state[:, 0, :]  # [CLS]
        h = F.normalize(h, dim=-1)
        return h


class TokenProjector(nn.Module):
    """Projection ϕ: encoder-dim → decoder token-embedding dim."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.proj(x)


class SelectPolicy(nn.Module):
    """
    Tiny policy π(ci) that outputs expansion prob per chunk.
    Input: chunk embedding ci (encoder space) + scalar pos (normalized [0,1]).
    Output: logits ∈ R (Bernoulli).
    """
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, c: torch.Tensor, pos01: torch.Tensor) -> torch.Tensor:
        x = torch.cat([c, pos01], dim=-1)
        return self.net(x).squeeze(-1)  # [L]


class REFRAG(nn.Module):
    """
    Builds decoder inputs consisting of:
      - question token embeddings (normal)
      - per-chunk compressed embeddings (projected from encoder) OR full token embeddings (expanded)
    """
    def __init__(self, cfg: REFRAGConfig):
        super().__init__()
        self.cfg = cfg
        self.device = now_device()

        # Modules
        self.encoder = ChunkEncoder(cfg.encoder_name).to(self.device)
        self.decoder_tok = AutoTokenizer.from_pretrained(cfg.decoder_name, use_fast=True)
        self.decoder = AutoModelForCausalLM.from_pretrained(cfg.decoder_name).to(self.device)

        self.dec_embed_dim = self.decoder.get_input_embeddings().weight.shape[1]
        self.projector = TokenProjector(self.encoder.out_dim, self.dec_embed_dim).to(self.device)
        self.policy = SelectPolicy(self.encoder.out_dim, hidden=cfg.policy_hidden).to(self.device)

        self.eos_id = self.decoder_tok.eos_token_id
        self.pad_id = self.decoder_tok.pad_token_id or self.decoder_tok.eos_token_id
        self.bos_id = self.decoder_tok.bos_token_id or self.decoder_tok.eos_token_id

    def _tokenize(self, text: str, max_len: int) -> Dict[str, torch.Tensor]:
        return self.decoder_tok(text, truncation=True, max_length=max_len, padding=False, return_tensors="pt")

    @torch.no_grad()
    def _decoder_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings()(input_ids.to(self.device))

    def _chunk_text(self, text: str, k_tokens: int) -> Tuple[List[str], List[torch.Tensor]]:
        toks = self.decoder_tok(text, truncation=True, max_length=self.cfg.max_ctx_tokens, return_tensors="pt")
        ids = toks.input_ids[0]  # [S]
        id_chunks = [ids[i:i+k_tokens] for i in range(0, ids.size(0), k_tokens)]
        str_chunks = [self.decoder_tok.decode(ch, skip_special_tokens=True) for ch in id_chunks]
        return str_chunks, id_chunks

    def _encode_chunks(self, chunk_strs: List[str]) -> torch.Tensor:
        return self.encoder(chunk_strs, device=self.device)

    def _project_chunks(self, c: torch.Tensor) -> torch.Tensor:
        return self.projector(c)

    def _select_expand_mask(self, c: torch.Tensor, p_max: float) -> torch.Tensor:
        L = c.size(0)
        if L == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        pos01 = torch.linspace(0, 1, steps=L, device=c.device).unsqueeze(-1)
        logits = self.policy(c, pos01)          # [L]
        probs = torch.sigmoid(logits)
        sample = torch.bernoulli(probs).bool()
        if p_max > 0.0:
            max_expand = max(0, int(round(p_max * L)))
            if sample.sum().item() > max_expand:
                topk = torch.topk(logits, k=max_expand).indices
                mask = torch.zeros_like(sample)
                mask[topk] = True
                sample = mask.bool()
        return sample

    def _heuristic_select(self, chunk_ids: List[torch.Tensor], q_text: str, p_max: float) -> torch.Tensor:
        L = len(chunk_ids)
        if L == 0 or p_max <= 0:
            return torch.zeros(L, dtype=torch.bool, device=self.device)
        scores = []
        with torch.no_grad():
            for ch in chunk_ids:
                inp = torch.cat([
                    self._tokenize(q_text, self.cfg.max_q_tokens).input_ids[0].to(self.device),
                    ch.to(self.device)
                ], dim=0).unsqueeze(0)
                labels = inp.clone()
                out = self.decoder(input_ids=inp, labels=labels)
                ppl = torch.exp(out.loss).item()
                scores.append(ppl)
        scores = np.asarray(scores)
        k = max(1, int(round(p_max * L)))
        top_idx = scores.argsort()[::-1][:k]  # expand highest perplexity chunks
        mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        mask[top_idx] = True
        return mask

    def build_decoder_inputs(self, question: str, passages: List[str], k: int, p: float, use_policy: bool = True) -> Tuple[torch.Tensor, Dict]:
        # 0) BOS token embedding - critical for LLM to understand sequence start
        bos_emb = self._decoder_token_embeddings(torch.tensor([[self.bos_id]], device=self.device))  # [1,1,D]

        # 1) Instruction prefix - helps the model understand the task
        instruction = "Use the following passages to answer the question. Be concise and accurate.\n\nPassages:\n"
        instr_ids = self._tokenize(instruction, 64).input_ids.to(self.device)
        instr_emb = self._decoder_token_embeddings(instr_ids)  # [1,I,D]

        # 2) Context → chunk (with proper passage formatting like RAG)
        # Format each passage with a number prefix like RAG does: "[1] passage1\n\n[2] passage2..."
        formatted_passages = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        ctx_text = formatted_passages
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        L = len(chunk_strs)

        # 3) Encode → project
        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)  # [L, D_enc]
            ecnk = self._project_chunks(c)       # [L, D_dec]

        # 4) Select expansions
        if use_policy:
            expand_mask = self._select_expand_mask(c, p_max=p)
        else:
            expand_mask = self._heuristic_select(chunk_ids, q_text=question, p_max=p)

        # 5) Build context embeddings (compressed or expanded)
        ctx_embs = []
        seg_flags = []                 # bookkeeping for diagnostics
        for i, ids in enumerate(chunk_ids):
            if expand_mask[i]:
                tok_emb = self._decoder_token_embeddings(ids.unsqueeze(0))  # [1,t_i,D]
                ctx_embs.append(tok_emb.squeeze(0))
                seg_flags.extend([1] * tok_emb.size(1))
            else:
                ctx_embs.append(ecnk[i].unsqueeze(0))  # single compressed slot
                seg_flags.append(0)

        # 6) Question with "Answer:" suffix to prime generation
        q_prompt = f"\n\nQuestion: {question}\n\nAnswer:"
        q_ids = self._tokenize(q_prompt, self.cfg.max_q_tokens).input_ids.to(self.device)
        q_emb = self._decoder_token_embeddings(q_ids)  # [1,Q,D]

        # 7) Final sequence: BOS → Instruction → Context → Question
        # Order: [BOS][instruction][compressed/expanded context][question + Answer:]
        seq_embs = [bos_emb.squeeze(0), instr_emb.squeeze(0)] + ctx_embs + [q_emb.squeeze(0)]
        final = torch.cat(seq_embs, dim=0).unsqueeze(0)  # [1, T', D]

        extras = {
            "expand_mask": expand_mask.detach().cpu().numpy().tolist(),
            "num_chunks": L,
            "token_positions_flag": seg_flags,
        }
        return final, extras

    @torch.no_grad()
    def generate(self, question: str, passages: List[str], k: int, p: float,
                 max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0,
                 use_policy: bool = True) -> Dict:
        self.decoder.eval()

        # If p >= 1.0 (or very close), use standard RAG-style generation with input_ids
        # This avoids the inputs_embeds issues that cause hallucinations
        if p >= 0.99:
            return self._generate_rag_style(question, passages, max_new_tokens, temperature, top_p)

        emb_in, extras = self.build_decoder_inputs(question, passages, k=k, p=p, use_policy=use_policy)

        # Prefill → KV cache
        t0 = time.time()
        out = self.decoder(inputs_embeds=emb_in, use_cache=True)
        past_key_values = out.past_key_values
        ttft = time.time() - t0

        generated = []
        ttit_list = []

        # Get first token from prefill output (not EOS!)
        logits = out.logits[:, -1, :]
        if temperature > 0.0:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        nid = next_id.item()
        if nid != self.eos_id:
            generated.append(nid)

        # Continue generating
        for _ in range(max_new_tokens - 1):
            if nid == self.eos_id:
                break
            step_emb = self.decoder.get_input_embeddings()(next_id)
            t1 = time.time()
            out = self.decoder(inputs_embeds=step_emb, use_cache=True, past_key_values=past_key_values)
            ttit_list.append(time.time() - t1)

            logits = out.logits[:, -1, :]
            past_key_values = out.past_key_values
            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid == self.eos_id:
                break
            generated.append(nid)

        text = self.decoder_tok.decode(generated, skip_special_tokens=True)
        throughput = (len(generated) / max(sum(ttit_list), 1e-6)) if ttit_list else 0.0
        return {
            "answer": text.strip(),
            "TTFT_sec": ttft,
            "TTIT_avg_sec": float(np.mean(ttit_list)) if ttit_list else 0.0,
            "throughput_tok_per_sec": throughput,
            "meta": extras,
        }

    @torch.no_grad()
    def _generate_rag_style(self, question: str, passages: List[str],
                            max_new_tokens: int = 128, temperature: float = 0.0, top_p: float = 1.0) -> Dict:
        """Standard RAG-style generation using input_ids (no compression/embedding mixing)."""
        # Build prompt exactly like RAG
        context = "\n\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
        prompt = f"""Use the following passages to answer the question. Be concise and accurate.

Passages:
{context}

Question: {question}

Answer:"""

        inputs = self.decoder_tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_ctx_tokens
        ).to(self.device)

        input_len = inputs.input_ids.shape[1]

        t0 = time.time()

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.decoder_tok.pad_token_id or self.decoder_tok.eos_token_id,
            "eos_token_id": self.eos_id,
        }

        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.decoder.generate(inputs.input_ids, **gen_kwargs)

        ttft = time.time() - t0

        generated_ids = outputs[0][input_len:]
        text = self.decoder_tok.decode(generated_ids, skip_special_tokens=True).strip()

        num_tokens = len(generated_ids)
        throughput = num_tokens / max(ttft, 1e-6)

        return {
            "answer": text,
            "TTFT_sec": ttft,
            "TTIT_avg_sec": ttft / max(num_tokens, 1),
            "throughput_tok_per_sec": throughput,
            "meta": {"expand_mask": [True] * len(passages), "num_chunks": len(passages), "mode": "rag_style"},
        }

    # ----------------------------
    # Losses for CPT & RL policy
    # ----------------------------
    def loss_reconstruction(self, ctx_text: str, k: int, num_chunks_cap: Optional[int] = None) -> torch.Tensor:
        """
        Train encoder+projector to reconstruct tokens chunk-by-chunk.

        The approach: For each chunk i, we build context from compressed embeddings
        of chunks [0..i-1] (if any) plus the current chunk's compressed embedding,
        then predict the tokens of chunk i. This teaches the model to use compressed
        context sequentially.
        """
        # 1) Chunk the context in decoder token space
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        if num_chunks_cap is not None:
            chunk_strs = chunk_strs[:num_chunks_cap]
            chunk_ids = chunk_ids[:num_chunks_cap]
        L = len(chunk_strs)
        if L == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # 2) Encode chunks (encoder space) → project to decoder embedding space
        c = self._encode_chunks(chunk_strs)      # [L, D_enc]
        e = self._project_chunks(c)              # [L, D_dec]

        # 3) Per-chunk reconstruction loss with sequential context
        loss_accum = 0.0
        for i, ids in enumerate(chunk_ids):
            # Build context: compressed embeddings from previous chunks + current chunk embedding
            # This creates a sequence: [c0, c1, ..., c_{i-1}, c_i] to predict tokens of chunk i
            if i == 0:
                # For first chunk, just use its compressed embedding
                ctx_emb = e[0].unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            else:
                # Use all previous chunk embeddings as context
                ctx_emb = e[:i+1].unsqueeze(0)  # [1, i+1, D]

            # Get token embeddings for the chunk we want to reconstruct (teacher forcing)
            chunk_token_ids = ids.to(self.device)
            chunk_token_embs = self._decoder_token_embeddings(chunk_token_ids.unsqueeze(0))  # [1, T, D]

            # Concatenate: [compressed_context] + [chunk_tokens]
            full_emb = torch.cat([ctx_emb, chunk_token_embs], dim=1)  # [1, ctx_len + T, D]

            # Labels: -100 for context positions, actual tokens for reconstruction
            ctx_len = ctx_emb.size(1)
            T = chunk_token_ids.size(0)
            labels = torch.full((1, ctx_len + T), -100, dtype=torch.long, device=self.device)
            labels[0, ctx_len:] = chunk_token_ids  # Only compute loss on chunk tokens

            out = self.decoder(inputs_embeds=full_emb, labels=labels)
            loss_accum = loss_accum + out.loss

        return loss_accum / max(L, 1)


    def loss_next_para(self, full_text: str, s: int, o: int, k: int, expand_frac: float = 0.0) -> torch.Tensor:
        """Feed first s tokens (compressed) and predict next o tokens (teacher-forced).

        If the text is shorter than s+o tokens, we adaptively split it:
        - Use 80% of tokens for context, 20% for prediction
        - Minimum: at least 2*k context tokens and k output tokens
        """
        toks = self.decoder_tok(full_text, truncation=True, max_length=s + o, return_tensors="pt")
        ids = toks.input_ids[0].to(self.device)
        total_len = ids.size(0)

        # Adaptive split: if text is shorter than s+o, use 80/20 split
        min_ctx = max(2 * k, 32)  # At least 2 chunks for context
        min_out = max(k, 16)       # At least 1 chunk for output

        if total_len < min_ctx + min_out:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Use provided s/o if text is long enough, otherwise adaptive split
        if total_len >= s + 2:
            ctx_len = s
            out_len = min(o, total_len - s)
        else:
            # Adaptive: 80% context, 20% output
            ctx_len = int(total_len * 0.8)
            out_len = total_len - ctx_len
            # Ensure minimum sizes
            if out_len < min_out:
                out_len = min_out
                ctx_len = total_len - out_len

        ctx_ids = ids[:ctx_len]
        out_ids = ids[ctx_len:ctx_len + out_len]
        ctx_str = self.decoder_tok.decode(ctx_ids, skip_special_tokens=True)

        chunk_strs, chunk_ids = self._chunk_text(ctx_str, k_tokens=k)
        c = self._encode_chunks(chunk_strs)
        e = self._project_chunks(c)

        L = len(chunk_ids)
        expand_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        if L > 0 and expand_frac > 0.0:
            top = max(1, int(round(expand_frac * L)))
            lengths = torch.tensor([len(ch) for ch in chunk_ids], device=self.device)
            top_idx = torch.topk(lengths, k=min(top, L)).indices
            expand_mask[top_idx] = True

        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if expand_mask[i]:
                seq.append(self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0))
            else:
                seq.append(e[i].unsqueeze(0))
        if len(seq) == 0:
            seq.append(self._decoder_token_embeddings(ctx_ids.unsqueeze(0)).squeeze(0))

        # Get output token embeddings for teacher forcing
        out_embs = self._decoder_token_embeddings(out_ids.unsqueeze(0)).squeeze(0)  # [out_len, D]

        # Concatenate: context embeddings + output embeddings
        ctx_embs = torch.cat(seq, dim=0)  # [ctx_len_compressed, D]
        full_embs = torch.cat([ctx_embs, out_embs], dim=0)  # [ctx + out, D]
        inp = full_embs.unsqueeze(0)  # [1, ctx + out, D]

        # Labels: -100 for context positions (no loss), actual token IDs for output
        ctx_labels = torch.full((ctx_embs.size(0),), -100, dtype=torch.long, device=self.device)
        full_labels = torch.cat([ctx_labels, out_ids], dim=0).unsqueeze(0)  # [1, ctx + out]

        out = self.decoder(inputs_embeds=inp, labels=full_labels)
        return out.loss

    def policy_step(self, question: str, passages: List[str], k: int, max_expand_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """One REINFORCE step: sample expansion mask, compute reward = -PPL of supervised continuation."""
        ctx_text = "\n".join(passages)
        chunk_strs, chunk_ids = self._chunk_text(ctx_text, k_tokens=k)
        if len(chunk_strs) == 0:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # ---- build compressed/expanded context sequence (no grad) ----
        with torch.no_grad():
            c = self._encode_chunks(chunk_strs)  # [L, Denc]
        L = c.size(0)
        pos01 = torch.linspace(0, 1, steps=L, device=self.device).unsqueeze(-1)
        logits = self.policy(c, pos01)          # [L]
        probs = torch.sigmoid(logits)
        bern = torch.distributions.Bernoulli(probs=probs)
        sample = bern.sample()                   # [L]

        max_expand = max(1, int(round(max_expand_frac * L)))
        if sample.sum().item() > max_expand:
            top_idx = torch.topk(logits, k=max_expand).indices
            mask = torch.zeros_like(sample)
            mask[top_idx] = 1.0
            sample = mask
        log_prob = bern.log_prob(sample).sum()

        with torch.no_grad():
            e = self._project_chunks(c)          # [L, Ddec]
        seq = []
        for i, ids_i in enumerate(chunk_ids):
            if sample[i] > 0.5:
                seq.append(self._decoder_token_embeddings(ids_i.unsqueeze(0)).squeeze(0))  # expanded tokens
            else:
                seq.append(e[i].unsqueeze(0))  # one-slot compressed chunk
        ctx_emb = torch.cat(seq, dim=0).unsqueeze(0)  # [1, T_ctx, D]

        # ---- prepend question embeddings (no grad) ----
        q_ids = self._tokenize(question, self.cfg.max_q_tokens).input_ids.to(self.device)
        with torch.no_grad():
            q_emb = self._decoder_token_embeddings(q_ids)  # [1, Q, D]
        dec_in = torch.cat([q_emb, ctx_emb], dim=1)        # [1, T_ctx+Q, D]

        # ---- build a short "target" continuation to score (no grad) ----
        with torch.no_grad():
            # quick greedy rollout conditioned on dec_in to synthesize a target
            out = self.decoder(inputs_embeds=dec_in, use_cache=True)
            past = out.past_key_values
            rollout = []
            last = torch.tensor([[self.eos_id]], device=self.device)
            for _ in range(32):
                step_emb = self.decoder.get_input_embeddings()(last)
                o2 = self.decoder(inputs_embeds=step_emb, use_cache=True, past_key_values=past)
                last = torch.argmax(o2.logits[:, -1, :], dim=-1, keepdim=True)
                nid = last.item()
                if nid == self.eos_id:
                    break
                rollout.append(nid)
                past = o2.past_key_values
            target = torch.tensor([rollout[:16] or [self.eos_id]], device=self.device, dtype=torch.long)  # [1, T_tgt]

        # ---- compute reward = -PPL with proper masked labels (no grad) ----
        with torch.no_grad():
            tgt_emb = self._decoder_token_embeddings(target)                        # [1, T_tgt, D]
            inputs = torch.cat([dec_in, tgt_emb], dim=1)                            # [1, T_ctx+Q+T_tgt, D]
            labels = torch.full((1, inputs.size(1)), -100, dtype=torch.long, device=self.device)
            labels[0, dec_in.size(1):dec_in.size(1) + target.size(1)] = target[0]   # only supervise target span
            out2 = self.decoder(inputs_embeds=inputs, labels=labels)
            ppl = torch.exp(out2.loss.detach())

        reward = -ppl
        return log_prob, reward



# ----------------------------
# Optim / Training helpers
# ----------------------------

def setup_optim(params, lr, wd, total_steps):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)
    return opt, sch


# ----------------------------
# Index build / load helpers
# ----------------------------

def cmd_index(args):
    seed_everything()
    enc = PassageEncoder(args.embed_model)
    with open(args.corpus, "r", encoding="utf-8") as f:
        passages = [ln.strip() for ln in f if ln.strip()]
    embs = enc.encode_passages(passages, bs=64)
    os.makedirs(args.index_dir, exist_ok=True)
    np.save(os.path.join(args.index_dir, "texts.npy"), np.array(passages, dtype=object))
    build_faiss_index(embs, os.path.join(args.index_dir, "faiss.index"))
    print(f"[index] built with {len(passages)} passages → {args.index_dir}")


def load_index_bundle(index_dir: str):
    texts = np.load(os.path.join(index_dir, "texts.npy"), allow_pickle=True).tolist()
    index = load_faiss_index(os.path.join(index_dir, "faiss.index"))
    return texts, index


# ----------------------------
# CLI Commands
# ----------------------------

def curriculum_schedule(total_steps: int, max_chunks: int):
    """Simple linear curriculum over steps: 1 → max_chunks."""
    plan = []
    for t in range(total_steps):
        c = 1 + int((max_chunks - 1) * (t / max(1, total_steps - 1)))
        plan.append(c)
    return plan


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def cmd_cpt_recon(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
    )

    # Setup MLflow tracking if enabled
    tracker = None
    if args.use_mlflow:
        tracker = MLflowTracker(
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
            run_name=f"recon_{args.run_name}"
        )
        tracker.start_run(params={
            "encoder": args.enc,
            "decoder": args.dec,
            "chunk_len_k": args.k,
            "lr": args.lr,
            "steps": args.steps,
            "phase": "reconstruction"
        })

    try:
        model = REFRAG(cfg).to(now_device())
        # Freeze decoder; train encoder+projector
        for p in model.decoder.parameters():
            p.requires_grad = False
        params = list(model.encoder.parameters()) + list(model.projector.parameters())
        steps = args.steps
        opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)

        data = list(load_jsonl(args.train_json))
        if len(data) == 0:
            print("[cpt_recon] no data.")
            return

        model.train()
        running_loss = 0.0
        for step in range(steps):
            ex = random.choice(data)
            text = ex["tokens"]
            chunk_strs, _ = model._chunk_text(text, k_tokens=cfg.chunk_len_tokens)
            max_chunks = max(1, len(chunk_strs))
            cap = curriculum_schedule(steps, max_chunks)[step]
            loss = model.loss_reconstruction(text, k=cfg.chunk_len_tokens, num_chunks_cap=cap)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sch.step()

            running_loss += loss.item()

            if step % max(1, args.log_every) == 0:
                avg_loss = running_loss / max(1, args.log_every)
                print(f"[cpt_recon] step {step}/{steps} loss={loss.item():.4f}")

                if tracker:
                    tracker.log_metrics({
                        "reconstruction_loss": loss.item(),
                        "avg_loss": avg_loss,
                        "curriculum_cap": cap,
                        "learning_rate": sch.get_last_lr()[0]
                    }, step=step)

                running_loss = 0.0

        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.encoder.state_dict(), os.path.join(args.out_dir, "encoder.pt"))
        torch.save(model.projector.state_dict(), os.path.join(args.out_dir, "projector.pt"))
        print(f"[cpt_recon] saved to {args.out_dir}")

        if tracker:
            tracker.log_model(args.out_dir, "reconstruction_model")

    finally:
        if tracker:
            tracker.end_run()


def cmd_cpt_next(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
    )

    # Setup MLflow tracking if enabled
    tracker = None
    if args.use_mlflow:
        tracker = MLflowTracker(
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
            run_name=f"cpt_{args.run_name}"
        )
        tracker.start_run(params={
            "encoder": args.enc,
            "decoder": args.dec,
            "chunk_len_k": args.k,
            "lr": args.lr,
            "steps": args.steps,
            "expand_frac": args.expand_frac,
            "phase": "cpt_next_para",
            "use_curriculum": args.use_curriculum,
        })

    try:
        model = REFRAG(cfg).to(now_device())
        # Load from recon phase if provided
        if args.load_dir:
            enc_p = os.path.join(args.load_dir, "encoder.pt")
            proj_p = os.path.join(args.load_dir, "projector.pt")
            if os.path.exists(enc_p):
                model.encoder.load_state_dict(torch.load(enc_p, map_location=now_device()))
            if os.path.exists(proj_p):
                model.projector.load_state_dict(torch.load(proj_p, map_location=now_device()))
            print("[cpt_next] loaded encoder/projector init.")

        params = list(model.parameters())  # unfreeze all
        steps = args.steps
        opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)
        data = list(load_jsonl(args.train_json))
        if len(data) == 0:
            print("[cpt_next] no data.")
            return

        # Pre-compute max chunks for curriculum learning
        # Sample a few examples to estimate max chunks
        sample_max_chunks = 1
        for ex in random.sample(data, min(10, len(data))):
            text = ex["tokens"]
            s = ex.get("split", {}).get("s", 2048)
            chunk_strs, _ = model._chunk_text(text[:s*4], k_tokens=cfg.chunk_len_tokens)  # Approximate
            sample_max_chunks = max(sample_max_chunks, len(chunk_strs))

        # Build curriculum schedule if enabled
        if args.use_curriculum:
            curriculum = curriculum_schedule(steps, sample_max_chunks)
            print(f"[cpt_next] curriculum learning enabled: 1 → {sample_max_chunks} chunks over {steps} steps")
        else:
            curriculum = None

        model.train()
        running_loss = 0.0
        for step in range(steps):
            ex = random.choice(data)
            text = ex["tokens"]
            s_config = ex.get("split", {}).get("s", 2048)
            o_config = ex.get("split", {}).get("o", 256)

            # Apply curriculum: limit number of context chunks used
            if curriculum is not None:
                # Limit context length based on curriculum
                # Each chunk is ~k tokens, so limit s to curriculum[step] * k tokens
                s = min(s_config, curriculum[step] * cfg.chunk_len_tokens)
            else:
                s = s_config

            loss = model.loss_next_para(text, s=s, o=o_config, k=cfg.chunk_len_tokens, expand_frac=args.expand_frac)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sch.step()

            running_loss += loss.item()

            if step % max(1, args.log_every) == 0:
                avg_loss = running_loss / max(1, args.log_every)
                curriculum_cap = curriculum[step] if curriculum else "N/A"
                print(f"[cpt_next] step {step}/{steps} loss={loss.item():.4f} curriculum_chunks={curriculum_cap}")

                if tracker:
                    metrics = {
                        "cpt_loss": loss.item(),
                        "avg_loss": avg_loss,
                        "learning_rate": sch.get_last_lr()[0]
                    }
                    if curriculum:
                        metrics["curriculum_chunks"] = curriculum[step]
                    tracker.log_metrics(metrics, step=step)

                running_loss = 0.0

        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "refrag_full.pt"))
        print(f"[cpt_next] saved full model to {args.out_dir}")

        if tracker:
            tracker.log_model(args.out_dir, "cpt_model")

    finally:
        if tracker:
            tracker.end_run()


def cmd_train_policy(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        lr=args.lr,
        fp16=False,
        policy_hidden=args.policy_hidden,
    )

    # Setup MLflow tracking if enabled
    tracker = None
    if args.use_mlflow:
        tracker = MLflowTracker(
            tracking_uri=args.mlflow_uri,
            experiment_name=args.experiment,
            run_name=f"policy_{args.run_name}"
        )
        tracker.start_run(params={
            "encoder": args.enc,
            "decoder": args.dec,
            "chunk_len_k": args.k,
            "lr": args.lr,
            "steps": args.steps,
            "max_expand_frac": args.p,
            "topk": args.topk,
            "policy_hidden": args.policy_hidden,
            "phase": "policy_training"
        })

    try:
        model = REFRAG(cfg).to(now_device())
        # Optional warm-start
        if args.load_dir:
            try:
                model.encoder.load_state_dict(torch.load(os.path.join(args.load_dir, "encoder.pt"), map_location=now_device()))
                model.projector.load_state_dict(torch.load(os.path.join(args.load_dir, "projector.pt"), map_location=now_device()))
                print("[train_policy] loaded encoder/projector init.")
            except Exception:
                pass

        # Train policy only
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.projector.parameters():
            p.requires_grad = False
        params = list(model.policy.parameters())
        steps = args.steps
        opt, sch = setup_optim(params, lr=cfg.lr, wd=cfg.wd, total_steps=steps)

        texts, index = load_index_bundle(args.index_dir)
        qenc = PassageEncoder(args.embed_model)

        data = list(load_jsonl(args.rag_json))
        if len(data) == 0:
            print("[train_policy] no data.")
            return

        baseline = None
        beta = 0.9  # EMA
        running_reward = 0.0

        model.train()
        for step in range(steps):
            ex = random.choice(data)
            q = ex["question"]
            qv = qenc.encode_query(q)
            _, I = search_index(index, qv, args.topk)
            passages = [texts[i] for i in I]

            log_prob, reward = model.policy_step(q, passages, k=cfg.chunk_len_tokens, max_expand_frac=args.p)
            r = reward.item()
            baseline = r if baseline is None else (beta*baseline + (1-beta)*r)
            advantage = r - baseline

            loss = -(log_prob * advantage)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); sch.step()

            running_reward += r

            if step % max(1, args.log_every) == 0:
                avg_reward = running_reward / max(1, args.log_every)
                print(f"[train_policy] step {step}/{steps} reward={r:.4f} baseline={baseline:.4f} advantage={advantage:.4f}")

                if tracker:
                    tracker.log_metrics({
                        "reward": r,
                        "avg_reward": avg_reward,
                        "baseline": baseline,
                        "advantage": advantage,
                        "learning_rate": sch.get_last_lr()[0]
                    }, step=step)

                running_reward = 0.0

        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.policy.state_dict(), os.path.join(args.out_dir, "policy.pt"))
        print(f"[train_policy] saved policy to {args.out_dir}")

        if tracker:
            tracker.log_model(args.out_dir, "policy_model")

    finally:
        if tracker:
            tracker.end_run()


def cmd_generate(args):
    seed_everything()
    cfg = REFRAGConfig(
        encoder_name=args.enc,
        decoder_name=args.dec,
        chunk_len_tokens=args.k,
        max_q_tokens=256,
        max_ctx_tokens=args.ctx_max,
        max_out_tokens=args.max_new,
        selective_p=args.p,
        fp16=False
    )
    model = REFRAG(cfg)
    # Optional load
    if args.load_dir:
        enc_p = os.path.join(args.load_dir, "encoder.pt")
        proj_p = os.path.join(args.load_dir, "projector.pt")
        pol_p = os.path.join(args.load_dir, "policy.pt")
        full_p = os.path.join(args.load_dir, "refrag_full.pt")
        if os.path.exists(full_p):
            model.load_state_dict(torch.load(full_p, map_location=now_device()), strict=False)
            print("[generate] loaded full model weights.")
        else:
            if os.path.exists(enc_p):
                model.encoder.load_state_dict(torch.load(enc_p, map_location=now_device()))
            if os.path.exists(proj_p):
                model.projector.load_state_dict(torch.load(proj_p, map_location=now_device()))
            if os.path.exists(pol_p):
                model.policy.load_state_dict(torch.load(pol_p, map_location=now_device()))
            print("[generate] loaded available component weights.")

    texts, index = load_index_bundle(args.index_dir)
    qenc = PassageEncoder(args.embed_model)
    qv = qenc.encode_query(args.question)
    _, I = search_index(index, qv, args.topk)
    passages = [texts[i] for i in I]

    out = model.generate(
        question=args.question,
        passages=passages,
        k=args.k,
        p=args.p,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        use_policy=(not args.heuristic),
    )
    print(json.dumps({"question": args.question, "passages": passages, **out}, indent=2))


# ----------------------------
# Argparse
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="REFRAG-style RAG (compress → sense/select → expand)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # index
    sp = sub.add_parser("index", help="Build FAISS index from corpus")
    sp.add_argument("--corpus", type=str, required=True, help="Text file, one passage per line")
    sp.add_argument("--index_dir", type=str, required=True, help="Output directory for index + texts.npy")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.set_defaults(func=cmd_index)

    # cpt_recon
    sp = sub.add_parser("cpt_recon", help="Continual pretraining phase A: reconstruction curriculum")
    sp.add_argument("--train_json", type=str, required=True, help="JSONL with {'tokens':..., 'split':{}}")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64, help="Chunk length in decoder tokens")
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--out_dir", type=str, default="runs/cpt_recon")
    # MLflow arguments
    sp.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    sp.add_argument("--mlflow-uri", type=str, default="mlruns", help="MLflow tracking URI")
    sp.add_argument("--experiment", type=str, default="REFRAG", help="MLflow experiment name")
    sp.add_argument("--run-name", type=str, default="run", help="MLflow run name")
    sp.set_defaults(func=cmd_cpt_recon)

    # cpt_next
    sp = sub.add_parser("cpt_next", help="Continual pretraining phase B: next-paragraph prediction")
    sp.add_argument("--train_json", type=str, required=True, help="JSONL with {'tokens':..., 'split':{'s','o'}}")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--expand_frac", type=float, default=0.25, help="Uniform expansion fraction during CPT-B")
    sp.add_argument("--use-curriculum", action="store_true", help="Enable curriculum learning (CRITICAL for training)")
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with encoder.pt/projector.pt")
    sp.add_argument("--out_dir", type=str, default="runs/cpt_next")
    # MLflow arguments
    sp.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    sp.add_argument("--mlflow-uri", type=str, default="mlruns", help="MLflow tracking URI")
    sp.add_argument("--experiment", type=str, default="REFRAG", help="MLflow experiment name")
    sp.add_argument("--run-name", type=str, default="run", help="MLflow run name")
    sp.set_defaults(func=cmd_cpt_next)

    # train_policy
    sp = sub.add_parser("train_policy", help="Train selective expansion policy with REINFORCE")
    sp.add_argument("--rag_json", type=str, required=True, help="JSONL with {'question':..., 'answers':...} (answers optional)")
    sp.add_argument("--index_dir", type=str, required=True, help="Directory containing texts.npy + faiss.index")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--k", type=int, default=64)
    sp.add_argument("--steps", type=int, default=1000)
    sp.add_argument("--lr", type=float, default=1e-4)
    sp.add_argument("--p", type=float, default=0.25, help="Max expansion fraction per example")
    sp.add_argument("--topk", type=int, default=8, help="#passages retrieved per query")
    sp.add_argument("--policy_hidden", type=int, default=256)
    sp.add_argument("--log_every", type=int, default=50)
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with encoder.pt/projector.pt")
    sp.add_argument("--out_dir", type=str, default="runs/policy")
    # MLflow arguments
    sp.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    sp.add_argument("--mlflow-uri", type=str, default="mlruns", help="MLflow tracking URI")
    sp.add_argument("--experiment", type=str, default="REFRAG", help="MLflow experiment name")
    sp.add_argument("--run-name", type=str, default="run", help="MLflow run name")
    sp.set_defaults(func=cmd_train_policy)

    # generate
    sp = sub.add_parser("generate", help="RAG generate with compression/expansion")
    sp.add_argument("--index_dir", type=str, required=True, help="Directory containing texts.npy + faiss.index")
    sp.add_argument("--embed_model", type=str, default="BAAI/bge-small-en-v1.5")
    sp.add_argument("--enc", type=str, default="roberta-base")
    sp.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    sp.add_argument("--question", type=str, required=True)
    sp.add_argument("--topk", type=int, default=8)
    sp.add_argument("--k", type=int, default=64, help="Chunk length in tokens")
    sp.add_argument("--p", type=float, default=0.25, help="Max expansion fraction")
    sp.add_argument("--ctx_max", type=int, default=2048)
    sp.add_argument("--max_new", type=int, default=256)
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--top_p", type=float, default=1.0)
    sp.add_argument("--heuristic", action="store_true", help="Use heuristic expansion instead of policy")
    sp.add_argument("--load_dir", type=str, default="", help="Optional: dir with saved weights (encoder/projector/policy or refrag_full.pt)")
    sp.set_defaults(func=cmd_generate)

    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
