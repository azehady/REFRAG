# REFRAG: Rethinking RAG-based Decoding

An implementation of the REFRAG paper from Meta Superintelligence Labs, featuring context compression for faster retrieval-augmented generation.

[![Paper](https://img.shields.io/badge/arXiv-2509.01092-b31b1b.svg)](https://arxiv.org/abs/2509.01092)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

**REFRAG** (REpresentation For RAG) compresses retrieved context by replacing token sequences with chunk embeddings, achieving **30× faster time-to-first-token** while maintaining accuracy.

### Key Innovation

Standard RAG processes all retrieved tokens through the decoder's attention mechanism. REFRAG exploits the observation that retrieved passages have **block-diagonal attention patterns** (passages don't attend to each other), enabling aggressive compression:

```
Standard RAG:  [Query] + [Passage1 tokens] + [Passage2 tokens] + ... → Decoder
REFRAG:        [Query] + [P1 embedding] + [P2 embedding] + ...      → Decoder
                         ↑ k tokens → 1 embedding (k× compression)
```

### Performance (from paper)

| Metric | Standard RAG | REFRAG (k=16) | Improvement |
|--------|--------------|---------------|-------------|
| TTFT (Time-to-First-Token) | 1,285 ms | 41.6 ms | **30.8×** |
| Throughput | 80 tok/s | 244 tok/s | **3×** |
| Memory | 24.5 GB | 8.6 GB | **2.8×** |
| Accuracy | Baseline | -0.3% | Minimal loss |

---

## Project Structure

```
REFRAG/
├── src/
│   ├── rag.py           # Standard RAG baseline (Qdrant + LLaMA)
│   ├── refrag.py        # REFRAG v1 implementation
│   ├── refrag_v2.py     # REFRAG v2 (paper-compliant)
│   └── ui/
│       └── app.py       # Streamlit experiment manager
├── scripts/
│   └── run_ui.sh        # Launch UI script
├── docs/
│   ├── REFRAG_KEY_INSIGHTS.md        # Paper insights
│   ├── IMPLEMENTATION_COMPARISON.md  # Detailed comparison
│   └── ...
├── data/                # Training/evaluation data
├── runs/                # Model checkpoints & indices
├── mlruns/              # MLflow experiment logs
└── justfile             # Command runner
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/REFRAG.git
cd REFRAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install torch transformers accelerate
pip install faiss-cpu qdrant-client sentence-transformers
pip install mlflow streamlit
```

### Using Just (Recommended)

```bash
# Install just: https://github.com/casey/just
brew install just  # macOS
# or: cargo install just

# See all available commands
just --list

# Run Streamlit UI
just ui

# Run with MLflow server
just ui-full
```

### Manual Commands

```bash
# 1) Build index
python src/refrag_v2.py index --corpus data/corpus.txt --index-dir runs/index

# 2) Train Stage 1: Reconstruction
python src/refrag_v2.py train_reconstruction \
    --data-dir data/ \
    --out-dir runs/stage1 \
    --use-mlflow

# 3) Train Stage 2: CPT
python src/refrag_v2.py train_cpt \
    --data-dir data/ \
    --load-dir runs/stage1 \
    --out-dir runs/stage2 \
    --use-mlflow

# 4) Generate
python src/refrag_v2.py generate \
    --load-dir runs/stage2 \
    --index-dir runs/index \
    --question "What is the capital of France?"

# 5) Evaluate
python src/refrag_v2.py evaluate \
    --load-dir runs/stage2 \
    --index-dir runs/index \
    --eval-file data/eval.jsonl \
    --use-mlflow
```

---

## Implementations

This repository contains three implementations:

### 1. Standard RAG (`src/rag.py`)
Baseline retrieval-augmented generation using Qdrant and LLaMA.

```bash
# Build index
python src/rag.py index --corpus data/corpus.txt --index_dir runs/rag_index

# Evaluate
python src/rag.py evaluate --index_dir runs/rag_index --test_json data/eval.jsonl
```

### 2. REFRAG v1 (`src/refrag.py`)
Initial REFRAG implementation with basic compression.

### 3. REFRAG v2 (`src/refrag_v2.py`) ⭐ Recommended
Paper-compliant implementation with:
- ✅ 9-stage curriculum learning
- ✅ Correct autoregressive reconstruction
- ✅ GRPO-style policy training
- ✅ Paper hyperparameters (2e-4/5e-5/2e-5)
- ✅ MLflow experiment tracking

See [Implementation Comparison](docs/IMPLEMENTATION_COMPARISON.md) for detailed differences.

---

## Compression Rate (k) Selection

The chunk size `k` controls the speed/quality tradeoff:

| k | Compression | TTFT Speedup | Accuracy Drop | Use Case |
|---|-------------|--------------|---------------|----------|
| 8 | 8× | 12.4× | -0.2% | Highest quality |
| **16** | **16×** | **30.8×** | **-0.3%** | **Recommended** |
| 32 | 32× | 48.2× | -1.9% | Speed priority |
| 64 | 64× | 62.1× | -8.5% | Not recommended |

```bash
# Use k=16 (default, recommended)
python src/refrag_v2.py train_reconstruction --data-dir data/ --out-dir runs/stage1

# Use k=8 for highest quality
python src/refrag_v2.py train_reconstruction --data-dir data/ --out-dir runs/stage1 --k 8

# Use k=32 for more speed
python src/refrag_v2.py train_reconstruction --data-dir data/ --out-dir runs/stage1 --k 32
```

---

## Training Pipeline

REFRAG requires a specific training order:

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Reconstruction (freeze decoder)                       │
│  - Train encoder + projector to compress k tokens → 1 embedding │
│  - LR: 2e-4, 9-stage curriculum                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: CPT - Continual Pre-Training (unfreeze all)           │
│  - Train decoder to use compressed embeddings                   │
│  - LR: 5e-5, 9-stage curriculum                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Policy Training (optional)                            │
│  - Train selective expansion policy                             │
│  - GRPO algorithm, reward = -perplexity                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Fine-tuning (task-specific)                           │
│  - Adapt to downstream tasks (QA, summarization, etc.)          │
│  - LR: 2e-5                                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Critical**: Curriculum learning is essential. Without it, training fails (see [paper ablations](docs/IMPLEMENTATION_COMPARISON.md#paper-experimental-results)).

---

## Streamlit UI

Launch the experiment manager UI:

```bash
# Using just
just ui-full  # Starts both Streamlit and MLflow UI

# Manual
streamlit run src/ui/app.py
mlflow ui --backend-store-uri mlruns  # In separate terminal
```

**Features:**
- Launch experiments with configurable parameters
- View MLflow results with direct links
- Compare runs across configurations
- Live log streaming

![UI Screenshot](docs/assets/ui_screenshot.png)

---

## MLflow Tracking

All training commands support MLflow:

```bash
# Enable MLflow tracking
python src/refrag_v2.py train_reconstruction \
    --data-dir data/ \
    --out-dir runs/stage1 \
    --use-mlflow \
    --experiment REFRAG_v2 \
    --run-name recon_k16_v1

# View results
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## Data Format

### Training Data (`data/cpt_train.jsonl`)
```jsonl
{"text": "Long document text for reconstruction and CPT training..."}
{"text": "Another document..."}
```

### Evaluation Data (`data/eval.jsonl`)
```jsonl
{"question": "What is X?", "answers": ["Answer1", "Answer2"]}
{"question": "Who invented Y?", "answers": ["Person Name"]}
```

### Corpus (`data/corpus.txt`)
```text
First passage text (one per line)
Second passage text
...
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [REFRAG Key Insights](docs/REFRAG_KEY_INSIGHTS.md) | Paper summary and key findings |
| [Implementation Comparison](docs/IMPLEMENTATION_COMPARISON.md) | Detailed RAG vs REFRAG v1 vs v2 comparison |
| [UI Guide](src/ui/README.md) | Streamlit UI documentation |
| [Contributing](docs/CONTRIBUTING.md) | Contribution guidelines |

---

## Justfile Commands

```bash
just --list              # Show all commands

# UI
just ui                  # Run Streamlit UI
just ui-full             # Run UI + MLflow server
just mlflow-ui           # Run MLflow UI only

# Index Building
just index-rag           # Build Qdrant index for RAG
just index-v2            # Build FAISS index for REFRAG v2

# REFRAG v2 Training
just v2-recon            # Stage 1: Reconstruction
just v2-cpt              # Stage 2: CPT
just v2-policy           # Stage 3: Policy
just v2-eval             # Evaluate
just v2-full-train       # Run full pipeline

# With MLflow
just v2-recon-mlflow     # Reconstruction with tracking
just v2-cpt-mlflow       # CPT with tracking

# Utilities
just clean-runs          # Clean runs directory
just clean-mlflow        # Clean MLflow data
```

---

## Citation

```bibtex
@article{lin2025refrag,
  title={REFRAG: Rethinking RAG based Decoding},
  author={Lin, Xiaoqiang and Ghosh, Aritra and Low, Bryan Kian Hsiang and Shrivastava, Anshumali and Mohan, Vijai},
  journal={arXiv preprint arXiv:2509.01092},
  year={2025}
}
```

---

## References

- **Paper**: [REFRAG: Rethinking RAG based Decoding](https://arxiv.org/abs/2509.01092) (arXiv:2509.01092v2)
- **Authors**: Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan
- **Affiliation**: Meta Superintelligence Labs, NUS, Rice University
- **Video Explanation**: [Weaviate - REFRAG Explained](https://www.youtube.com/watch?v=Ek0tZootK00)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Meta Superintelligence Labs for the REFRAG paper
- Weaviate for the excellent video explanation
- Hugging Face for transformers library
