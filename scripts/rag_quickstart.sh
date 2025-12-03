#!/usr/bin/env bash
# ============================================================================
# RAG Quickstart Script
# Standard Retrieval-Augmented Generation using Qdrant + HuggingFace
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

# ============================================================================
# Configuration (override with environment variables)
# ============================================================================
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
DEC_MODEL="${DEC_MODEL:-meta-llama/Llama-3.2-3B}"
TOPK="${TOPK:-4}"
CTX_MAX="${CTX_MAX:-2048}"
MAX_NEW="${MAX_NEW:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# Directories
INDEX_DIR="${INDEX_DIR:-runs/rag_index}"
CORPUS="${CORPUS:-data/corpus_large.txt}"

echo "=============================================="
echo "RAG Quickstart"
echo "=============================================="
echo "Embedding Model: ${EMBED_MODEL}"
echo "Decoder Model:   ${DEC_MODEL}"
echo "Top-K:           ${TOPK}"
echo "Corpus:          ${CORPUS}"
echo "Index Dir:       ${INDEX_DIR}"
echo "=============================================="

# ============================================================================
# Step 1: Install dependencies (if needed)
# ============================================================================
echo ""
echo "---- Step 1: Checking dependencies ----"
uv pip install --quiet torch transformers qdrant-client sentence-transformers tqdm accelerate 2>/dev/null || true
echo "Dependencies OK"

# ============================================================================
# Step 2: Build Qdrant index (if not exists)
# ============================================================================
echo ""
echo "---- Step 2: Building Qdrant index ----"
if [ -d "${INDEX_DIR}/qdrant_db" ]; then
    echo "Index already exists at ${INDEX_DIR}, skipping..."
else
    echo "Building index from ${CORPUS}..."
    uv run python src/rag.py index \
        --corpus "${CORPUS}" \
        --index_dir "${INDEX_DIR}" \
        --embed_model "${EMBED_MODEL}"
fi

# ============================================================================
# Step 3: Test with sample questions
# ============================================================================
echo ""
echo "---- Step 3: Testing with sample questions ----"

echo ""
echo "[Test 1] Which river flows through City_20?"
uv run python src/rag.py generate \
    --index_dir "${INDEX_DIR}" \
    --embed_model "${EMBED_MODEL}" \
    --dec "${DEC_MODEL}" \
    --question "Which river flows through City_20?" \
    --topk ${TOPK} \
    --ctx_max ${CTX_MAX} \
    --max_new ${MAX_NEW} \
    --temperature ${TEMPERATURE} 2>/dev/null

echo ""
echo "[Test 2] What is the melting point of Alloy_5?"
uv run python src/rag.py generate \
    --index_dir "${INDEX_DIR}" \
    --embed_model "${EMBED_MODEL}" \
    --dec "${DEC_MODEL}" \
    --question "What is the melting point of Alloy_5?" \
    --topk ${TOPK} \
    --ctx_max ${CTX_MAX} \
    --max_new 64 \
    --temperature ${TEMPERATURE} 2>/dev/null

echo ""
echo "[Test 3] In which field does Person_6 work?"
uv run python src/rag.py generate \
    --index_dir "${INDEX_DIR}" \
    --embed_model "${EMBED_MODEL}" \
    --dec "${DEC_MODEL}" \
    --question "In which field does Person_6 work?" \
    --topk ${TOPK} \
    --ctx_max ${CTX_MAX} \
    --max_new 64 \
    --temperature ${TEMPERATURE} 2>/dev/null

# ============================================================================
# Step 4: Run evaluation (optional)
# ============================================================================
echo ""
echo "---- Step 4: Running evaluation on 20 samples ----"

# Generate valid test set if it doesn't exist
if [ ! -f "data/rag_test_valid.jsonl" ]; then
    echo "Generating valid test set from corpus..."
    uv run python -c "
import json
import re

corpus_entities = {}
with open('data/corpus_large.txt', 'r') as f:
    for line in f:
        m = re.search(r'City_(\d+).*?River_(\d+)', line)
        if m:
            corpus_entities[f'city_{m.group(1)}_river'] = {
                'question': f'Which river flows through City_{m.group(1)}?',
                'answers': [f'River_{m.group(2)}']
            }
        m = re.search(r'Alloy_(\d+).*?Melting point: (\d+ °C)', line)
        if m:
            corpus_entities[f'alloy_{m.group(1)}_melting'] = {
                'question': f'What is the melting point of Alloy_{m.group(1)}?',
                'answers': [m.group(2)]
            }
        m = re.search(r'Person_(\d+).*?Field_(\d+)', line)
        if m:
            corpus_entities[f'person_{m.group(1)}_field'] = {
                'question': f'In which field does Person_{m.group(1)} work?',
                'answers': [f'Field_{m.group(2)}']
            }

import random
random.seed(42)
items = list(corpus_entities.values())
random.shuffle(items)
test_items = items[:100]

with open('data/rag_test_valid.jsonl', 'w') as f:
    for i, item in enumerate(test_items):
        item['id'] = f'test_{i}'
        f.write(json.dumps(item) + '\n')
print(f'Generated {len(test_items)} test samples')
"
fi

uv run python src/rag.py evaluate \
    --index_dir "${INDEX_DIR}" \
    --test_json "data/rag_test_valid.jsonl" \
    --embed_model "${EMBED_MODEL}" \
    --dec "${DEC_MODEL}" \
    --topk ${TOPK} \
    --ctx_max ${CTX_MAX} \
    --max_new 64 \
    --max_samples 20 \
    --output "runs/rag_eval_results.json" 2>/dev/null

echo ""
echo "=============================================="
echo "RAG Quickstart Complete!"
echo "=============================================="
echo "Results saved to: runs/rag_eval_results.json"
echo ""
echo "To run a custom query:"
echo "  uv run python src/rag.py generate \\"
echo "    --index_dir ${INDEX_DIR} \\"
echo "    --question \"Your question here\" \\"
echo "    --topk ${TOPK}"
echo "=============================================="
