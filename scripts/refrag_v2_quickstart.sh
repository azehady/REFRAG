#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# REFRAG v2 Quickstart Script
# =============================================================================
# This script runs the full REFRAG v2 training pipeline:
#   1) Build FAISS index from corpus
#   2) Stage 1: Reconstruction training (freeze decoder)
#   3) Stage 2: CPT training (unfreeze decoder)
#   4) Stage 3: Policy training (optional, GRPO-style)
#   5) Run comparison evaluation (RAG vs REFRAG v1 vs REFRAG v2)
#
# Usage:
#   ./scripts/refrag_v2_quickstart.sh
#   ./scripts/refrag_v2_quickstart.sh --skip-training  # Use existing checkpoints
#   ./scripts/refrag_v2_quickstart.sh --eval-only      # Only run evaluation
#
# Environment variables (customize):
#   ENC_MODEL     - Encoder model (default: roberta-large)
#   DEC_MODEL     - Decoder model (default: meta-llama/Llama-3.2-3B)
#   EMBED_MODEL   - Embedding model for retrieval (default: BAAI/bge-small-en-v1.5)
#   K             - Chunk size for compression (default: 16)
#   STEPS_RECON   - Reconstruction training steps (default: 500)
#   STEPS_CPT     - CPT training steps (default: 500)
#   STEPS_POLICY  - Policy training steps (default: 200)
#   EVAL_SAMPLES  - Number of evaluation samples (default: 20)
# =============================================================================

# ===== Default Settings =====
ENC_MODEL="${ENC_MODEL:-roberta-large}"
DEC_MODEL="${DEC_MODEL:-meta-llama/Llama-3.2-3B}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
K="${K:-16}"
TOPK="${TOPK:-4}"
STEPS_RECON="${STEPS_RECON:-250}"
STEPS_CPT="${STEPS_CPT:-250}"
STEPS_POLICY="${STEPS_POLICY:-200}"
LR_RECON="${LR_RECON:-2e-4}"
LR_CPT="${LR_CPT:-1e-5}"           # Reduced from 5e-5 to prevent gradient explosion
LR_CPT_DEC="${LR_CPT_DEC:-5e-7}"   # Very low LR for decoder to prevent gradient explosion
LR_POLICY="${LR_POLICY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-20}"
CORPUS="${CORPUS:-data/corpus_large.txt}"
# ============================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
SKIP_TRAINING=false
EVAL_ONLY=false
SKIP_INDEX=false
SKIP_POLICY=false
USE_MLFLOW=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --skip-index)
            SKIP_INDEX=true
            shift
            ;;
        --skip-policy)
            SKIP_POLICY=true
            shift
            ;;
        --mlflow)
            USE_MLFLOW=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-training   Skip training, use existing checkpoints"
            echo "  --eval-only       Only run evaluation (skip index and training)"
            echo "  --skip-index      Skip index building"
            echo "  --skip-policy     Skip policy training (Stage 3)"
            echo "  --mlflow          Enable MLflow tracking"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  ENC_MODEL=$ENC_MODEL"
            echo "  DEC_MODEL=$DEC_MODEL"
            echo "  K=$K"
            echo "  STEPS_RECON=$STEPS_RECON"
            echo "  STEPS_CPT=$STEPS_CPT"
            echo "  EVAL_SAMPLES=$EVAL_SAMPLES"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify required files
if [[ ! -f "src/refrag_v2.py" ]]; then
    echo "ERROR: src/refrag_v2.py not found in $PROJECT_DIR"
    exit 1
fi

# Environment settings
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1  # Force unbuffered Python output

echo "=============================================="
echo "REFRAG v2 Quickstart Pipeline"
echo "=============================================="
echo "Configuration:"
echo "  Encoder:     $ENC_MODEL"
echo "  Decoder:     $DEC_MODEL"
echo "  Embed Model: $EMBED_MODEL"
echo "  K (chunk):   $K"
echo "  TopK:        $TOPK"
echo "  Corpus:      $CORPUS"
echo "  Steps:       Recon=$STEPS_RECON, CPT=$STEPS_CPT, Policy=$STEPS_POLICY"
echo "  Batch Size:  $BATCH_SIZE"
echo "  Eval Samples: $EVAL_SAMPLES"
echo "=============================================="
echo ""

# Create directories
mkdir -p runs/refrag_v2_recon runs/refrag_v2_cpt runs/refrag_v2_policy data

# Check for training data
if [[ ! -f "data/cpt_train.jsonl" ]] && [[ "$EVAL_ONLY" == "false" ]] && [[ "$SKIP_TRAINING" == "false" ]]; then
    echo "WARNING: data/cpt_train.jsonl not found. Creating sample training data..."
    cat > data/cpt_train.jsonl << 'EOF'
{"text": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London. The discovery was accidental, when Fleming noticed that a mold had killed bacteria in a petri dish."}
{"text": "The capital of France is Paris, known for the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is also famous for its cuisine, art, and fashion."}
{"text": "Alan Turing proposed the Turing test in 1950. He is considered the father of theoretical computer science and artificial intelligence."}
{"text": "Large language models can use retrieval-augmented generation to access external knowledge. This improves factual accuracy and reduces hallucinations."}
EOF
fi

# =============================================================================
# Stage 0: Build Index
# =============================================================================
if [[ "$EVAL_ONLY" == "false" ]] && [[ "$SKIP_INDEX" == "false" ]]; then
    if [[ -f "$CORPUS" ]]; then
        echo ""
        echo "=============================================="
        echo "Stage 0: Building FAISS Index"
        echo "=============================================="

        if [[ -f "runs/index/faiss.index" ]]; then
            echo "Index already exists at runs/index/faiss.index"
            echo "Skipping index build. Use --skip-index=false to force rebuild."
        else
            uv run python src/refrag_v2.py index \
                --corpus "$CORPUS" \
                --index-dir runs/index \
                --embed-model "$EMBED_MODEL"
            echo "Index built successfully!"
        fi
    else
        echo "WARNING: Corpus file not found: $CORPUS"
        echo "Skipping index build."
    fi
fi

# =============================================================================
# Stage 1: Reconstruction Training
# =============================================================================
if [[ "$EVAL_ONLY" == "false" ]] && [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "Stage 1: Reconstruction Training"
    echo "=============================================="
    echo "Training encoder + projector (decoder frozen)"
    echo "LR: $LR_RECON, Steps: $STEPS_RECON"
    echo ""

    MLFLOW_ARGS=""
    if [[ "$USE_MLFLOW" == "true" ]]; then
        MLFLOW_ARGS="--use-mlflow --experiment REFRAG_v2 --run-name v2_recon"
    fi

    # Run with tee to both display and save logs
    mkdir -p runs/refrag_v2_recon/logs
    uv run python -u src/refrag_v2.py train_reconstruction \
        --data-dir data \
        --out-dir runs/refrag_v2_recon \
        --encoder "$ENC_MODEL" \
        --decoder "$DEC_MODEL" \
        --k "$K" \
        --lr "$LR_RECON" \
        --batch-size "$BATCH_SIZE" \
        $MLFLOW_ARGS 2>&1 | tee runs/refrag_v2_recon/logs/train.log

    echo "Stage 1 complete! Checkpoint saved to runs/refrag_v2_recon"
    echo "Logs saved to runs/refrag_v2_recon/logs/train.log"
fi

# =============================================================================
# Stage 2: Continual Pre-Training (CPT)
# =============================================================================
if [[ "$EVAL_ONLY" == "false" ]] && [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "Stage 2: Continual Pre-Training (CPT)"
    echo "=============================================="
    echo "Training full model (decoder unfrozen)"
    echo "LR: $LR_CPT, Steps: $STEPS_CPT"
    echo ""

    MLFLOW_ARGS=""
    if [[ "$USE_MLFLOW" == "true" ]]; then
        MLFLOW_ARGS="--use-mlflow --experiment REFRAG_v2 --run-name v2_cpt"
    fi

    # Run with tee to both display and save logs
    mkdir -p runs/refrag_v2_cpt/logs
    uv run python -u src/refrag_v2.py train_cpt \
        --data-dir data \
        --load-dir runs/refrag_v2_recon \
        --out-dir runs/refrag_v2_cpt \
        --encoder "$ENC_MODEL" \
        --decoder "$DEC_MODEL" \
        --k "$K" \
        --lr "$LR_CPT" \
        --lr-decoder "$LR_CPT_DEC" \
        --batch-size "$BATCH_SIZE" \
        $MLFLOW_ARGS 2>&1 | tee runs/refrag_v2_cpt/logs/train.log

    echo "Stage 2 complete! Checkpoint saved to runs/refrag_v2_cpt"
    echo "Logs saved to runs/refrag_v2_cpt/logs/train.log"
fi

# =============================================================================
# Stage 3: Policy Training (Optional)
# =============================================================================
if [[ "$EVAL_ONLY" == "false" ]] && [[ "$SKIP_TRAINING" == "false" ]] && [[ "$SKIP_POLICY" == "false" ]]; then
    if [[ -f "data/rag_train.jsonl" ]] && [[ -f "runs/index/faiss.index" ]]; then
        echo ""
        echo "=============================================="
        echo "Stage 3: Policy Training (GRPO)"
        echo "=============================================="
        echo "Training selective expansion policy"
        echo "LR: $LR_POLICY, Steps: $STEPS_POLICY"
        echo ""

        MLFLOW_ARGS=""
        if [[ "$USE_MLFLOW" == "true" ]]; then
            MLFLOW_ARGS="--use-mlflow --experiment REFRAG_v2 --run-name v2_policy"
        fi

        uv run python src/refrag_v2.py train_policy \
            --data-dir data \
            --index-dir runs/index \
            --load-dir runs/refrag_v2_cpt \
            --out-dir runs/refrag_v2_policy \
            --encoder "$ENC_MODEL" \
            --decoder "$DEC_MODEL" \
            --k "$K" \
            --lr "$LR_POLICY" \
            --steps "$STEPS_POLICY" \
            $MLFLOW_ARGS

        echo "Stage 3 complete! Checkpoint saved to runs/refrag_v2_policy"
    else
        echo ""
        echo "Skipping Stage 3 (Policy Training):"
        echo "  - Missing data/rag_train.jsonl or runs/index/faiss.index"
    fi
fi

# =============================================================================
# Quick Generation Test
# =============================================================================
echo ""
echo "=============================================="
echo "Testing Generation"
echo "=============================================="

# Determine which checkpoint to use
V2_LOAD_DIR="runs/refrag_v2_cpt"
if [[ -f "runs/refrag_v2_policy/policy.pt" ]]; then
    V2_LOAD_DIR="runs/refrag_v2_policy"
fi

if [[ -d "$V2_LOAD_DIR" ]]; then
    echo "Using checkpoint: $V2_LOAD_DIR"
    echo ""

    # Test question
    TEST_Q="What is the capital of France?"
    echo "Q: $TEST_Q"

    uv run python src/refrag_v2.py generate \
        --index-dir runs/index \
        --load-dir "$V2_LOAD_DIR" \
        --encoder "$ENC_MODEL" \
        --decoder "$DEC_MODEL" \
        --k "$K" \
        --topk "$TOPK" \
        --question "$TEST_Q" \
        --max-tokens 64 \
        --temperature 0.0 || echo "Generation test failed (may be expected if no index)"
else
    echo "No trained checkpoint found. Skipping generation test."
fi

# =============================================================================
# Evaluation: Compare RAG vs REFRAG v1 vs REFRAG v2
# =============================================================================
echo ""
echo "=============================================="
echo "Running Comparison Evaluation"
echo "=============================================="

# Check what's available
echo "Checking available models..."

EVAL_ARGS="--test_json data/rag_eval_test.jsonl --max_samples $EVAL_SAMPLES"

# Check test file
if [[ ! -f "data/rag_eval_test.jsonl" ]]; then
    if [[ -f "data/rag_train.jsonl" ]]; then
        echo "Using data/rag_train.jsonl for evaluation"
        EVAL_ARGS="--test_json data/rag_train.jsonl --max_samples $EVAL_SAMPLES"
    else
        echo "ERROR: No evaluation data found (rag_eval_test.jsonl or rag_train.jsonl)"
        echo "Skipping evaluation."
        exit 0
    fi
fi

# Set v2 load directory
EVAL_ARGS="$EVAL_ARGS --refrag_v2_load $V2_LOAD_DIR"
EVAL_ARGS="$EVAL_ARGS --v2_encoder $ENC_MODEL"
EVAL_ARGS="$EVAL_ARGS --v2_decoder $DEC_MODEL"
EVAL_ARGS="$EVAL_ARGS --v2_k $K"

# Run comparison
echo ""
echo "Running: uv run python eval/evaluate_comparison.py $EVAL_ARGS"
echo ""

uv run python eval/evaluate_comparison.py $EVAL_ARGS \
    --output runs/comparison_v2_results.json

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to: runs/comparison_v2_results.json"
echo ""
echo "To view detailed results:"
echo "  cat runs/comparison_v2_results.json | python -m json.tool"
echo ""
echo "To run MLflow UI (if enabled):"
echo "  mlflow ui --backend-store-uri mlruns"
echo ""
