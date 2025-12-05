# REFRAG Project - Command Runner
#
# Usage: just <command>
# Run `just --list` to see all available commands

# Default recipe - show help
default:
    @just --list

# ============================================================================
# Setup & Installation
# ============================================================================

# Install all dependencies
install:
    pip install torch transformers accelerate
    pip install faiss-cpu qdrant-client sentence-transformers
    pip install mlflow streamlit
    pip install numpy tqdm

# Install development dependencies
install-dev: install
    pip install pytest black isort mypy

# ============================================================================
# UI Commands
# ============================================================================

# Run the Streamlit experiment manager UI
ui:
    streamlit run src/ui/app.py

# Run UI with MLflow server
ui-full:
    ./scripts/run_ui.sh --mlflow

# Start MLflow UI server only
mlflow-ui:
    mlflow ui --backend-store-uri mlruns --port 5000

# ============================================================================
# Index Building
# ============================================================================

# Build FAISS index for REFRAG
index-refrag corpus="data/corpus.txt" index_dir="runs/index":
    python src/refrag.py index --corpus {{corpus}} --index_dir {{index_dir}}

# Build Qdrant index for RAG
index-rag corpus="data/corpus.txt" index_dir="runs/rag_index":
    python src/rag.py index --corpus {{corpus}} --index_dir {{index_dir}}

# Build FAISS index for REFRAG v2
index-v2 corpus="data/corpus.txt" index_dir="runs/index":
    python src/refrag_v2.py index --corpus {{corpus}} --index-dir {{index_dir}}

# ============================================================================
# RAG (Baseline)
# ============================================================================

# Evaluate standard RAG
rag-eval index_dir="runs/rag_index" test_json="data/rag_train.jsonl" topk="4":
    python src/rag.py evaluate \
        --index_dir {{index_dir}} \
        --test_json {{test_json}} \
        --topk {{topk}} \
        --output runs/rag_eval_results.json

# Evaluate RAG with MLflow tracking
rag-eval-mlflow index_dir="runs/rag_index" test_json="data/rag_train.jsonl" topk="4" run_name="rag_eval":
    python src/rag.py evaluate \
        --index_dir {{index_dir}} \
        --test_json {{test_json}} \
        --topk {{topk}} \
        --output runs/rag_eval_results.json \
        --use-mlflow \
        --experiment RAG_Baseline \
        --run-name {{run_name}}

# Generate answer with RAG
rag-generate index_dir="runs/rag_index" question="What is the answer?":
    python src/rag.py generate \
        --index_dir {{index_dir}} \
        --question "{{question}}"

# ============================================================================
# REFRAG v1
# ============================================================================

# Stage 1: Reconstruction training (freeze decoder)
refrag-recon train_json="data/cpt_train.jsonl" out_dir="runs/cpt_recon" steps="1000":
    python src/refrag.py cpt_recon \
        --train_json {{train_json}} \
        --out_dir {{out_dir}} \
        --steps {{steps}}

# Stage 1 with MLflow
refrag-recon-mlflow train_json="data/cpt_train.jsonl" out_dir="runs/cpt_recon" steps="1000" run_name="recon":
    python src/refrag.py cpt_recon \
        --train_json {{train_json}} \
        --out_dir {{out_dir}} \
        --steps {{steps}} \
        --use-mlflow \
        --experiment REFRAG \
        --run-name {{run_name}}

# Stage 2: CPT next-paragraph training (unfreeze decoder)
refrag-cpt train_json="data/cpt_train.jsonl" load_dir="runs/cpt_recon" out_dir="runs/cpt_next" steps="1000":
    python src/refrag.py cpt_next \
        --train_json {{train_json}} \
        --load_dir {{load_dir}} \
        --out_dir {{out_dir}} \
        --steps {{steps}}

# Stage 2 with MLflow
refrag-cpt-mlflow train_json="data/cpt_train.jsonl" load_dir="runs/cpt_recon" out_dir="runs/cpt_next" steps="1000" run_name="cpt":
    python src/refrag.py cpt_next \
        --train_json {{train_json}} \
        --load_dir {{load_dir}} \
        --out_dir {{out_dir}} \
        --steps {{steps}} \
        --use-mlflow \
        --experiment REFRAG \
        --run-name {{run_name}}

# Stage 3: Policy training (REINFORCE)
refrag-policy rag_json="data/rag_train.jsonl" index_dir="runs/index" load_dir="runs/cpt_recon" out_dir="runs/policy" steps="1000":
    python src/refrag.py train_policy \
        --rag_json {{rag_json}} \
        --index_dir {{index_dir}} \
        --load_dir {{load_dir}} \
        --out_dir {{out_dir}} \
        --steps {{steps}}

# Stage 3 with MLflow
refrag-policy-mlflow rag_json="data/rag_train.jsonl" index_dir="runs/index" load_dir="runs/cpt_recon" out_dir="runs/policy" steps="1000" run_name="policy":
    python src/refrag.py train_policy \
        --rag_json {{rag_json}} \
        --index_dir {{index_dir}} \
        --load_dir {{load_dir}} \
        --out_dir {{out_dir}} \
        --steps {{steps}} \
        --use-mlflow \
        --experiment REFRAG \
        --run-name {{run_name}}

# Generate with REFRAG
refrag-generate index_dir="runs/index" question="What is the answer?" load_dir="runs/cpt_next":
    python src/refrag.py generate \
        --index_dir {{index_dir}} \
        --question "{{question}}" \
        --load_dir {{load_dir}}

# ============================================================================
# REFRAG v2 (Paper-Compliant)
# ============================================================================

# Stage 1: Reconstruction training with curriculum
v2-recon data_dir="data" out_dir="runs/refrag_v2_recon":
    python src/refrag_v2.py train_reconstruction \
        --data-dir {{data_dir}} \
        --out-dir {{out_dir}}

# Stage 1 with MLflow
v2-recon-mlflow data_dir="data" out_dir="runs/refrag_v2_recon" run_name="v2_recon":
    python src/refrag_v2.py train_reconstruction \
        --data-dir {{data_dir}} \
        --out-dir {{out_dir}} \
        --use-mlflow \
        --experiment REFRAG_v2 \
        --run-name {{run_name}}

# Stage 2: CPT with curriculum
v2-cpt data_dir="data" load_dir="runs/refrag_v2_recon" out_dir="runs/refrag_v2_cpt":
    python src/refrag_v2.py train_cpt \
        --data-dir {{data_dir}} \
        --load-dir {{load_dir}} \
        --out-dir {{out_dir}}

# Stage 2 with MLflow
v2-cpt-mlflow data_dir="data" load_dir="runs/refrag_v2_recon" out_dir="runs/refrag_v2_cpt" run_name="v2_cpt":
    python src/refrag_v2.py train_cpt \
        --data-dir {{data_dir}} \
        --load-dir {{load_dir}} \
        --out-dir {{out_dir}} \
        --use-mlflow \
        --experiment REFRAG_v2 \
        --run-name {{run_name}}

# Stage 3: Policy training with GRPO
v2-policy data_dir="data" index_dir="runs/index" load_dir="runs/refrag_v2_cpt" out_dir="runs/refrag_v2_policy":
    python src/refrag_v2.py train_policy \
        --data-dir {{data_dir}} \
        --index-dir {{index_dir}} \
        --load-dir {{load_dir}} \
        --out-dir {{out_dir}}

# Stage 3 with MLflow
v2-policy-mlflow data_dir="data" index_dir="runs/index" load_dir="runs/refrag_v2_cpt" out_dir="runs/refrag_v2_policy" run_name="v2_policy":
    python src/refrag_v2.py train_policy \
        --data-dir {{data_dir}} \
        --index-dir {{index_dir}} \
        --load-dir {{load_dir}} \
        --out-dir {{out_dir}} \
        --use-mlflow \
        --experiment REFRAG_v2 \
        --run-name {{run_name}}

# Generate with REFRAG v2
v2-generate index_dir="runs/index" question="What is the answer?" load_dir="runs/refrag_v2_cpt":
    python src/refrag_v2.py generate \
        --index-dir {{index_dir}} \
        --question "{{question}}" \
        --load-dir {{load_dir}}

# Evaluate REFRAG v2
v2-eval eval_file="data/eval.jsonl" index_dir="runs/index" load_dir="runs/refrag_v2_cpt":
    python src/refrag_v2.py evaluate \
        --eval-file {{eval_file}} \
        --index-dir {{index_dir}} \
        --load-dir {{load_dir}} \
        --output runs/refrag_v2_eval_results.json

# Evaluate REFRAG v2 with MLflow
v2-eval-mlflow eval_file="data/eval.jsonl" index_dir="runs/index" load_dir="runs/refrag_v2_cpt" run_name="v2_eval":
    python src/refrag_v2.py evaluate \
        --eval-file {{eval_file}} \
        --index-dir {{index_dir}} \
        --load-dir {{load_dir}} \
        --output runs/refrag_v2_eval_results.json \
        --use-mlflow \
        --experiment REFRAG_v2 \
        --run-name {{run_name}}

# ============================================================================
# Full Pipelines
# ============================================================================

# Run full REFRAG v1 training pipeline
refrag-full-train train_json="data/cpt_train.jsonl" rag_json="data/rag_train.jsonl" index_dir="runs/index":
    @echo "=== Stage 1: Reconstruction ==="
    just refrag-recon-mlflow {{train_json}} runs/cpt_recon 1000 recon_full
    @echo ""
    @echo "=== Stage 2: CPT ==="
    just refrag-cpt-mlflow {{train_json}} runs/cpt_recon runs/cpt_next 1000 cpt_full
    @echo ""
    @echo "=== Stage 3: Policy ==="
    just refrag-policy-mlflow {{rag_json}} {{index_dir}} runs/cpt_recon runs/policy 1000 policy_full
    @echo ""
    @echo "=== Training Complete ==="

# Run full REFRAG v2 training pipeline
v2-full-train data_dir="data" index_dir="runs/index":
    @echo "=== Stage 1: Reconstruction ==="
    just v2-recon-mlflow {{data_dir}} runs/refrag_v2_recon v2_recon_full
    @echo ""
    @echo "=== Stage 2: CPT ==="
    just v2-cpt-mlflow {{data_dir}} runs/refrag_v2_recon runs/refrag_v2_cpt v2_cpt_full
    @echo ""
    @echo "=== Stage 3: Policy ==="
    just v2-policy-mlflow {{data_dir}} {{index_dir}} runs/refrag_v2_cpt runs/refrag_v2_policy v2_policy_full
    @echo ""
    @echo "=== Training Complete ==="

# REFRAG v2 Quickstart - Full pipeline with evaluation
v2-quickstart:
    ./scripts/refrag_v2_quickstart.sh

# REFRAG v2 Quickstart with MLflow
v2-quickstart-mlflow:
    ./scripts/refrag_v2_quickstart.sh --mlflow

# REFRAG v2 Quickstart - Evaluation only (use existing checkpoints)
v2-quickstart-eval:
    ./scripts/refrag_v2_quickstart.sh --eval-only

# REFRAG v2 Quickstart - Skip policy training
v2-quickstart-fast:
    ./scripts/refrag_v2_quickstart.sh --skip-policy

# ============================================================================
# Comparison & Analysis
# ============================================================================

# Full comparison: RAG vs REFRAG v1 vs REFRAG v2
compare test_json="data/rag_eval_test.jsonl" max_samples="20":
    uv run python eval/evaluate_comparison.py \
        --test_json {{test_json}} \
        --max_samples {{max_samples}} \
        --output runs/comparison_results.json

# Compare RAG vs REFRAG v1 only (skip v2)
compare-v1 test_json="data/rag_eval_test.jsonl" max_samples="20":
    uv run python eval/evaluate_comparison.py \
        --test_json {{test_json}} \
        --max_samples {{max_samples}} \
        --skip_refrag_v2 \
        --output runs/comparison_v1_results.json

# Compare with custom model paths
compare-full test_json="data/rag_eval_test.jsonl" rag_index="runs/rag_index" refrag_load="runs/policy_aligned" v2_load="runs/refrag_v2_cpt" max_samples="20":
    uv run python eval/evaluate_comparison.py \
        --test_json {{test_json}} \
        --rag_index {{rag_index}} \
        --refrag_load {{refrag_load}} \
        --refrag_v2_load {{v2_load}} \
        --max_samples {{max_samples}} \
        --output runs/comparison_full_results.json

# ============================================================================
# Utilities
# ============================================================================

# Clean up runs directory
clean-runs:
    rm -rf runs/*
    @echo "Cleaned runs directory"

# Clean MLflow runs
clean-mlflow:
    rm -rf mlruns/*
    @echo "Cleaned MLflow runs"

# Clean everything
clean-all: clean-runs clean-mlflow
    @echo "Cleaned all generated files"

# Show project structure
tree:
    @echo "Project Structure:"
    @find . -type f -name "*.py" | grep -v __pycache__ | sort
    @echo ""
    @echo "Data files:"
    @ls -la data/ 2>/dev/null || echo "No data directory"
    @echo ""
    @echo "Run outputs:"
    @ls -la runs/ 2>/dev/null || echo "No runs directory"

# Format code with black
format:
    black src/

# Check types with mypy
typecheck:
    mypy src/ --ignore-missing-imports

# Run tests
test:
    pytest tests/ -v
