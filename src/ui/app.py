#!/usr/bin/env python3
"""
REFRAG Experiment Manager - Streamlit UI

A unified interface for:
- Launching RAG, REFRAG, and REFRAG v2 experiments
- Viewing MLflow experiment results
- Comparing runs across different configurations

Usage:
    streamlit run src/ui/app.py

Author: REFRAG Project
"""

import os
import sys
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import queue

import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# MLflow imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Project paths (app.py is in src/ui/, so parent.parent.parent = project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Default MLflow tracking URI
MLFLOW_TRACKING_URI = str(MLRUNS_DIR)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="REFRAG Experiment Manager",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Session State Initialization
# ============================================================================

if "running_processes" not in st.session_state:
    st.session_state.running_processes = {}

if "log_queues" not in st.session_state:
    st.session_state.log_queues = {}

if "experiment_history" not in st.session_state:
    st.session_state.experiment_history = []


# ============================================================================
# Experiment Configurations
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for standard RAG experiments."""
    index_dir: str = "runs/rag_index"
    test_json: str = "data/rag_train.jsonl"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    decoder: str = "meta-llama/Llama-3.2-3B"
    topk: int = 4
    ctx_max: int = 2048
    max_new: int = 128
    max_samples: Optional[int] = None
    output: str = "runs/rag_eval_results.json"


@dataclass
class REFRAGConfig:
    """Configuration for REFRAG v1 experiments."""
    # Model settings
    encoder: str = "roberta-base"
    decoder: str = "meta-llama/Llama-3.2-3B"
    chunk_k: int = 64

    # Training settings
    lr: float = 2e-5
    steps: int = 1000
    expand_frac: float = 0.25
    log_every: int = 50

    # Paths
    train_json: str = "data/cpt_train.jsonl"
    index_dir: str = "runs/index"
    load_dir: str = ""
    out_dir: str = "runs/refrag"


@dataclass
class REFRAGv2Config:
    """Configuration for REFRAG v2 experiments."""
    # Model settings
    encoder: str = "roberta-large"
    decoder: str = "meta-llama/Llama-2-7b-hf"
    chunk_k: int = 16

    # Training settings
    lr_recon: float = 2e-4
    lr_cpt: float = 5e-5
    lr_policy: float = 1e-4
    batch_size: int = 8
    stages: int = 9

    # Paths
    data_dir: str = "data"
    index_dir: str = "runs/index"
    load_dir: str = ""
    out_dir: str = "runs/refrag_v2"


# ============================================================================
# Process Management
# ============================================================================

def run_command_async(cmd: List[str], run_name: str, log_queue: queue.Queue):
    """Run a command asynchronously and capture output."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )

        st.session_state.running_processes[run_name] = {
            "process": process,
            "start_time": datetime.now(),
            "cmd": " ".join(cmd)
        }

        for line in iter(process.stdout.readline, ''):
            if line:
                log_queue.put(line.strip())

        process.wait()
        log_queue.put(f"[COMPLETED] Exit code: {process.returncode}")

    except Exception as e:
        log_queue.put(f"[ERROR] {str(e)}")
    finally:
        if run_name in st.session_state.running_processes:
            del st.session_state.running_processes[run_name]


def launch_experiment(cmd: List[str], run_name: str):
    """Launch an experiment in a background thread."""
    log_queue = queue.Queue()
    st.session_state.log_queues[run_name] = log_queue

    thread = threading.Thread(
        target=run_command_async,
        args=(cmd, run_name, log_queue),
        daemon=True
    )
    thread.start()

    # Add to history
    st.session_state.experiment_history.append({
        "name": run_name,
        "cmd": " ".join(cmd),
        "start_time": datetime.now().isoformat(),
        "status": "running"
    })

    return thread


# ============================================================================
# MLflow Integration
# ============================================================================

def get_mlflow_client():
    """Get MLflow client instance."""
    if not MLFLOW_AVAILABLE:
        return None
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(MLFLOW_TRACKING_URI)


def get_experiments():
    """Get list of MLflow experiments."""
    client = get_mlflow_client()
    if client is None:
        return []
    try:
        return client.search_experiments()
    except Exception:
        return []


def get_runs(experiment_id: str, max_results: int = 50):
    """Get runs for an experiment."""
    client = get_mlflow_client()
    if client is None:
        return []
    try:
        return client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
    except Exception:
        return []


def get_mlflow_ui_url(port: int = 5000):
    """Get MLflow UI URL."""
    return f"http://localhost:{port}"


def get_run_url(run_id: str, port: int = 5000):
    """Get URL for a specific run."""
    return f"http://localhost:{port}/#/experiments/0/runs/{run_id}"


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("🔬 REFRAG Manager")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🚀 Launch Experiment", "📊 MLflow Results", "📜 Run History", "⚙️ Settings"]
    )

    st.sidebar.markdown("---")

    # MLflow status
    st.sidebar.subheader("MLflow Status")
    if MLFLOW_AVAILABLE:
        st.sidebar.success("✅ MLflow Available")
        mlflow_url = get_mlflow_ui_url()
        st.sidebar.markdown(f"[Open MLflow UI]({mlflow_url})")
    else:
        st.sidebar.error("❌ MLflow Not Installed")
        st.sidebar.code("pip install mlflow")

    # Running processes
    if st.session_state.running_processes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Running")
        for name, info in st.session_state.running_processes.items():
            elapsed = (datetime.now() - info["start_time"]).seconds
            st.sidebar.info(f"{name} ({elapsed}s)")

    return page


def render_dashboard():
    """Render main dashboard."""
    st.title("🔬 REFRAG Experiment Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Running Experiments", len(st.session_state.running_processes))

    with col2:
        experiments = get_experiments()
        st.metric("MLflow Experiments", len(experiments))

    with col3:
        st.metric("Total Runs", len(st.session_state.experiment_history))

    st.markdown("---")

    # Quick actions
    st.subheader("Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 New RAG Eval", use_container_width=True):
            st.session_state.selected_page = "launch"
            st.session_state.selected_experiment = "rag"
            st.rerun()

    with col2:
        if st.button("🔧 New REFRAG Train", use_container_width=True):
            st.session_state.selected_page = "launch"
            st.session_state.selected_experiment = "refrag"
            st.rerun()

    with col3:
        if st.button("📊 View MLflow", use_container_width=True):
            mlflow_url = get_mlflow_ui_url()
            st.markdown(f'<a href="{mlflow_url}" target="_blank">Open MLflow UI</a>', unsafe_allow_html=True)

    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    # Recent experiments
    st.subheader("Recent Experiments")

    experiments = get_experiments()
    if experiments:
        for exp in experiments[:5]:
            runs = get_runs(exp.experiment_id, max_results=3)
            with st.expander(f"📁 {exp.name} ({len(runs)} recent runs)"):
                for run in runs:
                    status_icon = "✅" if run.info.status == "FINISHED" else "🔄"
                    st.markdown(f"{status_icon} **{run.info.run_name or run.info.run_id[:8]}**")
                    if run.data.metrics:
                        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(run.data.metrics.items())[:3]])
                        st.caption(metrics_str)
    else:
        st.info("No experiments found. Launch your first experiment!")


def render_launch_experiment():
    """Render experiment launch page."""
    st.title("🚀 Launch Experiment")

    experiment_type = st.selectbox(
        "Experiment Type",
        ["RAG Evaluation", "REFRAG v1 - Reconstruction", "REFRAG v1 - CPT", "REFRAG v1 - Policy",
         "REFRAG v2 - Reconstruction", "REFRAG v2 - CPT", "REFRAG v2 - Policy", "REFRAG v2 - Evaluate"]
    )

    st.markdown("---")

    if experiment_type == "RAG Evaluation":
        render_rag_eval_form()
    elif experiment_type == "REFRAG v1 - Reconstruction":
        render_refrag_recon_form()
    elif experiment_type == "REFRAG v1 - CPT":
        render_refrag_cpt_form()
    elif experiment_type == "REFRAG v1 - Policy":
        render_refrag_policy_form()
    elif experiment_type == "REFRAG v2 - Reconstruction":
        render_refrag_v2_recon_form()
    elif experiment_type == "REFRAG v2 - CPT":
        render_refrag_v2_cpt_form()
    elif experiment_type == "REFRAG v2 - Policy":
        render_refrag_v2_policy_form()
    elif experiment_type == "REFRAG v2 - Evaluate":
        render_refrag_v2_eval_form()


def render_rag_eval_form():
    """Render RAG evaluation form."""
    st.subheader("RAG Evaluation Configuration")

    col1, col2 = st.columns(2)

    with col1:
        index_dir = st.text_input("Index Directory", value="runs/rag_index")
        test_json = st.text_input("Test Data (JSONL)", value="data/rag_train.jsonl")
        embed_model = st.text_input("Embedding Model", value="BAAI/bge-small-en-v1.5")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-3.2-3B")

    with col2:
        topk = st.number_input("Top-K Passages", value=4, min_value=1, max_value=20)
        ctx_max = st.number_input("Max Context Tokens", value=2048, min_value=256, max_value=8192)
        max_new = st.number_input("Max New Tokens", value=128, min_value=16, max_value=512)
        max_samples = st.number_input("Max Samples (0=all)", value=0, min_value=0)

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True)
        experiment_name = st.text_input("Experiment Name", value="RAG_Baseline")
    with col2:
        run_name = st.text_input("Run Name", value=f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        output_file = st.text_input("Output File", value="runs/rag_eval_results.json")

    st.markdown("---")

    if st.button("🚀 Launch RAG Evaluation", type="primary", use_container_width=True):
        cmd = [
            "python", "src/rag.py", "evaluate",
            "--index_dir", index_dir,
            "--test_json", test_json,
            "--embed_model", embed_model,
            "--dec", decoder,
            "--topk", str(topk),
            "--ctx_max", str(ctx_max),
            "--max_new", str(max_new),
            "--output", output_file,
        ]

        if max_samples > 0:
            cmd.extend(["--max_samples", str(max_samples)])

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_recon_form():
    """Render REFRAG v1 reconstruction training form."""
    st.subheader("REFRAG v1 - Reconstruction Training")

    col1, col2 = st.columns(2)

    with col1:
        train_json = st.text_input("Training Data (JSONL)", value="data/cpt_train.jsonl")
        encoder = st.text_input("Encoder Model", value="roberta-base")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-3.2-3B")
        out_dir = st.text_input("Output Directory", value="runs/cpt_recon")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=64, min_value=8, max_value=128)
        lr = st.number_input("Learning Rate", value=2e-5, format="%.1e")
        steps = st.number_input("Training Steps", value=1000, min_value=100, max_value=100000)
        log_every = st.number_input("Log Every N Steps", value=50, min_value=1)

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="recon_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG", key="recon_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="recon_run")

    st.markdown("---")

    if st.button("🚀 Launch Reconstruction Training", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag.py", "cpt_recon",
            "--train_json", train_json,
            "--enc", encoder,
            "--dec", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--steps", str(steps),
            "--log_every", str(log_every),
            "--out_dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_cpt_form():
    """Render REFRAG v1 CPT training form."""
    st.subheader("REFRAG v1 - CPT (Next-Paragraph) Training")

    col1, col2 = st.columns(2)

    with col1:
        train_json = st.text_input("Training Data (JSONL)", value="data/cpt_train.jsonl", key="cpt_train")
        encoder = st.text_input("Encoder Model", value="roberta-base", key="cpt_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-3.2-3B", key="cpt_dec")
        load_dir = st.text_input("Load From (Reconstruction)", value="runs/cpt_recon")
        out_dir = st.text_input("Output Directory", value="runs/cpt_next", key="cpt_out")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=64, min_value=8, max_value=128, key="cpt_k")
        lr = st.number_input("Learning Rate", value=2e-5, format="%.1e", key="cpt_lr")
        steps = st.number_input("Training Steps", value=1000, min_value=100, max_value=100000, key="cpt_steps")
        expand_frac = st.slider("Expansion Fraction", 0.0, 1.0, 0.25)
        log_every = st.number_input("Log Every N Steps", value=50, min_value=1, key="cpt_log")

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="cpt_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG", key="cpt_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"cpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="cpt_run")

    st.markdown("---")

    if st.button("🚀 Launch CPT Training", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag.py", "cpt_next",
            "--train_json", train_json,
            "--enc", encoder,
            "--dec", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--steps", str(steps),
            "--expand_frac", str(expand_frac),
            "--log_every", str(log_every),
            "--load_dir", load_dir,
            "--out_dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_policy_form():
    """Render REFRAG v1 policy training form."""
    st.subheader("REFRAG v1 - Policy Training (REINFORCE)")

    col1, col2 = st.columns(2)

    with col1:
        rag_json = st.text_input("RAG Data (JSONL)", value="data/rag_train.jsonl")
        index_dir = st.text_input("Index Directory", value="runs/index")
        encoder = st.text_input("Encoder Model", value="roberta-base", key="pol_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-3.2-3B", key="pol_dec")
        load_dir = st.text_input("Load From", value="runs/cpt_recon", key="pol_load")
        out_dir = st.text_input("Output Directory", value="runs/policy")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=64, min_value=8, max_value=128, key="pol_k")
        lr = st.number_input("Learning Rate", value=1e-4, format="%.1e", key="pol_lr")
        steps = st.number_input("Training Steps", value=1000, min_value=100, max_value=100000, key="pol_steps")
        topk = st.number_input("Top-K Passages", value=8, min_value=1, max_value=20, key="pol_topk")
        p = st.slider("Max Expansion Fraction", 0.0, 1.0, 0.25, key="pol_p")
        policy_hidden = st.number_input("Policy Hidden Dim", value=256, min_value=64, max_value=1024)

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="pol_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG", key="pol_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="pol_run")

    st.markdown("---")

    if st.button("🚀 Launch Policy Training", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag.py", "train_policy",
            "--rag_json", rag_json,
            "--index_dir", index_dir,
            "--enc", encoder,
            "--dec", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--steps", str(steps),
            "--topk", str(topk),
            "--p", str(p),
            "--policy_hidden", str(policy_hidden),
            "--load_dir", load_dir,
            "--out_dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_v2_recon_form():
    """Render REFRAG v2 reconstruction training form."""
    st.subheader("REFRAG v2 - Reconstruction Training (Paper-Compliant)")

    col1, col2 = st.columns(2)

    with col1:
        data_dir = st.text_input("Data Directory", value="data", key="v2r_data")
        encoder = st.text_input("Encoder Model", value="roberta-large", key="v2r_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-2-7b-hf", key="v2r_dec")
        out_dir = st.text_input("Output Directory", value="runs/refrag_v2_recon", key="v2r_out")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=16, min_value=8, max_value=64, key="v2r_k")
        lr = st.number_input("Learning Rate", value=2e-4, format="%.1e", key="v2r_lr")
        batch_size = st.number_input("Batch Size", value=8, min_value=1, max_value=64, key="v2r_bs")
        stages = st.number_input("Curriculum Stages", value=9, min_value=1, max_value=9, key="v2r_stages")

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="v2r_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG_v2", key="v2r_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"v2_recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="v2r_run")

    st.markdown("---")

    if st.button("🚀 Launch v2 Reconstruction", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag_v2.py", "train_reconstruction",
            "--data-dir", data_dir,
            "--encoder", encoder,
            "--decoder", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--stages", str(stages),
            "--out-dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_v2_cpt_form():
    """Render REFRAG v2 CPT training form."""
    st.subheader("REFRAG v2 - CPT Training (Paper-Compliant)")

    col1, col2 = st.columns(2)

    with col1:
        data_dir = st.text_input("Data Directory", value="data", key="v2c_data")
        encoder = st.text_input("Encoder Model", value="roberta-large", key="v2c_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-2-7b-hf", key="v2c_dec")
        load_dir = st.text_input("Load From (Reconstruction)", value="runs/refrag_v2_recon", key="v2c_load")
        out_dir = st.text_input("Output Directory", value="runs/refrag_v2_cpt", key="v2c_out")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=16, min_value=8, max_value=64, key="v2c_k")
        lr = st.number_input("Learning Rate", value=5e-5, format="%.1e", key="v2c_lr")
        batch_size = st.number_input("Batch Size", value=8, min_value=1, max_value=64, key="v2c_bs")
        stages = st.number_input("Curriculum Stages", value=9, min_value=1, max_value=9, key="v2c_stages")

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="v2c_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG_v2", key="v2c_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"v2_cpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="v2c_run")

    st.markdown("---")

    if st.button("🚀 Launch v2 CPT", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag_v2.py", "train_cpt",
            "--data-dir", data_dir,
            "--encoder", encoder,
            "--decoder", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--batch-size", str(batch_size),
            "--stages", str(stages),
            "--load-dir", load_dir,
            "--out-dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_v2_policy_form():
    """Render REFRAG v2 policy training form."""
    st.subheader("REFRAG v2 - Policy Training (GRPO)")

    col1, col2 = st.columns(2)

    with col1:
        data_dir = st.text_input("Data Directory", value="data", key="v2p_data")
        index_dir = st.text_input("Index Directory", value="runs/index", key="v2p_idx")
        encoder = st.text_input("Encoder Model", value="roberta-large", key="v2p_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-2-7b-hf", key="v2p_dec")
        load_dir = st.text_input("Load From (CPT)", value="runs/refrag_v2_cpt", key="v2p_load")
        out_dir = st.text_input("Output Directory", value="runs/refrag_v2_policy", key="v2p_out")

    with col2:
        chunk_k = st.number_input("Chunk Size (k)", value=16, min_value=8, max_value=64, key="v2p_k")
        lr = st.number_input("Learning Rate", value=1e-4, format="%.1e", key="v2p_lr")
        steps = st.number_input("Training Steps", value=1000, min_value=100, max_value=10000, key="v2p_steps")
        group_size = st.number_input("GRPO Group Size", value=4, min_value=2, max_value=16, key="v2p_grp")

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="v2p_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG_v2", key="v2p_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"v2_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="v2p_run")

    st.markdown("---")

    if st.button("🚀 Launch v2 Policy Training", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag_v2.py", "train_policy",
            "--data-dir", data_dir,
            "--index-dir", index_dir,
            "--encoder", encoder,
            "--decoder", decoder,
            "--k", str(chunk_k),
            "--lr", str(lr),
            "--steps", str(steps),
            "--group-size", str(group_size),
            "--load-dir", load_dir,
            "--out-dir", out_dir,
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_refrag_v2_eval_form():
    """Render REFRAG v2 evaluation form."""
    st.subheader("REFRAG v2 - Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        eval_file = st.text_input("Evaluation Data (JSONL)", value="data/eval.jsonl", key="v2e_eval")
        index_dir = st.text_input("Index Directory", value="runs/index", key="v2e_idx")
        load_dir = st.text_input("Load Model From", value="runs/refrag_v2_cpt", key="v2e_load")
        output = st.text_input("Output File", value="runs/refrag_v2_eval_results.json", key="v2e_out")

    with col2:
        encoder = st.text_input("Encoder Model", value="roberta-large", key="v2e_enc")
        decoder = st.text_input("Decoder Model", value="meta-llama/Llama-2-7b-hf", key="v2e_dec")
        chunk_k = st.number_input("Chunk Size (k)", value=16, min_value=8, max_value=64, key="v2e_k")
        topk = st.number_input("Top-K Passages", value=8, min_value=1, max_value=20, key="v2e_topk")

    st.markdown("---")

    # MLflow settings
    st.subheader("MLflow Tracking")
    col1, col2 = st.columns(2)
    with col1:
        use_mlflow = st.checkbox("Enable MLflow Tracking", value=True, key="v2e_mlflow")
        experiment_name = st.text_input("Experiment Name", value="REFRAG_v2", key="v2e_exp")
    with col2:
        run_name = st.text_input("Run Name", value=f"v2_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="v2e_run")

    st.markdown("---")

    if st.button("🚀 Launch v2 Evaluation", type="primary", use_container_width=True):
        cmd = [
            "python", "src/refrag_v2.py", "evaluate",
            "--eval-file", eval_file,
            "--index-dir", index_dir,
            "--load-dir", load_dir,
            "--output", output,
            "--encoder", encoder,
            "--decoder", decoder,
            "--k", str(chunk_k),
            "--topk", str(topk),
        ]

        if use_mlflow:
            cmd.extend([
                "--use-mlflow",
                "--experiment", experiment_name,
                "--run-name", run_name
            ])

        launch_experiment(cmd, run_name)
        st.success(f"✅ Launched: {run_name}")
        st.code(" ".join(cmd))


def render_mlflow_results():
    """Render MLflow results page."""
    st.title("📊 MLflow Results")

    if not MLFLOW_AVAILABLE:
        st.error("MLflow is not installed. Install with: `pip install mlflow`")
        return

    # MLflow UI link
    mlflow_url = get_mlflow_ui_url()
    st.markdown(f"### [🔗 Open Full MLflow UI]({mlflow_url})")

    st.markdown("---")

    # Experiment selector
    experiments = get_experiments()

    if not experiments:
        st.info("No experiments found. Launch your first experiment!")
        return

    exp_names = [exp.name for exp in experiments]
    selected_exp_name = st.selectbox("Select Experiment", exp_names)

    selected_exp = next((e for e in experiments if e.name == selected_exp_name), None)

    if selected_exp:
        runs = get_runs(selected_exp.experiment_id)

        st.subheader(f"Runs in '{selected_exp_name}'")

        if not runs:
            st.info("No runs found in this experiment.")
            return

        # Runs table
        run_data = []
        for run in runs:
            run_url = get_run_url(run.info.run_id)
            run_data.append({
                "Run Name": run.info.run_name or run.info.run_id[:8],
                "Status": run.info.status,
                "Start Time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                "Duration (s)": (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else "Running",
                "Link": run_url
            })

        # Display as dataframe with links
        st.dataframe(
            run_data,
            column_config={
                "Link": st.column_config.LinkColumn("View Run")
            },
            use_container_width=True
        )

        st.markdown("---")

        # Detailed run view
        st.subheader("Run Details")
        run_names = [r.info.run_name or r.info.run_id[:8] for r in runs]
        selected_run_name = st.selectbox("Select Run", run_names)

        selected_run = next((r for r in runs if (r.info.run_name or r.info.run_id[:8]) == selected_run_name), None)

        if selected_run:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Parameters**")
                if selected_run.data.params:
                    for k, v in selected_run.data.params.items():
                        st.text(f"{k}: {v}")
                else:
                    st.text("No parameters logged")

            with col2:
                st.markdown("**Metrics**")
                if selected_run.data.metrics:
                    for k, v in selected_run.data.metrics.items():
                        st.metric(k, f"{v:.4f}")
                else:
                    st.text("No metrics logged")

            # Link to MLflow
            run_url = get_run_url(selected_run.info.run_id)
            st.markdown(f"[🔗 View Full Run Details in MLflow]({run_url})")


def render_run_history():
    """Render run history page."""
    st.title("📜 Run History")

    # Live logs for running processes
    if st.session_state.running_processes:
        st.subheader("🔄 Currently Running")

        for run_name, info in st.session_state.running_processes.items():
            with st.expander(f"📋 {run_name}", expanded=True):
                st.text(f"Command: {info['cmd']}")
                st.text(f"Started: {info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

                # Show logs
                if run_name in st.session_state.log_queues:
                    log_container = st.empty()
                    logs = []
                    log_queue = st.session_state.log_queues[run_name]

                    while not log_queue.empty():
                        try:
                            logs.append(log_queue.get_nowait())
                        except queue.Empty:
                            break

                    if logs:
                        log_container.code("\n".join(logs[-50:]))  # Last 50 lines

    st.markdown("---")

    # History
    st.subheader("📜 Past Runs")

    if st.session_state.experiment_history:
        for i, run in enumerate(reversed(st.session_state.experiment_history[-20:])):  # Last 20
            status_icon = "✅" if run.get("status") == "completed" else "🔄"
            with st.expander(f"{status_icon} {run['name']} - {run['start_time'][:16]}"):
                st.code(run["cmd"])
    else:
        st.info("No experiment history yet. Launch your first experiment!")

    # Refresh button
    if st.button("🔄 Refresh"):
        st.rerun()


def render_settings():
    """Render settings page."""
    st.title("⚙️ Settings")

    st.subheader("MLflow Configuration")

    mlflow_uri = st.text_input("MLflow Tracking URI", value=MLFLOW_TRACKING_URI)
    mlflow_port = st.number_input("MLflow UI Port", value=5000, min_value=1000, max_value=65535)

    st.markdown("---")

    st.subheader("Start MLflow UI Server")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Start MLflow Server", use_container_width=True):
            cmd = ["mlflow", "ui", "--backend-store-uri", mlflow_uri, "--port", str(mlflow_port)]
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            st.success(f"MLflow UI started at http://localhost:{mlflow_port}")

    with col2:
        mlflow_url = f"http://localhost:{mlflow_port}"
        st.markdown(f"[🔗 Open MLflow UI]({mlflow_url})")

    st.markdown("---")

    st.subheader("Project Paths")
    st.text(f"Project Root: {PROJECT_ROOT}")
    st.text(f"Source Directory: {SRC_DIR}")
    st.text(f"Data Directory: {DATA_DIR}")
    st.text(f"Runs Directory: {RUNS_DIR}")
    st.text(f"MLflow Directory: {MLRUNS_DIR}")

    st.markdown("---")

    st.subheader("Clear History")
    if st.button("🗑️ Clear Experiment History", type="secondary"):
        st.session_state.experiment_history = []
        st.success("History cleared!")
        st.rerun()


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🚀 Launch Experiment":
        render_launch_experiment()
    elif page == "📊 MLflow Results":
        render_mlflow_results()
    elif page == "📜 Run History":
        render_run_history()
    elif page == "⚙️ Settings":
        render_settings()


if __name__ == "__main__":
    main()
