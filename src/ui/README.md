# REFRAG Experiment Manager UI

A Streamlit-based interface for managing REFRAG experiments with MLflow integration.

## Features

- **Dashboard**: Overview of running experiments and recent MLflow runs
- **Launch Experiments**: Forms to configure and launch:
  - RAG Evaluation
  - REFRAG v1 (Reconstruction, CPT, Policy training)
  - REFRAG v2 (Reconstruction, CPT, Policy, Evaluation)
- **MLflow Results**: Browse experiments and runs with direct links to MLflow UI
- **Run History**: View logs and history of launched experiments
- **Settings**: Configure MLflow and start the MLflow UI server

## Quick Start

```bash
# Install dependencies
pip install streamlit mlflow

# Start the UI (from project root)
cd /path/to/REFRAG
streamlit run ui/app.py

# In a separate terminal, start MLflow UI
mlflow ui --backend-store-uri mlruns
```

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run ui/app.py
   ```
   Opens at http://localhost:8501

2. **Start MLflow UI** (for viewing detailed results):
   ```bash
   mlflow ui --backend-store-uri mlruns
   ```
   Opens at http://localhost:5000

3. **Launch experiments** from the "Launch Experiment" page

4. **View results** in the "MLflow Results" page or click links to open MLflow UI

## Pages

### Dashboard
- Shows running experiments count
- Quick action buttons
- Recent experiments from MLflow

### Launch Experiment
Select experiment type and configure:
- Model paths (encoder, decoder)
- Training parameters (lr, steps, batch size)
- Data paths
- MLflow tracking settings (experiment name, run name)

### MLflow Results
- Browse all MLflow experiments
- View runs with parameters and metrics
- Direct links to MLflow UI for detailed analysis

### Run History
- Live logs from running experiments
- History of past runs with commands

### Settings
- Configure MLflow tracking URI
- Start MLflow UI server
- View project paths

## Architecture

```
ui/
├── app.py           # Main Streamlit application
├── requirements.txt # UI-specific dependencies
└── README.md        # This file
```

The UI launches experiments as background processes and captures their output.
All experiments log to MLflow for persistence and visualization.
