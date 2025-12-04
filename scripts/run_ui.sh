#!/bin/bash
# Run the REFRAG Experiment Manager UI
#
# Usage:
#   ./scripts/run_ui.sh [--mlflow] [--port PORT]
#
# Options:
#   --mlflow    Also start MLflow UI server
#   --port      Streamlit port (default: 8501)

set -e

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Default values
STREAMLIT_PORT=8501
MLFLOW_PORT=5000
START_MLFLOW=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mlflow)
            START_MLFLOW=true
            shift
            ;;
        --port)
            STREAMLIT_PORT="$2"
            shift 2
            ;;
        --mlflow-port)
            MLFLOW_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mlflow] [--port PORT] [--mlflow-port PORT]"
            echo ""
            echo "Options:"
            echo "  --mlflow        Also start MLflow UI server"
            echo "  --port          Streamlit port (default: 8501)"
            echo "  --mlflow-port   MLflow UI port (default: 5000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=================================="
echo "REFRAG Experiment Manager"
echo "=================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Start MLflow UI if requested
if [ "$START_MLFLOW" = true ]; then
    echo "Starting MLflow UI on port $MLFLOW_PORT..."
    mlflow ui --backend-store-uri mlruns --port "$MLFLOW_PORT" &
    MLFLOW_PID=$!
    echo "MLflow UI started (PID: $MLFLOW_PID)"
    echo "MLflow UI: http://localhost:$MLFLOW_PORT"
    echo ""

    # Cleanup on exit
    trap "echo 'Stopping MLflow UI...'; kill $MLFLOW_PID 2>/dev/null" EXIT
fi

echo "Starting Streamlit UI on port $STREAMLIT_PORT..."
echo "Streamlit UI: http://localhost:$STREAMLIT_PORT"
echo ""

# Run Streamlit
streamlit run src/ui/app.py --server.port "$STREAMLIT_PORT"
