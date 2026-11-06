#!/usr/bin/env bash
# Starts the API in the background and the Streamlit app in the foreground (macOS / Linux).
set -e
[ -d .venv ] && source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap "kill $API_PID" EXIT
sleep 4
python -m streamlit run frontend/app.py
