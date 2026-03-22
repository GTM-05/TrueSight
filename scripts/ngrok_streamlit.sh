#!/usr/bin/env bash
# Expose local TrueSight (Streamlit) on the public internet via ngrok.
#
# One-time:  ngrok config add-authtoken <token>   # from https://dashboard.ngrok.com
#
# Terminal A — start Streamlit (bind all interfaces; relax CORS/XSRF for tunneling):
#   cd /path/to/TrueSight && source .venv/bin/activate
#   streamlit run app.py --server.address 0.0.0.0 --server.port 8501 \
#     --server.enableCORS false --server.enableXsrfProtection false
#
# Terminal B — this script:
#   ./scripts/ngrok_streamlit.sh
#   # or:  PORT=8502 ./scripts/ngrok_streamlit.sh

set -euo pipefail
PORT="${PORT:-8501}"
exec ngrok http "$PORT"
