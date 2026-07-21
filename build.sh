#!/usr/bin/env bash
# Runs during deployment (Render's "Build Command") to regenerate the
# database and trained model from source files, since .db and .pkl files
# are generated artifacts and not committed to git.
set -e
pip install -r requirements.txt
python3 etl/load.py
python3 rag/retrieve.py
python3 ml/train_model.py
echo "Build complete: database, RAG index, and model all regenerated."
