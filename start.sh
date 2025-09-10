#!/usr/bin/env bash
set -e

gunicorn src.api.app:app \
  --workers ${GUNICORN_WORKERS:-2} \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --timeout 60 \
  --keep-alive 5 &

nginx -g 'daemon off;'
