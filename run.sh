#!/usr/bin/env bash

set -e

echo "Starting Speech-to-Text Service..."

uvicorn app.main:app --host 0.0.0.0 --port 8000
