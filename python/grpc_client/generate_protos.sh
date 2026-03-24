#!/usr/bin/env bash
# Run from this directory.
# Generates hasty_service_pb2.py and hasty_service_pb2_grpc.py into gen/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="$(realpath "$SCRIPT_DIR/../../cpp/server/protos")"
OUT_DIR="$SCRIPT_DIR/gen"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/hasty_service.proto"

# Fix bare imports in generated grpc stubs → relative imports so the gen/
# package works correctly when imported from a parent directory.
sed -i 's/^import \(.*_pb2\) as /from . import \1 as /' "$OUT_DIR"/*_grpc.py

echo "Generated stubs in $OUT_DIR"
