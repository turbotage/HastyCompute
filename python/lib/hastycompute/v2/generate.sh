#!/bin/sh
# Regenerate the gRPC stubs from rustlib/crates/hasty-server/proto/hasty.proto (needs grpcio-tools).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-../../../../rustlib/.venv/bin/python}
$PY -I -m grpc_tools.protoc -I ../../../../rustlib/crates/hasty-server/proto --python_out=. --grpc_python_out=. hasty.proto
sed -i 's/^import hasty_pb2 as hasty__pb2/from . import hasty_pb2 as hasty__pb2/' hasty_pb2_grpc.py
