#!/bin/bash
# Start hasty-serve with the ONNX Runtime environment (models in data/hasty/models, example volumes in data/hasty/examples).
#   rustlib/tools/models/serve.sh [hasty-serve args]      then open https://localhost:8443/node_editor/
# In the node editor: IO/NIfTI Load -> Models/Run Model (device cpu while the GPU is busy).
set -e
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$R/tools/models/ort_env.sh"
cd "$R"
cargo build --release -p hasty-server --bin hasty-serve
exec "$R/target/release/hasty-serve" "$@"
