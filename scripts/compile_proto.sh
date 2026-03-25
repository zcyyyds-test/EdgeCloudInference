#!/bin/bash
# Compile .proto files to Python gRPC stubs.
# Usage: bash scripts/compile_proto.sh

set -e

PROTO_DIR="edgerouter/proto"
OUT_DIR="edgerouter/proto"

python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    --pyi_out="$OUT_DIR" \
    "$PROTO_DIR/edgerouter.proto"

# Fix imports in generated code (grpc_tools generates absolute imports)
sed -i 's/^import edgerouter_pb2/from edgerouter.proto import edgerouter_pb2/' \
    "$OUT_DIR/edgerouter_pb2_grpc.py" 2>/dev/null || true

echo "Proto compilation complete: $OUT_DIR/edgerouter_pb2*.py"
