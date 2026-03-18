#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
OUT_DIR="$PROJECT_ROOT/lib/rag_common/generated"

echo "==> Generating Python gRPC code from proto files..."

# Clean and recreate output directory
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Generate using grpcio-tools (no buf dependency required)
python -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    --pyi_out="$OUT_DIR" \
    "$PROTO_DIR"/common/*.proto \
    "$PROTO_DIR"/services/*.proto

# Create __init__.py files for generated packages
touch "$OUT_DIR/__init__.py"
touch "$OUT_DIR/common/__init__.py" 2>/dev/null || true
touch "$OUT_DIR/services/__init__.py" 2>/dev/null || true

# Fix relative imports in generated files (grpcio-tools generates absolute imports)
if [[ "$OSTYPE" == "darwin"* ]]; then
    find "$OUT_DIR" -name "*_pb2_grpc.py" -exec sed -i '' \
        's/^from common/from rag_common.generated.common/g; s/^from services/from rag_common.generated.services/g' {} +
    find "$OUT_DIR" -name "*_pb2.py" -exec sed -i '' \
        's/^from common/from rag_common.generated.common/g; s/^from services/from rag_common.generated.services/g' {} +
else
    find "$OUT_DIR" -name "*_pb2_grpc.py" -exec sed -i \
        's/^from common/from rag_common.generated.common/g; s/^from services/from rag_common.generated.services/g' {} +
    find "$OUT_DIR" -name "*_pb2.py" -exec sed -i \
        's/^from common/from rag_common.generated.common/g; s/^from services/from rag_common.generated.services/g' {} +
fi

echo "==> Proto generation complete. Output: $OUT_DIR"
