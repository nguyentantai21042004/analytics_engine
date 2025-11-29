#!/bin/bash
set -e

# PhoBERT Model Download Script
# Downloads model artifacts from MinIO self-hosted server

BUCKET_NAME="phobert-onnx-artifacts"
TARGET_DIR="infrastructure/phobert/models"
REQUIRED_FILES=(
    "model_quantized.onnx"
    "config.json"
    "vocab.txt"
    "special_tokens_map.json"
    "tokenizer_config.json"
)

echo "=== PhoBERT Model Download Script ==="
echo ""

# Check if models already exist
if [ -f "$TARGET_DIR/model_quantized.onnx" ]; then
    echo "✓ Model already exists at $TARGET_DIR"
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Prompt for MinIO credentials if not set
if [ -z "$MINIO_ENDPOINT" ]; then
    read -p "Enter MinIO server IP/hostname (e.g., 192.168.1.100:9000): " MINIO_ENDPOINT
fi

if [ -z "$MINIO_ACCESS_KEY" ]; then
    read -p "Enter MinIO Access Key: " MINIO_ACCESS_KEY
fi

if [ -z "$MINIO_SECRET_KEY" ]; then
    read -sp "Enter MinIO Secret Key: " MINIO_SECRET_KEY
    echo
fi

echo ""
echo "Downloading from: $MINIO_ENDPOINT/$BUCKET_NAME"
echo "Target directory: $TARGET_DIR"
echo ""

# Check if mc (MinIO Client) is installed
if command -v mc &> /dev/null; then
    echo "Using MinIO Client (mc)..."
    
    # Configure mc alias
    ALIAS_NAME="phobert_temp"
    mc alias set "$ALIAS_NAME" "http://$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api S3v4
    
    # Download files
    for file in "${REQUIRED_FILES[@]}"; do
        echo "Downloading $file..."
        mc cp "$ALIAS_NAME/$BUCKET_NAME/$file" "$TARGET_DIR/$file"
    done
    
    # Remove alias
    mc alias remove "$ALIAS_NAME"
    
else
    echo "MinIO Client (mc) not found. Using curl..."
    echo "Note: This requires the bucket to be publicly accessible or proper S3 signature."
    
    # Simple curl download (works if bucket is public)
    for file in "${REQUIRED_FILES[@]}"; do
        echo "Downloading $file..."
        URL="http://$MINIO_ENDPOINT/$BUCKET_NAME/$file"
        
        # Try with curl
        if ! curl -f -o "$TARGET_DIR/$file" "$URL" 2>/dev/null; then
            echo "❌ Failed to download $file"
            echo "   Please ensure:"
            echo "   1. MinIO server is accessible at $MINIO_ENDPOINT"
            echo "   2. Bucket '$BUCKET_NAME' exists and is accessible"
            echo "   3. Install MinIO Client for authenticated downloads: brew install minio/stable/mc"
            exit 1
        fi
    done
fi

echo ""
echo "✅ Download complete!"
echo ""

# Verify files
echo "Verifying downloaded files..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$TARGET_DIR/$file" ]; then
        SIZE=$(du -h "$TARGET_DIR/$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    else
        echo "  ✗ $file (missing)"
        exit 1
    fi
done

echo ""
echo "🎉 All model files downloaded successfully!"
echo "   Location: $TARGET_DIR"
