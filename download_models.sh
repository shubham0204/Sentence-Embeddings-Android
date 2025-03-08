#!/bin/bash

# Script to download ONNX models from HuggingFace
# Improved with error handling, progress tracking, and directory validation

set -e  # Exit immediately if a command exits with non-zero status

# Configuration
ASSETS_DIR="app/src/main/assets"
MODELS=(
  "sentence-transformers/all-MiniLM-L6-v2:all-minilm-l6-v2"
  "BAAI/bge-small-en-v1.5:bge-small-en-v1.5"
  "Snowflake/snowflake-arctic-embed-s:snowflake-arctic-embed-s"
)
FILES=("model.onnx" "tokenizer.json")

# Create assets directory if it doesn't exist
if [ ! -d "$ASSETS_DIR" ]; then
  echo "Creating assets directory at $ASSETS_DIR"
  mkdir -p "$ASSETS_DIR"
fi

# Function to download a file with error handling
download_file() {
  local source_url=$1
  local target_dir=$2
  local filename=$3
  
  echo "Downloading $filename to $target_dir"
  
  # Create target directory if it doesn't exist
  mkdir -p "$target_dir"
  
  # Attempt download with retries
  for attempt in {1..3}; do
    if wget --progress=bar:force:noscroll "$source_url" -O "$target_dir/$filename" --no-check-certificate; then
      echo "✓ Successfully downloaded $filename"
      return 0
    else
      echo "Attempt $attempt failed. Retrying..."
      sleep 2
    fi
  done
  
  echo "❌ Failed to download $filename after 3 attempts"
  return 1
}

# Main download process
echo "=== Starting ONNX model downloads to $ASSETS_DIR ==="
echo "Models to download: ${#MODELS[@]}"

total_downloads=$((${#MODELS[@]} * ${#FILES[@]}))
current=0
failed=0

for model_pair in "${MODELS[@]}"; do
  # Split model info
  IFS=':' read -r hf_path local_dir <<< "$model_pair"
  
  echo ""
  echo "📦 Processing model: $hf_path"
  target_dir="$ASSETS_DIR/$local_dir"
  
  for file in "${FILES[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_downloads] Downloading $file..."
    
    if [ "$file" = "model.onnx" ]; then
      source_url="https://huggingface.co/$hf_path/resolve/main/onnx/$file"
    else
      source_url="https://huggingface.co/$hf_path/resolve/main/$file"
    fi
    
    if ! download_file "$source_url" "$target_dir" "$file"; then
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "=== Download Summary ==="
echo "Total files: $total_downloads"
echo "Successfully downloaded: $((total_downloads - failed))"

if [ $failed -gt 0 ]; then
  echo "Failed downloads: $failed"
  exit 1
else
  echo "✅ All downloads completed successfully"
fi