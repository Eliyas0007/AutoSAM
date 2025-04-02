#!/bin/bash

# Define checkpoint URLs
CHECKPOINTS=(
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt"
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt"
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt"
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt"
)

# Directory to store the checkpoints
DEST_DIR="./checkpoints/"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download each checkpoint
for URL in "${CHECKPOINTS[@]}"; do
  FILE_NAME=$(basename "$URL")
  echo "Downloading $FILE_NAME..."
  wget -q --show-progress -P "$DEST_DIR" "$URL"
done

echo "Download complete! Checkpoints saved in $DEST_DIR."
