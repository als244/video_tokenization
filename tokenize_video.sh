#!/bin/bash

# This command ensures that the script will exit immediately if any command fails.
set -e

# --- Script Arguments ---
# We'll check if all required arguments are provided.
if [ "$#" -ne 8 ]; then
    echo "Usage: $0 <orig_video_dir> <orig_video_file> <video_nickname> <is_discrete> <temporal_factor> <spatial_factor> <output_tokens_dir> <reconstructed_dir>"
    exit 1
fi

ORIG_VIDEO_DIR=$1
ORIG_VIDEO_FILE=$2
ORIG_VIDEO_FILEPATH="${ORIG_VIDEO_DIR}/${ORIG_VIDEO_FILE}"
VIDEO_NICKNAME=$3
# either 0 or 1
IS_DISCRETE=$4
# either 4 or 8
TEMPORAL_COMPRESSION_FACTOR=$5
# either 8 or 16
SPATIAL_COMPRESSION_FACTOR=$6
OUTPUT_TOKENS_DIR=$7
RECONSTRUCTED_DIR=$8

# --- Main Processing Steps ---

# 'set -x' will print each command to the terminal before it is executed.
set -x

echo "--- [Step 1/3] Starting Visual Tokenization for '${VIDEO_NICKNAME}'... ---"
# Run this command inside the 'visual' directory, using 'python3'
(cd visual && python3 visual_tokenizer.py \
    "../${ORIG_VIDEO_FILEPATH}" \
    "${IS_DISCRETE}" \
    "${TEMPORAL_COMPRESSION_FACTOR}" \
    "${SPATIAL_COMPRESSION_FACTOR}" \
    "../${OUTPUT_TOKENS_DIR}/${VIDEO_NICKNAME}_visual" \
    "../${RECONSTRUCTED_DIR}/${VIDEO_NICKNAME}_reconstructed_visual.mp4")
echo "--- [Step 1/3] Visual Tokenization Complete. ---"
echo "" # Add a blank line for better readability

echo "--- [Step 2/3] Starting Audio Tokenization for '${VIDEO_NICKNAME}'... ---"
# Run this command inside the 'visual' directory, using 'python3'
(cd audio && python3 audio_tokenizer.py \
    "../${ORIG_VIDEO_FILEPATH}" \
    "../${OUTPUT_TOKENS_DIR}/${VIDEO_NICKNAME}_visual.npy" \
    "${TEMPORAL_COMPRESSION_FACTOR}" \
    "../${OUTPUT_TOKENS_DIR}/${VIDEO_NICKNAME}_audio" \
    "../${RECONSTRUCTED_DIR}/${VIDEO_NICKNAME}_reconstructed_audio.wav")
echo "--- [Step 2/3] Audio Tokenization Complete. ---"
echo ""

echo "--- [Step 3/3] Merging Audio and Visual for Final Reconstruction... ---"
# Use 'python3' for this step as well for consistency
python3 audio_visual_merge.py \
    "${RECONSTRUCTED_DIR}/${VIDEO_NICKNAME}_reconstructed_visual.mp4" \
    "${RECONSTRUCTED_DIR}/${VIDEO_NICKNAME}_reconstructed_audio.wav" \
    "${RECONSTRUCTED_DIR}/${VIDEO_NICKNAME}_reconstructed.mp4"
echo "--- [Step 3/3] Reconstruction Complete. ---"
echo ""

# 'set +x' will turn off the command printing.
set +x

echo "✅ All steps for '${VIDEO_NICKNAME}' completed successfully!"
