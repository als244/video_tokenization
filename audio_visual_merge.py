import sys
import subprocess

if len(sys.argv) != 4:
    print("Usage: python audio_visual_merge.py <video_file> <audio_file> <output_file>")
    sys.exit(1)

# --- File Paths ---
video_file = sys.argv[1]
audio_file = sys.argv[2]
output_file = sys.argv[3]

# --- FFmpeg Command ---
# This is the same command from before, but formatted for Python
command = [
    'ffmpeg',
    '-i', video_file,
    '-i', audio_file,
    '-c:v', 'copy',      # Copy the video stream without re-encoding
    '-c:a', 'aac',       # Re-encode the audio to AAC
    '-shortest',         # Finish encoding when the shortest input stream ends
    output_file
]

# --- Execute the Command ---
# 'check=True' will raise an error if FFmpeg fails
try:
    subprocess.run(command, check=True)
    print(f"Successfully created '{output_file}'!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")