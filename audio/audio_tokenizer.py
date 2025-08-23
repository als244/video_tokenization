import dac
from audiotools import AudioSignal
import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import sys
import cv2
import ffmpeg
import numpy as np

def extract_audio_ffmpeg_python(mp4_filepath, wav_filepath):
    """
    Extracts audio from an MP4 file and saves it as a WAV file
    using the ffmpeg-python wrapper.

    Args:
        mp4_filepath (str): The path to the input MP4 file.
        wav_filepath (str): The path to save the output WAV file.
    """
    try:
        (
            ffmpeg
            .input(mp4_filepath)
            .output(wav_filepath, acodec='pcm_s16le', ar='44100') # Set audio codec and sample rate
            .run(overwrite_output=True, quiet=True)
        )
        print(f"Audio extracted successfully to '{wav_filepath}'")
    except ffmpeg.Error as e:
        print(f"An FFmpeg error occurred: {e.stderr.decode('utf8')}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: The file '{mp4_filepath}' was not found or FFmpeg is not installed/in PATH.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during audio extraction: {e}", file=sys.stderr)

def do_audio_tokenizer(INPUT_WAVE_FILE, VISUAL_TOKENS_PATH, AUDIO_TOKENIZED_PATH):
    """
    Tokenizes an audio file into discrete codes synchronized with visual latent frames.

    Args:
        INPUT_WAVE_FILE (str): Path to the input WAV audio file.
        VISUAL_TOKENS_PATH (str): Path to the numpy file containing visual tokens.
        AUDIO_TOKENIZED_PATH (str): Path to save the output audio tokens.
    """

    print("--- Starting Audio Tokenization ---")
    # --- 1. Setup Model ---
    print("Loading DAC model...")
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    # --- 2. Load Audio and Visual Data ---
    try:
        original_signal = AudioSignal(INPUT_WAVE_FILE)
        visual_tokens = np.load(VISUAL_TOKENS_PATH)
    except Exception as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        return

    print(f"Loaded signal: Duration={original_signal.duration:.2f}s, Sample Rate={original_signal.sample_rate}Hz")

    # --- 3. Calculate Audio Chunk Size using Proportional Slicing ---
    total_audio_samples = original_signal.audio_data.shape[-1]
    num_visual_latent_frames = visual_tokens.shape[1]

    if num_visual_latent_frames == 0:
        print("Error: No visual latent frames found. Cannot process audio.", file=sys.stderr)
        return

    avg_samples_per_latent = total_audio_samples / num_visual_latent_frames
    
    # Define a single, fixed target length for all chunks.
    target_samples_per_chunk = round(avg_samples_per_latent)

    print(f"Number of visual latent frames: {num_visual_latent_frames}")
    print(f"Average audio samples per latent frame: {avg_samples_per_latent:.2f}")
    print(f"All audio chunks will be resampled to a fixed length of {target_samples_per_chunk} samples.")

    # --- 4. Chunk and Preprocess Audio ---
    print("Chunking and preprocessing audio...")
    original_audio_chunks = []
    
    for i in range(num_visual_latent_frames):
        
        
        # Step A: Slice the audio proportionally to get a variable-length chunk.
        start_sample = round(i * avg_samples_per_latent)
        end_sample = round((i + 1) * avg_samples_per_latent)
        end_sample = min(end_sample, total_audio_samples)

        audio_chunk_data = original_signal.audio_data[:, :, start_sample:end_sample]
        current_len = audio_chunk_data.shape[2]
        
        if current_len > 0:
            # Resample the variable-length chunk to the fixed target length.
            if current_len != target_samples_per_chunk:
                # Use resampling to time-stretch or compress the audio chunk.
                # This ensures every chunk has the exact same length before encoding.
                resampled_chunk_data = F.resample(
                    audio_chunk_data,
                    orig_freq=current_len,
                    new_freq=target_samples_per_chunk
                )
            else:
                # No resampling needed if it's already the target length
                resampled_chunk_data = audio_chunk_data

            chunk_signal = AudioSignal(resampled_chunk_data, sample_rate=original_signal.sample_rate)
            original_audio_chunks.append(chunk_signal)
        
        if (i + 1) % 100 == 0 or i == num_visual_latent_frames - 1:
            print(f"   > Preprocessed audio corresponding to latent frame {i+1}/{num_visual_latent_frames}", end="\r")

    print(f"Created and resampled {len(original_audio_chunks)} audio chunks to a fixed size.")

    # --- 5. Preprocess Chunks for DAC Model ---
    # The DAC model expects a specific sample rate (44.1kHz) and mono audio.
    target_sample_rate = 44100
    preprocessed_audio_chunks = []
    resampler = T.Resample(
        orig_freq=original_signal.sample_rate,
        new_freq=target_sample_rate,
    ).to(device)

    for chunk in original_audio_chunks:
        # Convert to mono
        if chunk.num_channels > 1:
            chunk = chunk.to_mono()
        
        # Resample if necessary
        if chunk.sample_rate != target_sample_rate:
            resampled_data = resampler(chunk.audio_data.to(device))
            processed_signal = AudioSignal(resampled_data, sample_rate=target_sample_rate)
        else:
            processed_signal = chunk.to(device)
            
        # The model's preprocess step adds the necessary padding to a fixed size
        preprocessed_chunk = model.preprocess(processed_signal.audio_data, processed_signal.sample_rate)
        preprocessed_audio_chunks.append(preprocessed_chunk)

    # --- 6. Encode Chunks into Tokens ---
    print("Encoding audio chunks to tokens...")
    encoded_audio_codes = []
    with torch.no_grad():
        for i, chunk in enumerate(preprocessed_audio_chunks):
            cur_clip = chunk.to(model.device)
            _, codes, _, _, _ = model.encode(cur_clip)
            encoded_audio_codes.append(codes.detach().cpu())
            if (i + 1) % 100 == 0 or i == len(preprocessed_audio_chunks) - 1:
                print(f"  > Encoded chunk {i+1}/{len(preprocessed_audio_chunks)}", end="\r")

    # --- 7. Combine and Save Tokens ---
    if not encoded_audio_codes:
        print("Error: No audio codes were generated. Aborting.", file=sys.stderr)
        return
        
    audio_tokens = torch.stack(encoded_audio_codes, dim=1)
    audio_tokens_np = audio_tokens.cpu().numpy()
    np.save(AUDIO_TOKENIZED_PATH, audio_tokens_np)
    print(f"Full audio tokens shape: {audio_tokens.shape}")
    print(f"Successfully saved audio tokens at: {AUDIO_TOKENIZED_PATH}!\n")

    return target_samples_per_chunk


def do_audio_detokenizer(
    AUDIO_TOKENIZED_PATH: str, 
    AUDIO_RECONSTRUCT_PATH: str, 
    target_samples_per_chunk: int
):
    """
    Reconstructs an audio waveform from tokens based on a target FPS.
    """
    print("--- Starting Audio Detokenization ---")
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    try:
        audio_tokens_np = np.load(AUDIO_TOKENIZED_PATH)
        audio_tokens = torch.from_numpy(audio_tokens_np).to(device)
    except Exception as e:
        print(f"Error loading token file '{AUDIO_TOKENIZED_PATH}': {e}", file=sys.stderr)
        return


    print(f"Each decoded chunk will be resampled to {target_samples_per_chunk} samples.")

    print("Decoding tokens back to audio...")
    decoded_audio_chunks = []
    num_latent_frames = audio_tokens.shape[1]

    with torch.no_grad():
        for i in range(num_latent_frames):
            codes = audio_tokens[:, i, :, :]
            z = model.quantizer.from_codes(codes)[0]
            decoded_chunk = model.decode(z)
            
            # Resample the decoded chunk to match the target duration
            current_len = decoded_chunk.shape[-1]
            if current_len == 0: continue

            if current_len != target_samples_per_chunk:
                # Use resampling to time-stretch/compress the chunk to the correct length
                resampled_chunk = F.resample(
                    decoded_chunk,
                    orig_freq=current_len,
                    new_freq=target_samples_per_chunk
                )
            decoded_audio_chunks.append(resampled_chunk.cpu())
            if (i + 1) % 100 == 0 or i == num_latent_frames - 1:
                print(f"  > Decoded chunk {i+1}/{num_latent_frames}", end="\r")

    if not decoded_audio_chunks:
        print("Error: No audio chunks were decoded.", file=sys.stderr)
        return

    reconstructed_audio_data = torch.cat(decoded_audio_chunks, dim=-1)

    ### CHANGE: Remove the unnecessary batch dimension for a standard audio format.
    # Shape changes from (1, 1, total_samples) to (1, total_samples)
    final_audio_data = reconstructed_audio_data.squeeze(0)

    reconstructed_audio = AudioSignal(final_audio_data, sample_rate=model.sample_rate)
    reconstructed_audio.write(AUDIO_RECONSTRUCT_PATH)
    
    print(f"\nReconstructed audio duration: {reconstructed_audio.duration:.2f}s")
    print(f"Successfully reconstructed audio at: {AUDIO_RECONSTRUCT_PATH}\n")


def get_video_properties(input_mp4_filepath):
    cap = cv2.VideoCapture(input_mp4_filepath)
    # Handle potential warning message during property reading
    if not cap.isOpened(): print(f"Error: Could not open video file: {input_mp4_filepath}"); exit()
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # Check for valid properties
    if original_fps <= 0 or original_width <= 0 or original_height <= 0:
        print(f"Warning: Invalid video properties read for {input_mp4_filepath}. Using defaults (24fps, 640x480).")
        original_fps = 24
        # You might need to set width/height manually if reading fails completely
        original_width = 640
        original_height = 480
    print(f"Original video properties: {original_width}x{original_height}, {original_fps:.2f} FPS")
    return original_fps, original_width, original_height


if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Usage: python audio_tokenizer.py <orig_mp4_path> <visual_tokens_path> <visual_temporal_compression_factor> <output_tokens_path> [optional_reconstructed_audio_path]")
        sys.exit(1)

    ORIG_MP4_PATH = sys.argv[1]
    VISUAL_TOKENS_PATH = sys.argv[2]
    VISUAL_TEMPORAL_COMPRESSION = int(sys.argv[3])
    AUDIO_TOKENIZED_PATH = sys.argv[4]
    
    # The path for the reconstructed audio is now optional
    AUDIO_RECONSTRUCT_PATH = sys.argv[5] if len(sys.argv) == 6 else None

    # 1. Extract audio from the source MP4 file
    AUDIO_WAV_PATH = ORIG_MP4_PATH.rsplit('.', 1)[0] + "_audio.wav"
    print(f"\n\nExtracting audio from: {ORIG_MP4_PATH}...\n")
    extract_audio_ffmpeg_python(ORIG_MP4_PATH, AUDIO_WAV_PATH)

    # 2. Run the tokenizer to generate discrete audio codes
    print(f"\n\nTokenizing audio from: {AUDIO_WAV_PATH}...\n")
    samples_per_chunk = do_audio_tokenizer(
        INPUT_WAVE_FILE=AUDIO_WAV_PATH,
        VISUAL_TOKENS_PATH=VISUAL_TOKENS_PATH,
        AUDIO_TOKENIZED_PATH=AUDIO_TOKENIZED_PATH
    )

    original_fps, original_width, original_height = get_video_properties(ORIG_MP4_PATH)

    # 3. If a reconstruction path is provided, run the detokenizer
    if AUDIO_RECONSTRUCT_PATH:
        print(f"\n\nReconstructing audio from tokens at: {AUDIO_RECONSTRUCT_PATH}...\n")
        do_audio_detokenizer(
            AUDIO_TOKENIZED_PATH=AUDIO_TOKENIZED_PATH + ".npy",
            AUDIO_RECONSTRUCT_PATH=AUDIO_RECONSTRUCT_PATH,
            target_samples_per_chunk=samples_per_chunk
        )
