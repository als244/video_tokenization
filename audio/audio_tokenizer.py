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

def do_audio_tokenizer(INPUT_WAVE_FILE, ORIG_MP4_PATH, VISUAL_TOKENS_PATH, VISUAL_TEMPORAL_COMPRESSION, AUDIO_TOKENIZED_PATH):
    """
    Tokenizes an audio file into discrete codes synchronized with visual latent frames.

    Args:
        INPUT_WAVE_FILE (str): Path to the input WAV audio file.
        ORIG_MP4_PATH (str): Path to the original MP4 video file to get metadata.
        VISUAL_TOKENS_PATH (str): Path to the numpy file containing visual tokens.
        VISUAL_TEMPORAL_COMPRESSION (int): The temporal compression factor of the visual tokenizer.
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

    # --- 2. Load Audio and Video Metadata ---
    try:
        original_signal = AudioSignal(INPUT_WAVE_FILE)
        visual_tokens = np.load(VISUAL_TOKENS_PATH)
    except Exception as e:
        print(f"Error loading input files: {e}", file=sys.stderr)
        return

    print(f"Loaded signal: Duration={original_signal.duration:.2f}s, Sample Rate={original_signal.sample_rate}Hz")

    # --- 3. Calculate Audio Chunk Size based on Visual Frames ---
    cap = cv2.VideoCapture(ORIG_MP4_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {ORIG_MP4_PATH}", file=sys.stderr)
        return
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    visual_latent_frames = visual_tokens.shape[1]

    # Use the same "ideal" calculation as the detokenizer.
    # This ensures the chunking process is symmetrical.
    audio_sample_rate = original_signal.sample_rate
    duration_per_latent = VISUAL_TEMPORAL_COMPRESSION / orig_fps
    samples_per_chunk = round(duration_per_latent * audio_sample_rate)

    print(f"Original video FPS: {orig_fps:.2f}")
    print(f"Number of visual latent frames: {visual_latent_frames}")
    print(f"Audio samples per latent frame (calculated): {samples_per_chunk}")

    # --- 4. Chunk and Preprocess Audio ---
    print("Chunking and preprocessing audio...")
    original_audio_chunks = []
    for i in range(visual_latent_frames):
        # Use the precise integer calculation to avoid floating point drift
        start_sample = i * samples_per_chunk
        end_sample = start_sample + samples_per_chunk
        audio_chunk_data = original_signal.audio_data[:, :, start_sample:end_sample]
        
        # Ensure the chunk is not empty and handles the end of the audio file
        current_len = audio_chunk_data.shape[2]
        if current_len > 0:
            
            if current_len < samples_per_chunk:
                padding_needed = samples_per_chunk - current_len
                # Pad on the right side (end of the time dimension)
                audio_chunk_data = torch.nn.functional.pad(audio_chunk_data, (0, padding_needed))

            chunk_signal = AudioSignal(audio_chunk_data, sample_rate=original_signal.sample_rate)
            original_audio_chunks.append(chunk_signal)

    print(f"Created {len(original_audio_chunks)} audio chunks.")

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
            
        # Preprocess for the model (e.g., padding)
        preprocessed_chunk = model.preprocess(processed_signal.audio_data, processed_signal.sample_rate)
        preprocessed_audio_chunks.append(preprocessed_chunk)

    # --- 5. Encode Chunks into Tokens ---
    print("Encoding audio chunks to tokens...")
    encoded_audio_codes = []
    with torch.no_grad():
        for i, chunk in enumerate(preprocessed_audio_chunks):
            cur_clip = chunk.to(model.device)
            # Encode returns: z (quantized), codes, latents (pre-quantization), ...
            _, codes, _, _, _ = model.encode(cur_clip)
            encoded_audio_codes.append(codes.detach().cpu())
            if (i + 1) % 100 == 0 or i == len(preprocessed_audio_chunks) - 1:
                print(f"  > Encoded chunk {i+1}/{len(preprocessed_audio_chunks)}")

    # --- 6. Combine and Save Tokens ---
    # Stack the list of code tensors into a single tensor.
    # The new dimension represents the sequence of latent frames.
    # Shape becomes: (batch, seq_len, n_quantizers, n_tokens_per_chunk)
    if not encoded_audio_codes:
        print("Error: No audio codes were generated. Aborting.", file=sys.stderr)
        return
        
    audio_tokens = torch.stack(encoded_audio_codes, dim=1)

    audio_tokens_np = audio_tokens.cpu().numpy()
    np.save(AUDIO_TOKENIZED_PATH, audio_tokens_np)
    print(f"Full audio tokens shape: {audio_tokens.shape}")
    print(f"Successfully saved audio tokens at: {AUDIO_TOKENIZED_PATH}!\n")


def do_audio_detokenizer(
    AUDIO_TOKENIZED_PATH: str, 
    AUDIO_RECONSTRUCT_PATH: str, 
    VISUAL_TEMPORAL_COMPRESSION: int,
    TARGET_FPS: float,
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

    # Calculate target length based on FPS, not original files
    # This is the core of the generative-friendly approach.
    output_sample_rate = model.sample_rate  # 44100
    duration_per_latent = VISUAL_TEMPORAL_COMPRESSION / TARGET_FPS
    target_samples_per_chunk = round(duration_per_latent * output_sample_rate)
    
    print(f"Target FPS: {TARGET_FPS}, Compression: {VISUAL_TEMPORAL_COMPRESSION}x")
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

            # Use resampling to time-stretch/compress the chunk to the correct length
            resampled_chunk = F.resample(
                decoded_chunk,
                orig_freq=current_len,
                new_freq=target_samples_per_chunk
            )
            decoded_audio_chunks.append(resampled_chunk.cpu())
            if (i + 1) % 100 == 0 or i == num_latent_frames - 1:
                print(f"  > Decoded chunk {i+1}/{num_latent_frames}")

    if not decoded_audio_chunks:
        print("Error: No audio chunks were decoded.", file=sys.stderr)
        return

    reconstructed_audio_data = torch.cat(decoded_audio_chunks, dim=-1)
    reconstructed_audio = AudioSignal(reconstructed_audio_data, sample_rate=output_sample_rate)
    reconstructed_audio.write(AUDIO_RECONSTRUCT_PATH)
    print(f"Reconstructed audio duration: {reconstructed_audio.duration:.2f}s")
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
    do_audio_tokenizer(
        INPUT_WAVE_FILE=AUDIO_WAV_PATH,
        ORIG_MP4_PATH=ORIG_MP4_PATH,
        VISUAL_TOKENS_PATH=VISUAL_TOKENS_PATH,
        VISUAL_TEMPORAL_COMPRESSION=VISUAL_TEMPORAL_COMPRESSION,
        AUDIO_TOKENIZED_PATH=AUDIO_TOKENIZED_PATH
    )

    original_fps, original_width, original_height = get_video_properties(ORIG_MP4_PATH)

    # 3. If a reconstruction path is provided, run the detokenizer
    if AUDIO_RECONSTRUCT_PATH:
        print(f"\n\nReconstructing audio from tokens at: {AUDIO_RECONSTRUCT_PATH}...\n")
        do_audio_detokenizer(
            AUDIO_TOKENIZED_PATH=AUDIO_TOKENIZED_PATH + ".npy",
            AUDIO_RECONSTRUCT_PATH=AUDIO_RECONSTRUCT_PATH,
            VISUAL_TEMPORAL_COMPRESSION=VISUAL_TEMPORAL_COMPRESSION,
            TARGET_FPS=original_fps
        )
