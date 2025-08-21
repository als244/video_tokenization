import cv2
import numpy as np
import torch
import os
import time
import importlib
import cosmos_tokenizer.video_lib
import sys
import math

importlib.reload(cosmos_tokenizer.video_lib)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


# --- Configuration ---

def do_visual_tokenizer(input_mp4_filepath, tokenizer_temporal_window_size, temporal_compression, spatial_compression, output_tokens_filepath):

    TEMPORAL_COMP = temporal_compression
    SPATIAL_COMP = spatial_compression

    model_name = f"Cosmos-Tokenizer-DV{TEMPORAL_COMP}x{SPATIAL_COMP}x{SPATIAL_COMP}"

    ckpt_dir = "pretrained_ckpts"
    encoder_ckpt = os.path.join(ckpt_dir, model_name, "encoder.jit")
    decoder_ckpt = os.path.join(ckpt_dir, model_name, "decoder.jit")

    if not os.path.exists(encoder_ckpt) or not os.path.exists(decoder_ckpt):
        print(f"Error: Checkpoint files not found at {encoder_ckpt} or {decoder_ckpt}")
        exit()

    if not os.path.exists(input_mp4_filepath):
        print(f"Error: Input video file not found at {input_mp4_filepath}")
        exit()

    # --- Video Loading (Frame-by-Frame) ---
    # (Keep the cv2.VideoCapture loop as before to read frames into 'frames' list)
    print(f"Reading video: {input_mp4_filepath} using OpenCV...")
    cap = cv2.VideoCapture(input_mp4_filepath)
    if not cap.isOpened(): print(f"Error: Could not open video file: {input_mp4_filepath}"); exit()
    frames = []
    frame_count = 0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Renamed to avoid confusion
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {width}x{height}, {fps:.2f} FPS, {total_frames_video} frames")

    # Handle case where total frames couldn't be read
    if total_frames_video <= 0:
        print("Warning: Could not determine total frames. Reading until end.")
        # We cannot pre-allocate perfectly, fallback to list append or estimate large size
        # For simplicity, let's exit here, but could implement list append as fallback
        print("Error: Cannot pre-allocate memory without knowing total frames.")
        cap.release()
        exit()

    start_time = time.time()

    # --- Pre-allocate NumPy Array ---
    print(f"Pre-allocating NumPy array for {total_frames_video} frames (uint8)...")
    try:
        # Shape: T, H, W, C (RGB, uint8)
        input_video_thwc = np.zeros((total_frames_video, height, width, 3), dtype=np.uint8)
    except MemoryError:
        print("\nError: Out of memory even when pre-allocating the uint8 array.")
        print(f"Required RAM roughly: {total_frames_video * height * width * 3 / (1024**3):.2f} GB")
        cap.release()
        exit()
    except Exception as e:
        print(f"Error during pre-allocation: {e}")
        cap.release()
        exit()

    # --- Fill Array Frame-by-Frame ---
    print("Reading video and filling array...")
    frame_count = 0
    start_time = time.time()
    while frame_count < total_frames_video: # Read only up to the allocated size
        ret, frame = cap.read() # BGR, uint8 [0, 255]
        if not ret:
            print(f"\nWarning: Video stream ended unexpectedly after {frame_count} frames (expected {total_frames_video}).")
            break # Stop reading if stream ends

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB, uint8 [0, 255]
        frame_rgb = frame_rgb[..., :3] # Ensure 3 channels

        # Assign frame directly into the pre-allocated array
        input_video_thwc[frame_count] = frame_rgb
        frame_count += 1

        if frame_count % 500 == 0 or frame_count == total_frames_video - 1:
            elapsed = time.time() - start_time
            print(f"Read {frame_count}/{total_frames_video} frames... ({elapsed:.2f} seconds)", end='\r')

    cap.release()
    print(f"\nFinished reading {frame_count} actual frames.")

    # Trim array if fewer frames were read than expected
    if frame_count < total_frames_video:
        print(f"Trimming array to actual frames read: {frame_count}")
        input_video_thwc = input_video_thwc[:frame_count]


    # --- Prepare Input Tensor (B C T H W - RGB - uint8) ---
    # (No stacking needed now)
    print("Permuting axes to (B, C, T, H, W - RGB uint8)...")
    try:
        # input_video_thwc is T H W C
        # Permute T H W C -> C T H W, then add B dim
        batched_input_video_bcthw_uint8 = np.expand_dims(input_video_thwc.transpose(3, 0, 1, 2), axis=0)
        print(f"Batched uint8 video shape (BCTHW): {batched_input_video_bcthw_uint8.shape}")
        del input_video_thwc # Free memory
    except Exception as e:
        print(f"Error during permutation/expand_dims: {e}")
        exit()


    # --- Convert Full uint8 NumPy array to uint8 PyTorch Tensor (on CPU first) ---
    # (This part remains the same as the previous fix)
    print("Converting uint8 NumPy array to uint8 PyTorch tensor (on CPU)...")
    try:
        input_tensor_full_uint8_cpu = torch.from_numpy(batched_input_video_bcthw_uint8)
        print(f"Full uint8 tensor shape: {input_tensor_full_uint8_cpu.shape}, device: {input_tensor_full_uint8_cpu.device}")
        del batched_input_video_bcthw_uint8
    except Exception as e: print(f"\nError converting NumPy uint8 array to PyTorch uint8 tensor: {e}"); exit()


    # --- Tokenization Setup ---
    print("Initializing tokenizer...")
    tokenizer_device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer_device == "cuda" and torch.cuda.is_bf16_supported():
        dtype_string = "bfloat16"; tokenizer_dtype = torch.bfloat16
    else:
        dtype_string = "float32"; tokenizer_dtype = torch.float32
    print(f"Using device: {tokenizer_device}, dtype: {dtype_string}")

    tokenizer = CausalVideoTokenizer(
        checkpoint_enc=encoder_ckpt, checkpoint_dec=decoder_ckpt,
        device=tokenizer_device, dtype=dtype_string,
    )

    # --- Process in Chunks (Including Chunk-wise Normalization) ---
    # (Remains the same as the previous fix)
    print(f"Processing in chunks of T={tokenizer_temporal_window_size}, with each chunk being {tokenizer_temporal_window_size // temporal_compression + 1} frames, normalizing chunk-by-chunk...")
    all_token_indices = []

    ## REF: https://github.com/NVIDIA/Cosmos-Tokenizer/blob/main/cosmos_tokenizer/video_lib.py

    num_chunks = math.ceil(total_frames_video / tokenizer_temporal_window_size)
    start_time_encoding = time.time()

    for i in range(num_chunks):
        t_start = i * tokenizer_temporal_window_size
        t_end = min((i + 1) * tokenizer_temporal_window_size, total_frames_video)

        print(f"Processing chunk {i+1}/{num_chunks} (Frames {t_start}-{t_end})...", end='\r')
        input_chunk_uint8 = input_tensor_full_uint8_cpu[:, :, t_start:t_end, :, :]
        input_chunk_tensor = (input_chunk_uint8.to(device=tokenizer_device, dtype=tokenizer_dtype) / 127.5) - 1.0

        try:
            with torch.no_grad():
                indices_chunk, codes_chunk = tokenizer.encode(input_chunk_tensor)
                indices_chunk = indices_chunk[:, :, :, :]
                all_token_indices.append(indices_chunk.cpu())
                del indices_chunk
                del codes_chunk

        except torch.cuda.OutOfMemoryError: print(f"\nError: CUDA out of memory during tokenization of chunk {i+1}."); exit()
        except Exception as e: print(f"\nAn error occurred during tokenization of chunk {i+1}: {e}"); exit()

        del input_chunk_tensor
        torch.cuda.empty_cache()

    del input_tensor_full_uint8_cpu

    encoding_time = time.time() - start_time_encoding
    print(f"Finished encoding all chunks in {encoding_time:.2f} seconds.")

    # --- Concatenate and Save Tokens ---
    if not all_token_indices: print("Error: No tokens were generated."); exit()
    print("Concatenating tokens from all chunks...")
    try:
        final_tokens_tensor = torch.cat(all_token_indices, dim=1) # Assuming dim=1 is correct
        print(f"Final concatenated tokens tensor shape: {final_tokens_tensor.shape}")
    except Exception as e: print(f"Error concatenating token chunks: {e}"); exit()

    tokens_np = final_tokens_tensor.numpy()

    return tokens_np


def do_visual_detokenizer(input_tokens_filepath, original_fps, original_width, original_height, tokenizer_temporal_chunk_size, temporal_compression, spatial_compression, output_mp4_filepath):

    TEMPORAL_COMP = temporal_compression
    SPATIAL_COMP = spatial_compression


    model_name = f"Cosmos-Tokenizer-DV{TEMPORAL_COMP}x{SPATIAL_COMP}x{SPATIAL_COMP}"

    decoder_device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_dtype = torch.bfloat16 if decoder_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    dtype_string = "bfloat16" if decoder_dtype == torch.bfloat16 else "float32"
    print(f"Using device: {decoder_device}, dtype: {dtype_string}")

    ckpt_dir = "pretrained_ckpts"
    decoder_ckpt = os.path.join(ckpt_dir, model_name, "decoder.jit")

    if not os.path.exists(decoder_ckpt):
        print(f"Error: Decoder checkpoint file not found at {decoder_ckpt}")
        exit()

    decoder = CausalVideoTokenizer(
        checkpoint_dec=decoder_ckpt, device=decoder_device, dtype=dtype_string,
    )

    # --- Prepare Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_mp4_filepath, fourcc, original_fps, (original_width, original_height), isColor=True)

    if not video_writer.isOpened():
        print(f"Error: Could not open VideoWriter for path: {output_mp4_filepath}"); exit()
    print(f"Opened VideoWriter. Saving reconstruction to: {output_mp4_filepath}")


    
    print(f"Loading tokens from: {input_tokens_filepath}")
    try:
        tokens_np = np.load(input_tokens_filepath)
        final_tokens_tensor = torch.from_numpy(tokens_np) # Keep on CPU for now
        print(f"Loaded tokens tensor shape: {final_tokens_tensor.shape}")
        if final_tokens_tensor.dim() != 4 or final_tokens_tensor.shape[0] != 1:
            print("Error: Unexpected token tensor shape."); exit()
    except Exception as e: print(f"Error loading tokens: {e}"); exit()

    # --- Decode and Write Video Chunk by Chunk ---
    print(f"Decoding tokens in chunks corresponding to T_tok={tokenizer_temporal_window_size}...")

    all_frames_written = 0
    total_latent_frames = final_tokens_tensor.shape[1]
    latent_frames_per_chunk = tokenizer_temporal_window_size // temporal_compression + 1

    total_latent_chunks = math.ceil(total_latent_frames / latent_frames_per_chunk)

    total_orig_frames = total_latent_frames * temporal_compression + 1

    start_decode_write_time = time.time()

    cur_orig_frame = 0

    try:
        for i in range(total_latent_chunks):
           
            latent_start_frame = i * latent_frames_per_chunk
            latent_end_frame = min((i + 1) * latent_frames_per_chunk, total_latent_frames)

           

            indices_chunk = final_tokens_tensor[:, latent_start_frame:latent_end_frame, :, :].to(decoder_device)

            with torch.no_grad():
                reconstructed_chunk = decoder.decode(indices_chunk) # (B, C, T_vid_chunk, H, W)

            reconstructed_video_chunk = reconstructed_chunk.squeeze(0) # (C, T_vid_chunk, H, W)
            num_frames_in_chunk = reconstructed_video_chunk.shape[1]

            print(f"Processing latent chunk {i+1}/{total_latent_chunks} (Original frames {cur_orig_frame}-{cur_orig_frame + num_frames_in_chunk - 1})...", end='\r')

            for t in range(num_frames_in_chunk):
                frame_tensor = reconstructed_video_chunk[:, t, :, :] # (C, H, W), bfloat16 on GPU
                frame_permuted = frame_tensor.permute(1, 2, 0)     # (H, W, C), bfloat16 on GPU
                
                # >>> Convert dtype to float32 <<<
                frame_float32 = frame_permuted.float() # Converts bfloat16 to float32 on GPU
                
                frame_np_float32 = frame_float32.cpu().numpy() # float32 NumPy array
                
                # Maps assumed [-1, 1] range to [0, 1], then scales to [0, 255]
                frame_np_uint8 = ((frame_np_float32 * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)

                # Convert color space (Assuming RGB -> BGR)
                frame_np_bgr = cv2.cvtColor(frame_np_uint8, cv2.COLOR_RGB2BGR)
                
                # --- ASSUME frame_np_uint8 is ALREADY BGR ---
                frame_to_write = frame_np_bgr

                # Write frame
                video_writer.write(frame_to_write)
                all_frames_written += 1

                del frame_tensor
                del frame_permuted
                del frame_np_float32
                
            
            cur_orig_frame += num_frames_in_chunk

            del indices_chunk
            del reconstructed_chunk
            del reconstructed_video_chunk

            torch.cuda.empty_cache()

        # Clear print progress line
        print()

    except torch.cuda.OutOfMemoryError:
        print(f"\nError: CUDA out of memory during decoding chunk {i+1}.")
    except Exception as e:
        print(f"\nAn error occurred during chunked decoding or video writing (chunk {i+1}): {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging

    finally:
        # --- Release Video Writer ---
        print(f"Releasing VideoWriter... Total frames written: {all_frames_written}")
        video_writer.release()

    total_time = time.time() - start_decode_write_time
    print(f"Total decoding and writing time: {total_time:.2f} seconds.")
    print("\nScript finished.")


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
        print("Error. Usage: visual_tokenizer.py <input_mp4_filepath> <temporal_compression_factor: [4|8]> <spatial_compression_factor: [8|16]> <output_tokens_filepath> [reconstruct_video_output_mp4_filepath]")
        sys.exit(1)

    input_mp4_filepath = sys.argv[1]

    temporal_compression = int(sys.argv[2])

    if (temporal_compression != 4) and (temporal_compression != 8):
        print(f"Error: temporal compression must be set to 4 or 8, {temporal_compression} is not supported...")
        sys.exit(1)

    spatial_compression = int(sys.argv[3])

    if (spatial_compression != 8) and (spatial_compression != 16):
        print(f"Error: spatial compression must be set to 8 or 16, {spatial_compression} is not supported...")
        sys.exit(1)

    output_tokens_filepath = sys.argv[4]

    if len(sys.argv) == 6:
        output_mp4_filepath = sys.argv[5]
    else:
        output_mp4_filepath = None

    ## 49 is default in https://colab.research.google.com/github/nvidia/Cosmos-Tokenizer/blob/main/notebook/Video_Tokenization.ipynb#scrollTo=gZFPrGCBGwtC
    ## ALso have seen 17 and 25 used
    ## Lower window size adds more "metadata" latent frames (1 per chunk, where # chunks == original frames / window_size) and can improve temporal coherence
    ## thus for every "window_size" original frames, there are "window_size // temporal_compression + 1" latent frames
    tokenizer_temporal_window_size = 129

    print(f"Tokenizing video: {input_mp4_filepath}...")
    tokens_np = do_visual_tokenizer(input_mp4_filepath, tokenizer_temporal_window_size, temporal_compression, spatial_compression, output_tokens_filepath)
    print(f"\tVisual tokens shape: {tokens_np.shape}\n\n")

    print(f"Saving tokens to: {output_tokens_filepath}...\n\n")
    np.save(output_tokens_filepath, tokens_np)

    if output_mp4_filepath is not None:
        print(f"Reconstructing video from tokens and saving to: {output_mp4_filepath}...\n\n")
        original_fps, original_width, original_height = get_video_properties(input_mp4_filepath)
        do_visual_detokenizer(output_tokens_filepath + ".npy", original_fps, original_width, original_height, tokenizer_temporal_window_size, temporal_compression, spatial_compression, output_mp4_filepath)
        print(f"\nFinished. Reconstructed video saved to: {output_mp4_filepath}\n")
