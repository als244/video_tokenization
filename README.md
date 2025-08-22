# Video Tokenization

Convert `.mp4` to latent audio & visual frames. 

- Visual Tokenizer: [Cosmos](https://github.com/NVIDIA/Cosmos-Tokenizer) (causal VAE)
    - 4x8x8, 8x8x8, or 8x16x16 compression (temporal x spatial x spatial)
    - Discete or Continuous
        - Discrete tokenizer produces latent frames with 16-bit token identifiers
            - Only recommended for 4x8x8, but continuous is preferred
        - Continous tokenizer produces latent frames with 16 channels of bfloat16
            - Thus reconstruction quality of continous tokenizer is thus significantly better as it uses 16x more bytes for a given compression ratio

- Audio Tokenizer: [DAC](https://github.com/descriptinc/descript-audio-codec) (VQ-GAN)
    - Handles 44.1Khz audio waves and produces discrete 10-bit codes
        - Audio contains significantly less information than visual percept; even for low visual resolution (480p) < 5% of tokens are audio


## Usage

You can use this tool to easily create training data and do eye-tests of reconstruction quality (to ensure the training data is high-fidelity).

To produce audio & visual tokens run the following script:

```shell
./tokenize_video <orig_video_dir> <orig_video_file> <video_nickname> <is_discrete> <temporal_factor> <spatial_factor> <output_tokens_dir> <reconstructed_dir>
```

Arguments:
- `orig_video_dir`: path to directory in which the `.mp4` resides (e.g. `example_videos`)
    - The extracted audio waveform will be saved down in this directory
- `orig_video_file`: the filename of input video (e.g. `my_first_video.mp4`)
- `video_nickname`: a nickname for the video which will be used to create filenames for tokens and reconstructed files (e.g. `my_first_video_continuous_8x16x16`)
- `is_discrete`: 0 or 1. Determines if visual tokenizer produces discrete tokens. Recommended setting is 0.
- `temporal_factor`: 4 or 8. Determiens the temporal compression rate. Temporal factor of 4 is only available paired with spatial factor of 8.
- `spatial_factor`: 8 or 16. Determines the spatial (for both height & width) compression rate. 
- `output_tokens_dir`: path to directory in which the tokens will be saved (e.g. `example_tokens`)
- `reconstructed_dir`: path to direcotry in which the reconstructed files (visual, audio, & combined) will be saved (e.g. `example_videos/reconstructed`)

#### Description

This will first produce visual latents and save down visual tokens (along with reconstructing visual aspect of video based on the tokens produced). For continuous tokenization(i.e. `is_discrete = 0`) the resulting visual tokens will have shape `[1, \# latent frames, \# latent height, \# latent width, 16]`. For discrete tokenization there are only 4 dimensions with the last dimension (the 16 channels) being removed.

Then the audio tokenizer tokenizer will extract a `.wav` from the `.mp4` and tokenize this in chunks corresponding to the latent frames and save down audio tokens (along with reconstructing the audio aspect of video based on audio tokens, visual temporal compression factor, & target fps == original video fps).

Finally, the reconstructed visual and audio representations will be merged into a full video that can be compared against the original.

