import os
import sys
import argparse
from audio_processor import process_audio_file, save_output
from audio_reconstructor import reconstruct_audio

def run_pipeline(input_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Setup output directory
    output_dir = "generated_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 1. Encode
    print(f"--- Step 1: Encoding {input_file} ---")
    audio_data, sr = process_audio_file(input_file)
    
    if audio_data is None:
        print("Encoding failed.")
        return

    npz_file = os.path.join(output_dir, f"{base_name}_data.npz")
    save_output(audio_data, sr, filename=npz_file)

    # 2. Decode
    print(f"\n--- Step 2: Decoding to {npz_file} ---")
    output_mp3 = os.path.join(output_dir, f"reconstructed_{base_name}.mp3")
    reconstruct_audio(input_file=npz_file, output_file=output_mp3)

    print(f"\nPipeline complete!")
    print(f"Intermediate data: {npz_file}")
    print(f"Reconstructed audio: {output_mp3}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode and decode an audio file.")
    parser.add_argument("input_file", help="Path to the input audio file (mp3/wav)")
    args = parser.parse_args()

    run_pipeline(args.input_file)
