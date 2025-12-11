import numpy as np
import soundfile as sf
import sys
import os

def reconstruct_audio(input_file="audio_data.npz", output_file="reconstructed_audio.mp3"):
    """
    Loads audio data from an .npz file and saves it as an audio file (wav, mp3, etc).
    
    Args:
        input_file (str): Path to the .npz file containing 'audio' and 'sr'.
        output_file (str): Path to save the reconstructed audio.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    try:
        print(f"Loading data from {input_file}...")
        data = np.load(input_file)
        
        if 'audio' not in data or 'sr' not in data:
            print("Error: .npz file must contain 'audio' and 'sr' keys.")
            return

        audio = data['audio']
        sr = int(data['sr']) # Ensure sr is an integer
        
        print(f"Audio shape: {audio.shape}")
        print(f"Sample rate: {sr}")
        
        print(f"Saving reconstructed audio to {output_file}...")
        sf.write(output_file, audio, sr)
        print("Done.")
        
    except Exception as e:
        print(f"Error reconstruction audio: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "audio_data.npz"
        
    reconstruct_audio(target_file)
