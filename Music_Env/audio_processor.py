import librosa
import numpy as np
import sys

def process_audio_file(file_path):
    """
    Loads an audio file and returns the 1D time-series array.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        np.ndarray: 1D array of audio time series.
        float: Sample rate of the audio.
    """
    try:
        # librosa.load returns a tuple (audio_time_series, sample_rate)
        # sr=None preserves the original sampling rate
        print(f"Loading {file_path}...")
        y, sr = librosa.load(file_path, sr=None)
        
        print("Audio successfully loaded.")
        print(f"Shape of the array: {y.shape}")
        print(f"Sample rate: {sr}")
        print(f"Duration: {len(y) / sr:.2f} seconds")
        
        return y, sr
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

def save_output(audio, sr, filename="audio_data.npz"):
    """
    Saves the audio array and sample rate to a compressed .npz file.
    """
    np.savez(filename, audio=audio, sr=sr)
    print(f"Saved audio data and sample rate to {filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        target_file = "test_audio.wav"
        
    audio_data, rate = process_audio_file(target_file)
    
    if audio_data is not None:
        # Example: showing first 10 values
        print(f"First 10 values of the signal: {audio_data[:10]}")
        save_output(audio_data, rate)
