import numpy as np
import soundfile as sf
import sys

def test_mp3_write():
    sr = 22050
    t = np.linspace(0, 1, sr)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    try:
        sf.write('test_output.mp3', audio, sr)
        print("Success: soundfile wrote mp3")
    except Exception as e:
        print(f"Failure: {e}")

if __name__ == "__main__":
    test_mp3_write()
