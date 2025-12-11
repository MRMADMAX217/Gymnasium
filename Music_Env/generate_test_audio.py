import numpy as np
import soundfile as sf

def generate_sine_wave(filename='test_audio.mp3', duration=2, sample_rate=22050, frequency=440):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a sine wave
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    # Soundfile handles normalization and format conversion automatically for many formats
    # but plain float -1 to 1 is standard.
    
    print(f"Generating {filename} with sample rate {sample_rate} Hz...")
    sf.write(filename, audio, sample_rate)
    print("Done.")

if __name__ == "__main__":
    generate_sine_wave()
