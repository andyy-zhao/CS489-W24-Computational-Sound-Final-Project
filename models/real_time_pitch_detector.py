from utils.pitch_data import get_closest_note
import pyaudio
import numpy as np

# sample rate
fs=44100
# 16-bit
FORMAT = pyaudio.paInt16
# Mono
CHANNELS = 1
CHUNK = 1024 * 4

THRESHOLD = 700  # Adjust this based on noise level

def detect_pitch(stream):
    data = stream.read(CHUNK)

    # Convert the byte data to a NumPy array
    audio_data = np.frombuffer(data, dtype=np.int16)

    if np.max(np.abs(audio_data)) < THRESHOLD:
        return None, None, None  # Return None if signal is too weak

    # FFT analysis
    N = len(audio_data)
    df = fs / N
    Npos = N // 2
    f = df * np.arange(Npos)

    # use Hann window
    w = np.hanning(N)

    # Compute FFT
    X = np.fft.fft(w * audio_data)
    Xpos = np.sqrt(np.mean(w ** 2)) * np.abs(2 * X[:Npos]) / N  # Scaling by N for reconstruction
    XdB = 20 * np.log10(Xpos)
   
    max_idx = np.argmax(XdB) 
    max_freq = f[max_idx]
    
    # Normalize XdB for visualization
    XdB_normalized = (XdB - np.min(XdB)) / (np.max(XdB) - np.min(XdB))
    
    return get_closest_note(max_freq), max_freq, (f, XdB_normalized)

def create_audio_stream():
    p = pyaudio.PyAudio()
    return p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=fs,
        input=True,
        frames_per_buffer=CHUNK
    )



    