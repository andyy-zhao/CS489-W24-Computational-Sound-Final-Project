from utils.pitch_data import get_closest_note
import pyaudio
import numpy as np

fs=44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024 * 4

THRESHOLD = 700 

def detect_pitch(stream):
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.int16)

    if np.max(np.abs(audio_data)) < THRESHOLD:
        return None, None, None

    N = len(audio_data)
    df = fs / N
    Npos = N // 2
    f = df * np.arange(Npos)
    w = np.hanning(N)
    X = np.fft.fft(w * audio_data)
    Xpos = np.sqrt(np.mean(w ** 2)) * np.abs(2 * X[:Npos]) / N
    XdB = 20 * np.log10(Xpos)
   
    max_idx = np.argmax(XdB) 
    max_freq = f[max_idx]
    
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