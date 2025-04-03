import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from utils.pitch_data import get_closest_note

def get_note():
    try:
        x, fs = sf.read("./data/F#3.wav")
    except FileNotFoundError:
        print("File not found. Please check the filename and try again.")
        exit()
    
    N = len(x)
    df = fs / N
    Npos = N // 2 - 1
    f = df * np.arange(1, Npos + 1)

    w = np.hanning(N)
    X = np.fft.fft(w * x)
    Xpos = np.sqrt(np.mean(w ** 2)) * np.abs(2 * X[:Npos]) / N
    XdB = 20 * np.log10(Xpos)

    plt.figure()
    plt.step(f - df / 2, XdB, where='mid')
    plt.title(f'Python Spectrum N: {N}')
    plt.xlim(0, fs / 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('20log10(|X_k|)')
    plt.grid(True)
    plt.show()

    max_idx = np.argmax(XdB) 
    max_freq = f[max_idx]
    print(get_closest_note(max_freq))