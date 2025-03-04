import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from utils.pitch_data import get_closest_note

def get_note():
    # Prompt for filename
    # filename = input("Enter the audio filename (with extension): ")
    try:
        x, fs = sf.read("./data/F#3.wav")
    except FileNotFoundError:
        print("File not found. Please check the filename and try again.")
        exit()
    
    N = len(x)
    df = fs / N
    Npos = N // 2 - 1
    f = df * np.arange(1, Npos + 1)

    # Windowing (rectangular by default)
    # w = np.ones(N)
    # Uncomment the next line to use Hann window
    w = np.hanning(N)

    # Compute FFT
    X = np.fft.fft(w * x)
    Xpos = np.sqrt(np.mean(w ** 2)) * np.abs(2 * X[:Npos]) / N  # Scaling by N for reconstruction
    XdB = 20 * np.log10(Xpos)

    # Plot spectrum
    # plt.ion()
    plt.figure()
    plt.step(f - df / 2, XdB, where='mid')  # Center plot on bins
    plt.title(f'Python Spectrum N: {N}')
    plt.xlim(0, fs / 2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('20log10(|X_k|)')
    plt.grid(True)
    plt.show()

    max_idx = np.argmax(XdB) 
    max_freq = f[max_idx]

    print(get_closest_note(max_freq))