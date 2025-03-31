import librosa
import numpy as np
import soundfile as sf

def interpolate_freq(idxs: np.ndarray, arr: np.ndarray):
    start = idxs.astype(int)
    frac = (idxs - start)[None, :, None]
    shifted_arr = np.concatenate((arr[:, 1:, :], np.zeros((arr.shape[0], 1, arr.shape[2]))), axis=1)
    return arr[:, start, :] * (1 - frac) + shifted_arr[:, start, :] * frac

def round_interpolate_freq(idxs: np.ndarray, arr: np.ndarray):
    return arr[:, (idxs + 0.5).astype(int), :]

def interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    start = (idxs + 0.5).astype(int)
    frac = (idxs - start)
    shifted_arr = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1)
    return arr[:, start] * (1 - frac) + shifted_arr[:, start] * frac


def round_interpolate_time(idxs: np.ndarray, arr: np.ndarray):
    return arr[:, (idxs + 0.5).astype(int)]  

def pitch_shift():
    y, sr = librosa.load("../data/E4.wav", sr=None, mono=True)

    # compute STFT
    # consistent with pitch detection CHUNK size
    w_len = 1024 * 4

    # default uses Hanning window
    X = librosa.stft(y, n_fft=w_len, win_length=w_len)

    # num of frequency bins and time frames
    num_freqs, num_frames = X.shape

    # shift by 1 semitone. (calculate later in real time)
    # 1 octave is speed up by 2x
    scaling = 2 ** (2 / 12)

    '''
    When the pitch of an audio signal is changed, 
    the time-scale of the signal is also changed. For example:
    If pitch is being increased, you're making the sound happen faster (compressing the time).
    If pitch is decreased, you're slowing down the sound (stretching the time).

    While updated_num_frames represents the new number of time frames in the pitch-shifted signal, 
    the frame rate is adjusted so that the duration of the output signal stays the same
    '''

    updated_num_frames = np.floor(num_frames * scaling).astype(int)

    updated_t_frames = np.arange(updated_num_frames)


    # increased time_frames due to scaling (more dense, later will use interpolation to fill them in)
    original_indices = np.minimum(updated_t_frames / scaling, num_frames - 1)

    # getting magnitude and phase from the stft
    magnitude = np.abs(X)
    phases = np.angle(X)


    phase_diffs = phases - np.concatenate((np.zeros((num_freqs, 1)), phases[:, :-1]), axis=1)
    phase_diffs = np.mod(phase_diffs, np.pi * 2)



    # # Interpolate the magnitude and phase differences over the new time indices.
    # # (Assume that interpolate_time and round_interpolate_time work along the time axis.)
    shifted_magnitude = interpolate_time(original_indices, magnitude)
    shifted_phase_diffs = interpolate_time(original_indices, phase_diffs)
    unshifted_phases = round_interpolate_time(original_indices, phases)

    # # Initialize an array for the new (shifted) phases
    shifted_phases = np.zeros((num_freqs, updated_num_frames))
    shifted_phases[:, 0] = shifted_phase_diffs[:, 0]

    # # Accumulate phase information for each new time frame
    for t in range(1, updated_num_frames):
        # Accumulate phase from previous frame and the current phase difference
        time_phases = shifted_phases[:, t - 1] + shifted_phase_diffs[:, t]
        # Get the interpolated original phase for the current time frame
        freq_phases = unshifted_phases[:, t]
        # Compute a transient factor to help smooth abrupt changes in magnitude
        transient = (shifted_magnitude[:, t] - shifted_magnitude[:, t - 1]) / (shifted_magnitude[:, t] + shifted_magnitude[:, t - 1])
        transient[transient < 0.5] = 0
        transient[transient >= 0.5] = 1
        # Blend the accumulated phase and the frequency-phase based on transient detection
        shifted_phases[:, t] = np.mod(freq_phases * transient + time_phases * (1 - transient), np.pi * 2)

    # # Reconstruct the complex STFT from the shifted magnitude and phases
    synth_stft = shifted_magnitude * np.exp(shifted_phases * 1j)

    # # Reconstruct the time-domain waveform using the inverse STFT
    new_waveform = librosa.istft(synth_stft, n_fft=w_len, window="hann")

    # Save the new waveform as a .wav file
    sf.write('../data/pitch_shifted_output.wav', new_waveform, int(sr * scaling))


pitch_shift()