import librosa
import numpy as np
import soundfile as sf

class PitchShifter:
    def __init__(self):
        self.sr = None
        self.w_len = 1024 * 4  # consistent with pitch detection CHUNK size

    def interpolate_time(self, idxs: np.ndarray, arr: np.ndarray):
        start = (idxs + 0.5).astype(int)  # Start indices for interpolation
        frac = (idxs - start)  # Fractional part for interpolation
        shifted_arr = np.concatenate((arr[:, 1:], np.zeros((arr.shape[0], 1))), axis=1)  # Shift arr by 1 along time axis
        return arr[:, start] * (1 - frac) + shifted_arr[:, start] * frac  # Interpolation along time axis

    def round_interpolate_time(self, idxs: np.ndarray, arr: np.ndarray):
        return arr[:, (idxs + 0.5).astype(int)]

    def shift_pitch(self, input_file: str, output_file: str, semitones: float):
        """
        Shift the pitch of an audio file by a specified number of semitones.
        
        Args:
            input_file (str): Path to input WAV file
            output_file (str): Path to output WAV file
            semitones (float): Number of semitones to shift (positive = up, negative = down)
        """
        # Load the audio file
        y, self.sr = librosa.load(input_file, sr=None, mono=True)

        # Compute STFT
        X = librosa.stft(y, n_fft=self.w_len, win_length=self.w_len)

        # Get dimensions
        num_freqs, num_frames = X.shape

        # Calculate scaling factor based on semitones
        # 12 semitones = 1 octave (2x frequency)
        scaling = 2 ** (semitones / 12)

        # Calculate new number of frames after scaling
        updated_num_frames = np.floor(num_frames * scaling).astype(int)
        updated_t_frames = np.arange(updated_num_frames)

        # Calculate original time indices
        original_indices = np.minimum(updated_t_frames / scaling, num_frames - 1)

        # Get magnitude and phase
        magnitude = np.abs(X)
        phases = np.angle(X)

        # Calculate phase differences
        phase_diffs = phases - np.concatenate((np.zeros((num_freqs, 1)), phases[:, :-1]), axis=1)
        phase_diffs = np.mod(phase_diffs, np.pi * 2)

        # Interpolate magnitude and phase differences
        shifted_magnitude = self.interpolate_time(original_indices, magnitude)
        shifted_phase_diffs = self.interpolate_time(original_indices, phase_diffs)
        unshifted_phases = self.round_interpolate_time(original_indices, phases)

        # Initialize shifted phases
        shifted_phases = np.zeros((num_freqs, updated_num_frames))
        shifted_phases[:, 0] = shifted_phase_diffs[:, 0]

        # Accumulate phase information
        for t in range(1, updated_num_frames):
            # Accumulate phase from previous frame and current phase difference
            time_phases = shifted_phases[:, t - 1] + shifted_phase_diffs[:, t]
            # Get interpolated original phase for current frame
            freq_phases = unshifted_phases[:, t]
            # Compute transient factor for smooth magnitude changes
            transient = (shifted_magnitude[:, t] - shifted_magnitude[:, t - 1]) / (shifted_magnitude[:, t] + shifted_magnitude[:, t - 1])
            transient[transient < 0.5] = 0
            transient[transient >= 0.5] = 1
            # Blend accumulated phase and frequency-phase based on transient detection
            shifted_phases[:, t] = np.mod(freq_phases * transient + time_phases * (1 - transient), np.pi * 2)

        # Reconstruct complex STFT
        synth_stft = shifted_magnitude * np.exp(shifted_phases * 1j)

        # Inverse STFT to get time-domain waveform
        new_waveform = librosa.istft(synth_stft, n_fft=self.w_len, window="hann")

        # Save the pitch-shifted audio
        sf.write(output_file, new_waveform, int(self.sr * scaling))
