import pygame
import sys
import threading
import numpy as np
from models.real_time_pitch_detector import detect_pitch, create_audio_stream, fs, CHUNK
import pyaudio
import wave
from datetime import datetime

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
CHART_HEIGHT = 200
CHART_WIDTH = 700
CHART_X = (WINDOW_WIDTH - CHART_WIDTH) // 2
CHART_Y = 280  # Move chart back up since buttons will be at bottom
NUM_BARS = 500  # Increased for smoother visualization

# Audio Recording Settings
RECORD_FORMAT = pyaudio.paInt16
RECORD_CHANNELS = 1
RECORD_RATE = 44100
RECORD_CHUNK = 1024
RECORD_DURATION = 10  # Recording duration in seconds

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (245, 247, 250)
DARK_GRAY = (75, 85, 99)
BLUE = (59, 130, 246)  # Modern blue
LIGHT_BLUE = (96, 165, 250)
RED = (239, 68, 68)
PURPLE = (147, 51, 234)
GREEN = (34, 197, 94)
BACKGROUND_TOP = (249, 250, 251)  # Very light gray
BACKGROUND_BOTTOM = (243, 244, 246)  # Slightly darker light gray
CHART_GRID = (229, 231, 235)  # Light gray for grid lines

def create_gradient(color1, color2, height):
    gradient = []
    for i in range(height):
        ratio = i / height
        color = tuple(int(color1[j] + (color2[j] - color1[j]) * ratio) for j in range(3))
        gradient.append(color)
    return gradient

class PitchVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Real-time Pitch Detector")
        self.clock = pygame.time.Clock()
        
        # Create gradients
        self.background_gradient = create_gradient(BACKGROUND_TOP, BACKGROUND_BOTTOM, WINDOW_HEIGHT)
        self.bar_gradient = create_gradient(BLUE, PURPLE, CHART_HEIGHT)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.note_font = pygame.font.Font(None, 86)  # Increased size for note display
        self.freq_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 24)
        self.button_font = pygame.font.Font(None, 32)  # Smaller font for buttons
        
        # Initialize variables
        self.is_recording = False
        self.is_recording_wav = False
        self.stream = None
        self.update_thread = None
        self.current_note = "--"
        self.current_freq = "--"
        
        # Thread synchronization
        self.stream_lock = threading.Lock()
        
        # Chart variables
        self.spectrum_data = None
        self.max_freq = 4200
        self.detected_freq = None
        
        # Button properties with updated dimensions
        self.button_width = 180  # Smaller width
        self.button_height = 45  # Smaller height
        self.button_x = WINDOW_WIDTH // 2 - 200  # Left of center
        self.button_y = WINDOW_HEIGHT - 70  # Near bottom of window
        self.button_color = BLUE
        self.button_hover_color = LIGHT_BLUE
        
        # WAV recording button properties
        self.wav_button_width = 180  # Smaller width
        self.wav_button_height = 45  # Smaller height
        self.wav_button_x = WINDOW_WIDTH // 2 + 20  # Right of center
        self.wav_button_y = WINDOW_HEIGHT - 70  # Same height as other button
        self.wav_button_color = GREEN
        self.wav_button_hover_color = (45, 212, 108)  # Lighter green
        
        # Frequency label step
        self.freq_label_step = 600
        
    def draw_text(self, text, font, color, x, y, background=None):
        text_surface = font.render(text, True, color, background)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)
        
    def draw_button(self, text, x, y, width, height, color, hover_color):
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(x, y, width, height)
        
        # Draw button shadow
        shadow_rect = pygame.Rect(x + 2, y + 2, width, height)
        pygame.draw.rect(self.screen, (*DARK_GRAY, 100), shadow_rect, border_radius=12)
        
        # Draw button with rounded corners
        current_color = hover_color if button_rect.collidepoint(mouse_pos) else color
        pygame.draw.rect(self.screen, current_color, button_rect, border_radius=12)
        
        # Add subtle gradient overlay
        gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for i in range(height):
            alpha = int(100 * (1 - i/height))  # Gradient from semi-transparent to transparent
            pygame.draw.line(gradient_surface, (255, 255, 255, alpha), (0, i), (width, i))
        gradient_surface = pygame.transform.scale(gradient_surface, (width, height))
        
        # Apply gradient with rounded corners
        mask = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255), mask.get_rect(), border_radius=12)
        gradient_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(gradient_surface, button_rect)
        
        # Draw text with slight shadow using smaller button font
        self.draw_text(text, self.button_font, (*BLACK, 50), x + width // 2 + 1, y + height // 2 + 1)  # Shadow
        self.draw_text(text, self.button_font, WHITE, x + width // 2, y + height // 2)  # Text
        
        return button_rect
    
    def draw_chart(self):
        # Draw chart background with rounded corners
        chart_surface = pygame.Surface((CHART_WIDTH, CHART_HEIGHT))
        chart_surface.fill(WHITE)
        
        # Draw grid lines with updated color
        num_vertical_lines = 8
        for i in range(num_vertical_lines):
            x = (CHART_WIDTH * i) // (num_vertical_lines - 1)
            pygame.draw.line(chart_surface, CHART_GRID, (x, 0), (x, CHART_HEIGHT), 1)
        for i in range(5):
            y = (CHART_HEIGHT * i) // 4
            pygame.draw.line(chart_surface, CHART_GRID, (0, y), (CHART_WIDTH, y), 1)
        
        if self.spectrum_data is not None:
            freqs, magnitudes = self.spectrum_data
            
            # Find indices corresponding to our frequency range
            freq_mask = freqs <= self.max_freq
            freqs = freqs[freq_mask]
            magnitudes = magnitudes[freq_mask]
            
            # Calculate bar properties
            bar_width = CHART_WIDTH / NUM_BARS
            bar_spacing = 0.3  # Reduced spacing for smoother appearance
            
            # Create frequency bins
            freq_bins = np.linspace(0, self.max_freq, NUM_BARS + 1)
            
            # Draw frequency bars with gradient and glow effect
            for i in range(NUM_BARS):
                mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1])
                if np.any(mask):
                    magnitude = np.max(magnitudes[mask])
                    bar_height = int(magnitude * CHART_HEIGHT)
                    
                    if bar_height > 0:
                        # Draw glow effect
                        glow_surface = pygame.Surface((int(bar_width), bar_height), pygame.SRCALPHA)
                        for y in range(bar_height):
                            alpha = int(150 * (y/bar_height))  # Gradient alpha
                            color = self.bar_gradient[min(y, CHART_HEIGHT-1)]
                            glow_color = (*color, alpha)
                            pygame.draw.line(
                                glow_surface,
                                glow_color,
                                (0, bar_height - y),
                                (bar_width - bar_spacing, bar_height - y)
                            )
                        chart_surface.blit(glow_surface, (i * bar_width, 0))
            
            # Draw detected frequency line with glow effect
            if self.detected_freq and self.detected_freq <= self.max_freq:
                x = int((self.detected_freq / self.max_freq) * CHART_WIDTH)
                glow_width = 5
                for offset in range(-glow_width, glow_width + 1):
                    alpha = 255 - int(200 * abs(offset) / glow_width)
                    pygame.draw.line(chart_surface, (*RED, alpha),
                                   (x + offset, 0),
                                   (x + offset, CHART_HEIGHT),
                                   2 if offset == 0 else 1)
        
        # Create a rounded rectangle mask
        mask = pygame.Surface((CHART_WIDTH, CHART_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(mask, WHITE, mask.get_rect(), border_radius=15)
        
        # Apply the mask to the chart surface
        final_surface = pygame.Surface((CHART_WIDTH, CHART_HEIGHT), pygame.SRCALPHA)
        final_surface.blit(chart_surface, (0, 0))
        final_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Draw the final surface
        self.screen.blit(final_surface, (CHART_X, CHART_Y))
        pygame.draw.rect(self.screen, DARK_GRAY, (CHART_X, CHART_Y, CHART_WIDTH, CHART_HEIGHT), 2, border_radius=15)
        
        # Draw frequency labels
        for i, freq in enumerate(range(0, self.max_freq + 1, self.freq_label_step)):
            x = CHART_X + (CHART_WIDTH * i * self.freq_label_step) // self.max_freq
            self.draw_text(str(freq), self.label_font, DARK_GRAY, x, CHART_Y + CHART_HEIGHT + 20)
    
    def draw_background(self):
        for y in range(WINDOW_HEIGHT):
            pygame.draw.line(self.screen, self.background_gradient[y], (0, y), (WINDOW_WIDTH, y))
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.stream = create_audio_stream()
        self.update_thread = threading.Thread(target=self.update_display)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop_recording(self):
        with self.stream_lock:
            self.is_recording = False
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                finally:
                    self.stream = None
            
            # Wait for the update thread to finish
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1.0)
            
            # Reset display variables
            self.current_note = "--"
            self.current_freq = "--"
            self.spectrum_data = None
            self.detected_freq = None
    
    def update_display(self):
        while self.is_recording:
            with self.stream_lock:
                if not self.stream:
                    break
                try:
                    note, freq, spectrum = detect_pitch(self.stream)
                    if note and freq and spectrum:
                        self.current_note = note
                        self.current_freq = f"{freq:.1f}"
                        self.detected_freq = freq
                        self.spectrum_data = spectrum
                except Exception as e:
                    print(f"Error in update_display: {e}")
                    break
    
    def record_to_wav(self):
        if self.is_recording_wav:
            return
            
        self.is_recording_wav = True
        output_file = "recording.wav"
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(format=RECORD_FORMAT, channels=RECORD_CHANNELS, rate=RECORD_RATE,
                          input=True, frames_per_buffer=RECORD_CHUNK)
        
        print(f"Recording to {output_file}...")
        frames = []
        
        # Calculate total chunks to record
        total_chunks = int(RECORD_RATE / RECORD_CHUNK * RECORD_DURATION)
        
        # Record in real-time with manual stop option
        for _ in range(total_chunks):
            if not self.is_recording_wav:  # Check if recording should stop
                break
            data = stream.read(RECORD_CHUNK)
            frames.append(data)
        
        print("Recording complete.")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save as a .wav file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(RECORD_CHANNELS)
            wf.setsampwidth(audio.get_sample_size(RECORD_FORMAT))
            wf.setframerate(RECORD_RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"Audio saved to {output_file}")
        self.is_recording_wav = False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check pitch detection button
                    pitch_button_rect = pygame.Rect(self.button_x, self.button_y, self.button_width, self.button_height)
                    if pitch_button_rect.collidepoint(event.pos):
                        self.toggle_recording()
                    
                    # Check WAV recording button
                    wav_button_rect = pygame.Rect(self.wav_button_x, self.wav_button_y, self.wav_button_width, self.wav_button_height)
                    if wav_button_rect.collidepoint(event.pos):
                        if self.is_recording_wav:
                            self.is_recording_wav = False
                        else:
                            threading.Thread(target=self.record_to_wav).start()
            
            # Draw background gradient
            self.draw_background()
            
            # Draw title with enhanced shadow effect
            shadow_offset = 2
            for offset in range(1, 4):
                alpha = 100 - offset * 25
                self.draw_text("Real-time Pitch Detector", self.title_font, (*DARK_GRAY, alpha), 
                             WINDOW_WIDTH // 2 + offset, 52 + offset)
            self.draw_text("Real-time Pitch Detector", self.title_font, BLUE, WINDOW_WIDTH // 2, 50)
            
            # Draw note with enhanced background
            note_bg = pygame.Surface((320, 100), pygame.SRCALPHA)
            pygame.draw.rect(note_bg, (*WHITE, 230), note_bg.get_rect(), border_radius=15)
            self.screen.blit(note_bg, (WINDOW_WIDTH // 2 - 160, 100))
            self.draw_text(f"Note: {self.current_note}", self.note_font, BLUE, WINDOW_WIDTH // 2, 150)
            
            # Draw frequency with subtle shadow
            self.draw_text(f"Frequency: {self.current_freq} Hz", self.freq_font, (*DARK_GRAY, 100), 
                         WINDOW_WIDTH // 2 + 1, 221)
            self.draw_text(f"Frequency: {self.current_freq} Hz", self.freq_font, DARK_GRAY, 
                         WINDOW_WIDTH // 2, 220)
            
            # Draw chart
            self.draw_chart()
            
            # Draw buttons
            pitch_button_text = "Stop Recording" if self.is_recording else "Start Recording"
            self.draw_button(pitch_button_text, self.button_x, self.button_y, self.button_width, self.button_height,
                           self.button_color, self.button_hover_color)
            
            wav_button_text = "Recording..." if self.is_recording_wav else "Record to WAV"
            self.draw_button(wav_button_text, self.wav_button_x, self.wav_button_y, self.wav_button_width, self.wav_button_height,
                           self.wav_button_color, self.wav_button_hover_color)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Clean up
        self.stop_recording()
        pygame.quit()
        sys.exit()

def main():
    app = PitchVisualizer()
    app.run()

if __name__ == "__main__":
    main() 