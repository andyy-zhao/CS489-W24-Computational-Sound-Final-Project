import pygame
import sys
import threading
import numpy as np
from models.real_time_pitch_detector import detect_pitch, create_audio_stream, fs, CHUNK

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
CHART_HEIGHT = 200
CHART_WIDTH = 700
CHART_X = (WINDOW_WIDTH - CHART_WIDTH) // 2
CHART_Y = 280
NUM_BARS = 500  # Increased for smoother visualization

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
BLUE = (41, 128, 185)
LIGHT_BLUE = (52, 152, 219)
RED = (231, 76, 60)
PURPLE = (142, 68, 173)
GREEN = (46, 204, 113)

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
        self.background_gradient = create_gradient(WHITE, LIGHT_GRAY, WINDOW_HEIGHT)
        self.bar_gradient = create_gradient(BLUE, PURPLE, CHART_HEIGHT)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.note_font = pygame.font.Font(None, 72)
        self.freq_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 24)
        
        # Initialize variables
        self.is_recording = False
        self.stream = None
        self.update_thread = None
        self.current_note = "--"
        self.current_freq = "--"
        
        # Thread synchronization
        self.stream_lock = threading.Lock()
        
        # Chart variables
        self.spectrum_data = None
        self.max_freq = 4200  # Extended frequency range
        self.detected_freq = None
        
        # Button properties
        self.button_width = 200
        self.button_height = 50
        self.button_x = WINDOW_WIDTH // 2 - self.button_width // 2
        self.button_y = WINDOW_HEIGHT - 80
        self.button_color = BLUE
        self.button_hover_color = LIGHT_BLUE
        
        # Frequency label step
        self.freq_label_step = 600  # Adjusted for new range
        
    def draw_text(self, text, font, color, x, y, background=None):
        text_surface = font.render(text, True, color, background)
        text_rect = text_surface.get_rect(center=(x, y))
        self.screen.blit(text_surface, text_rect)
        
    def draw_button(self, text):
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(self.button_x, self.button_y, self.button_width, self.button_height)
        
        # Draw button with rounded corners
        color = self.button_hover_color if button_rect.collidepoint(mouse_pos) else self.button_color
        pygame.draw.rect(self.screen, color, button_rect, border_radius=10)
        
        # Draw text
        self.draw_text(text, self.freq_font, WHITE, WINDOW_WIDTH // 2, self.button_y + self.button_height // 2)
        
        return button_rect
    
    def draw_chart(self):
        # Draw chart background
        chart_surface = pygame.Surface((CHART_WIDTH, CHART_HEIGHT))
        chart_surface.fill(WHITE)
        
        # Draw grid lines
        num_vertical_lines = 8  # Increased for more frequent grid lines
        for i in range(num_vertical_lines):
            x = (CHART_WIDTH * i) // (num_vertical_lines - 1)
            pygame.draw.line(chart_surface, LIGHT_GRAY, (x, 0), (x, CHART_HEIGHT), 1)
        for i in range(5):
            y = (CHART_HEIGHT * i) // 4
            pygame.draw.line(chart_surface, LIGHT_GRAY, (0, y), (CHART_WIDTH, y), 1)
        
        if self.spectrum_data is not None:
            freqs, magnitudes = self.spectrum_data
            
            # Find indices corresponding to our frequency range
            freq_mask = freqs <= self.max_freq
            freqs = freqs[freq_mask]
            magnitudes = magnitudes[freq_mask]
            
            # Calculate bar properties
            bar_width = CHART_WIDTH / NUM_BARS
            bar_spacing = 0.5  # Reduced spacing for smoother appearance
            
            # Create frequency bins
            freq_bins = np.linspace(0, self.max_freq, NUM_BARS + 1)
            
            # Draw frequency bars with gradient
            for i in range(NUM_BARS):
                # Get frequencies in this bin
                mask = (freqs >= freq_bins[i]) & (freqs < freq_bins[i + 1])
                if np.any(mask):
                    magnitude = np.max(magnitudes[mask])
                    
                    # Calculate bar height with smoothing
                    bar_height = int(magnitude * CHART_HEIGHT)
                    
                    # Create gradient for this bar
                    if bar_height > 0:
                        for y in range(bar_height):
                            color = self.bar_gradient[min(y, CHART_HEIGHT-1)]
                            pygame.draw.line(
                                chart_surface,
                                color,
                                (i * bar_width, CHART_HEIGHT - y),
                                ((i + 1) * bar_width - bar_spacing, CHART_HEIGHT - y)
                            )
            
            # Draw detected frequency line
            if self.detected_freq and self.detected_freq <= self.max_freq:
                x = int((self.detected_freq / self.max_freq) * CHART_WIDTH)
                for offset in range(-2, 3):
                    pygame.draw.line(chart_surface, (*RED, 100),
                                   (x + offset, 0),
                                   (x + offset, CHART_HEIGHT),
                                   2 if offset == 0 else 1)
        
        # Draw the chart surface
        self.screen.blit(chart_surface, (CHART_X, CHART_Y))
        pygame.draw.rect(self.screen, DARK_GRAY, (CHART_X, CHART_Y, CHART_WIDTH, CHART_HEIGHT), 2)
        
        # Draw frequency labels with more points
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
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    button_rect = pygame.Rect(self.button_x, self.button_y, self.button_width, self.button_height)
                    if button_rect.collidepoint(event.pos):
                        self.toggle_recording()
            
            # Draw background gradient
            self.draw_background()
            
            # Draw title with shadow effect
            self.draw_text("Real-time Pitch Detector", self.title_font, DARK_GRAY, WINDOW_WIDTH // 2, 52)
            self.draw_text("Real-time Pitch Detector", self.title_font, BLUE, WINDOW_WIDTH // 2, 50)
            
            # Draw note with background
            note_bg = pygame.Surface((300, 80))
            note_bg.fill(WHITE)
            note_bg.set_alpha(200)
            self.screen.blit(note_bg, (WINDOW_WIDTH // 2 - 150, 110))
            self.draw_text(f"Note: {self.current_note}", self.note_font, BLUE, WINDOW_WIDTH // 2, 150)
            
            # Draw frequency
            self.draw_text(f"Frequency: {self.current_freq} Hz", self.freq_font, DARK_GRAY, WINDOW_WIDTH // 2, 220)
            
            # Draw chart
            self.draw_chart()
            
            # Draw button
            button_text = "Stop Recording" if self.is_recording else "Start Recording"
            self.draw_button(button_text)
            
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