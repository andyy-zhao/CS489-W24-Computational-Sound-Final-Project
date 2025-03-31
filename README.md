# CS489-W24-Computational-Sound-Final-Project

A real-time audio processing application that provides pitch detection, autotuning, and pitch shifting capabilities.

## Features

- Real-time pitch detection
- Autotuning functionality
- Pitch shifting
- Audio recording and playback
- Interactive GUI interface

## Project Structure

```
.
├── data/                      # Audio data files (not tracked in git)
├── gui/                       # GUI components
│   ├── pitch_visualizer.py    # Real-time pitch visualization
│   └── __init__.py
├── models/                    # Core model implementations
│   ├── detect_note_from_wav.py
│   ├── pitch_shifter.py       # Pitch shifting implementation
│   ├── real_time_pitch_detector.py
│   ├── vocoder.py            # Audio processing
│   └── __init__.py
├── recordings/                # Recorded audio files
├── utils/                     # Utility functions
│   ├── audio_recorder.py      # Audio recording functionality
│   ├── pitch_data.py         # Pitch data utilities
│   └── __init__.py
├── main.py                    # Main application entry point
├── requirements.txt           # Project dependencies
├── .gitignore                # Git ignore rules
└── README.md                  # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/andyy-zhao/CS489-W24-Computational-Sound-Final-Project.git
cd CS489-W24-Computational-Sound-Final-Project
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `data` folder in the project root and add your audio files:
```bash
mkdir data
```

## Data Folder

The `data/` folder is not tracked in Git due to the large size of audio files. You'll need to:
1. Create the `data/` folder locally
2. Add your own audio files to the folder
3. The folder is already included in `.gitignore` to prevent accidental commits

## Usage

To run the application with the pitch visualization interface:
```bash
python gui/pitch_visualizer.py
```

Alternatively, you can run the main application:
```bash
python main.py
```

## Dependencies

Key dependencies include:
- PyAudio for audio input/output
- NumPy for numerical computations
- Librosa for audio processing
- SoundFile for audio file handling

For a complete list of dependencies, see `requirements.txt`.

## License

This project is part of CS489 - Computational Sound at the University of Waterloo.