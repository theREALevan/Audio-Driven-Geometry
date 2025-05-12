# Audio-Driven Geometry Visualization

An interactive application that creates dynamic geometric visualizations driven by audio input. The application allows users to draw or load shapes, analyze them using Fourier transforms, and animate them based on audio features.

## Features

- **Drawing Interface**
  - Free-hand drawing of shapes
  - Background image support
  - Stroke management (add/delete/clear)
  - Real-time preview

- **Fourier Analysis**
  - Shape decomposition using Fourier transforms
  - Adjustable number of coefficients
  - Shape reconstruction with rotation control
  - Interactive coefficient manipulation

- **Audio Analysis**
  - Support for WAV and MP3 files
  - Real-time audio visualization
  - Spectral analysis
  - Time-frequency representation
  - Spectral centroid tracking

- **Shape Animation**
  - Audio-driven shape modulation
  - Multiple shape support
  - Interactive shape manipulation
    - Translation
    - Rotation
    - Scaling (mouse wheel)
  - Color modulation based on spectral centroid

## Requirements

- Python 3.6+
- PyQt5
- NumPy
- Librosa
- SciPy
- Matplotlib
- Pillow (PIL)

## Installation

1. Clone the repository:
```bash
git clone git@github.com:theREALevan/Audio-Driven-Geometry.git
cd Audio_driven_geometry
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python integrated_visualizer.py
```

2. Drawing Shapes:
   - Use the drawing canvas to create shapes
   - Load a background image for reference
   - Click "Done Drawing" when finished

3. Fourier Analysis:
   - Adjust the number of coefficients using the slider
   - Rotate the shape using the rotation slider
   - Click "Add to Scene" to add the shape to the animation

4. Audio Integration:
   - Load an audio file (WAV or MP3)
   - The spectrum and spectrogram will update automatically
   - Click on the spectrum to add modulations at specific frequencies

5. Shape Animation:
   - Select a shape to modify its properties
   - Add modulations through the interface
   - Adjust the disturbance level using the slider
   - Use mouse wheel to scale selected shapes

## Controls

- **Drawing**
  - Left mouse button: Draw
  - "Delete Last": Remove last stroke
  - "Clear Canvas": Reset drawing

- **Shape Manipulation**
  - Slider: Size
  - Mouse wheel: Scale
  - Left click + drag: Move shape

- **General**
  - ESC: Quit application

## Acknowledgments

- PyQt5 for the GUI framework
- Librosa for audio processing
- NumPy for numerical computations
- Matplotlib for visualization 
