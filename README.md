# MedicalScribeLocal

MedicalScribeLocal is a desktop application designed to act as an automated medical scribe running locally on your machine. It utilizes state-of-the-art local AI models for speech-to-text, speaker diarization, and semantic analysis to generate comprehensive medical reports from audio recordings or real-time dictation.

## Features

- **Local Processing**: Ensures patient data privacy by processing audio and generating reports entirely on your local machine without relying on external cloud APIs.
- **Audio Transcription**: Powered by `faster-whisper` for highly accurate and fast speech recognition.
- **Speaker Diarization**: Uses `pyannote.audio` to distinguish between different speakers (e.g., Doctor and Patient) in the audio recordings.
- **Real-time Processing**: Supports real-time audio capture and transcription via the real-time pipeline.
- **Report Generation**: Automatically generates formatted medical reports as Word documents (using `python-docx`).
- **Modern GUI**: Features an intuitive and modern graphical user interface built with `customtkinter`.
- **Semantic Disambiguation**: Employs advanced semantic processing to accurately interpret and categorize medical terminology.

## Prerequisites

- Python 3.9 or higher
- A CUDA-compatible GPU is highly recommended for optimal performance, as the application relies heavily on PyTorch-based AI models.

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd MedicalScribeLocal
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   .\venv\Scripts\activate
   
   # Activate on Linux/macOS:
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up local AI models:**
   Run the setup script to download and configure the necessary models.
   ```bash
   python setup_models.py
   ```

## Usage

You can start the application using the provided launch scripts or directly via Python.

**Using launch scripts:**
- **Windows**: Double-click `run.bat` or execute it from the command line.
- **Linux/macOS**: Execute `./build.sh` (ensure it has executable permissions).

**Directly with Python:**
```bash
python main.py
```

## Project Structure

- `main.py`: The main entry point for the application.
- `setup_models.py`: Utility script to download and prepare the required local AI models.
- `requirements.txt`: Python package dependencies.
- `src/`: Core application source code.
  - `gui.py`: The `customtkinter` graphical user interface implementation.
  - `pipeline.py`: Core processing pipeline for handling pre-recorded audio files.
  - `realtime_pipeline.py`: Pipeline dedicated to processing real-time audio input streams.
  - `report_generator.py`: Logic for formatting the transcribed and analyzed text into professional medical reports (e.g., DOCX format).
  - `semantic_disambiguator.py`: NLP component responsible for resolving ambiguous medical terms.
- `doctor_profiles/`: Directory for storing doctor-specific settings, templates, or profiles.
- `models/`: Directory where downloaded local AI models are stored.
- `output/`: Directory where the final generated medical reports are saved.

## License

[Specify License Here]
