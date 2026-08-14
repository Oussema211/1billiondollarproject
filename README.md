# MedicalScribeLocal

<div align="center">

**Qatar Clinical Medical Scribe - Real-Time AI Documentation**

A sophisticated, privacy-focused desktop application that transforms doctor-patient consultations into professional clinical medical reports using state-of-the-art local AI models.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()

</div>

## 🌟 Overview

MedicalScribeLocal is an advanced automated medical scribe that runs entirely on your local machine, ensuring complete patient data privacy by never sending audio or clinical information to external cloud services. The application combines multiple cutting-edge AI technologies to deliver hospital-grade clinical documentation:

- **Multilingual Speech Recognition** using Whisper AI
- **Speaker Diarization & Identification** using Pyannote and SpeechBrain
- **Clinical Report Generation** using local LLM (Llama 3.1)
- **Real-time & Batch Processing** capabilities
- **Modern Intuitive GUI** built with CustomTkinter

Designed specifically for the Qatar healthcare context (Hamad Medical Corporation standards), it supports the multilingual clinical environment where consultations commonly occur in Arabic, English, Hindi, Urdu, Tagalog, Malayalam, and French.

---

## ✨ Key Features

### 🔒 Complete Privacy & Local Processing
- **100% Offline Operation**: All AI processing occurs locally on your machine
- **No Cloud Dependencies**: No data leaves your system, ensuring HIPAA/GDPR compliance
- **Patient Data Protection**: Audio recordings and clinical notes never transmitted externally

### 🎯 Advanced AI Capabilities
- **High-Accuracy Transcription**: Powered by `faster-whisper` with multilingual support
- **Speaker Diarization**: Distinguishes between Doctor and Patient using `pyannote.audio`
- **Voice Recognition**: Doctor voice enrollment for automatic speaker identification
- **Semantic Analysis**: Clinical linguistic analysis to ensure accurate role assignment
- **Clinical Report Generation**: Local LLM transforms transcripts into structured medical reports

### 🚀 Processing Modes
- **Real-time Consultation**: Live audio capture and transcription during patient visits
- **Batch File Processing**: Process pre-recorded audio files (WAV, MP3, M4A, FLAC)
- **Fast Mode**: Optimized processing for quick turnaround

### 📋 Professional Report Output
- **Structured Clinical Notes**: Chief complaint, history, examination, diagnosis, tests, prescriptions, follow-up
- **Multiple Languages**: Generate reports in Medical English or Arabic
- **Export Formats**: Microsoft Word (.docx) and JSON
- **Hospital-Grade Quality**: Designed to meet HMC and international medical documentation standards

### 🎨 Modern User Interface
- **Intuitive Design**: Clean, professional interface with CustomTkinter
- **Three-Tab Layout**: Consultation processing, Doctor voice enrollment, Patient records
- **Real-time Feedback**: Progress indicators and status updates
- **Easy Navigation**: Simple workflow for clinical staff

---

## 🏗️ Architecture

### Core Components

```
MedicalScribeLocal/
├── Audio Processing Layer
│   ├── Whisper STT (Speech-to-Text)
│   ├── Pyannote Diarization (Speaker Segmentation)
│   └── SpeechBrain Embeddings (Voice Identification)
├── Clinical Intelligence Layer
│   ├── Semantic Disambiguator (Role Assignment)
│   └── Report Generator (LLM-based Synthesis)
├── User Interface Layer
│   ├── Real-time Recording Interface
│   ├── File Processing Interface
│   └── Voice Enrollment Interface
└── Data Management Layer
    ├── Doctor Profiles (Voice Embeddings)
    ├── Patient Records (Generated Reports)
    └── Model Storage (Local AI Models)
```

### Processing Pipeline

1. **Audio Input**: Live microphone or uploaded audio file
2. **Speech-to-Text**: Whisper transcribes audio to text with timestamps
3. **Speaker Diarization**: Pyannote identifies speaker segments
4. **Speaker Identification**: SpeechBrain matches voices to enrolled profiles
5. **Semantic Analysis**: Clinical patterns ensure correct Doctor/Patient roles
6. **Report Generation**: Local LLM synthesizes structured clinical report
7. **Output Export**: Word document and JSON generation

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.9 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 15 GB free space (for AI models)
- **Processor**: Modern CPU with AVX2 support

### Recommended Requirements
- **GPU**: NVIDIA CUDA-compatible GPU (RTX 3060 or higher)
- **VRAM**: 8 GB+ GPU memory
- **RAM**: 16 GB+ system memory
- **Storage**: SSD for faster model loading

### Hardware Acceleration
- **CUDA**: Highly recommended for optimal performance
- **CPU-only**: Supported but significantly slower processing
- **Apple Silicon**: MPS acceleration supported on M1/M2/M3 Macs

---

## 🚀 Installation

### Step 1: Prerequisites

Ensure you have Python 3.9+ installed. Check with:
```bash
python --version
```

For GPU acceleration, install NVIDIA CUDA Toolkit and cuDNN (Windows/Linux).

### Step 2: Clone or Download

```bash
cd Desktop/MedicalScribeLocal
```

### Step 3: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `customtkinter` - Modern GUI framework
- `torch>=2.3.0` - PyTorch for AI model execution
- `torchaudio>=2.3.0` - Audio processing
- `pyannote.audio>=3.3.0` - Speaker diarization
- `speechbrain>=1.0.0` - Speaker embeddings
- `faster-whisper>=1.0.3` - Fast speech recognition
- `llama-cpp-python` - Local LLM inference
- `python-docx>=1.1.2` - Word document generation
- `sounddevice>=0.4.6` - Audio capture

### Step 5: HuggingFace Authentication

The application requires HuggingFace authentication for downloading Pyannote models:

```bash
huggingface-cli login
```

Enter your HuggingFace access token (requires accepting user conditions for pyannote models at hf.co/pyannote/speaker-diarization-community-1).

### Step 6: Download AI Models

Run the setup script to download and configure all required AI models:

```bash
python setup_models.py
```

**This downloads approximately 10 GB of models:**
- Pyannote speaker diarization model
- SpeechBrain ECAPA voice embedding model
- Whisper Large-V3 transcription model
- Llama 3.1 8B Instruct (Q4 quantized) for report generation

### Step 7: Verify Installation

Test the installation by running:

```bash
python main.py
```

The application should launch and display "Ready" in the status bar after models load.

---

## 🎯 Usage Guide

### Launching the Application

**Windows:**
- Double-click `run.bat`
- Or run: `python main.py`

**Linux/macOS:**
```bash
chmod +x build.sh
./build.sh
# or directly: python main.py
```

### Main Interface Features

#### Tab 1: Consultation Note (🩺)

**Real-time Consultation:**
1. Enter Patient Name (required)
2. Enter Doctor Name (optional, defaults to "Attending Physician")
3. Select Report Language (Medical English or Arabic)
4. Click "🎙 Start Consultation" to begin live recording
5. Conduct the consultation naturally
6. Click "⏹ Stop Consultation" when finished
7. Report generates automatically

**File Processing:**
1. Browse and select an audio file (WAV, MP3, M4A, FLAC)
2. Enable "⚡ Fast Mode" for quicker processing
3. Click "▶ Generate Report"
4. Report appears in the main window

**Report Actions:**
- **💬 View Dialogue Transcript**: See raw conversation with speaker labels
- **📄 Export Word Document**: Save as .docx file
- **📋 Copy Report**: Copy to clipboard
- Status bar shows processing progress

#### Tab 2: Doctor Voice ID (🎙)

**Voice Enrollment:**
1. Enter Physician Name
2. Browse to select a voice sample audio file (30+ seconds recommended)
3. Click "Enroll Voice Profile"
4. System creates acoustic embedding for automatic recognition
5. Enrolled profiles are stored in `doctor_profiles/` directory

**Benefits:**
- Automatic doctor identification in consultations
- Improved speaker diarization accuracy
- Multiple doctors can be enrolled

#### Tab 3: Patient Records (📁)

**Report Management:**
- View all generated consultation reports
- Open reports folder in file explorer
- Reports saved as JSON with timestamps
- Organized by patient name and date

### Supported Languages

**Transcription Languages:**
- English, Arabic, Hindi, Urdu, Tagalog, Malayalam, French
- Automatic language detection
- Code-switching support (mixed languages)

**Report Output Languages:**
- Medical English (Standard/HMC format)
- Arabic (العربية - medical report format)

---

## 🔧 Configuration

### Hardware Configuration

Edit `src/config.py` to adjust hardware settings:

```python
# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Speaker similarity threshold (0.0-1.0)
SIMILARITY_THRESHOLD = 0.70
```

### Directory Structure

Application automatically creates these directories:

```
MedicalScribeLocal/
├── models/              # AI models (~10 GB)
│   ├── whisper/        # Whisper STT models
│   ├── speechbrain/   # Speaker embedding models
│   ├── llm/           # Llama LLM models
│   └── hf_cache/      # HuggingFace cache
├── doctor_profiles/    # Voice enrollment profiles
├── output/            # Generated reports
├── temp/              # Temporary processing files
└── assets/            # Application icons
```

### Model Selection

To use different Whisper models, edit `src/pipeline.py`:

```python
self.whisper = WhisperModel(
    "base",  # Options: "tiny", "base", "small", "medium", "large-v3"
    device=config.DEVICE,
    compute_type=compute,
)
```

**Model Trade-offs:**
- `tiny`: Fastest, lower accuracy
- `base`: Good balance (default)
- `small`: Better accuracy, slower
- `medium`: High accuracy, slower
- `large-v3`: Best accuracy, slowest

---

## 🏗️ Building Executable

### Windows Build

```bash
# Build using PyInstaller
build.bat
```

This creates a standalone executable in `dist/MedicalScribe.exe`

### Linux/macOS Build

```bash
# Make build script executable
chmod +x build.sh

# Run build
./build.sh
```

### Build Requirements

Install PyInstaller first:
```bash
pip install pyinstaller
```

The build script includes all necessary models and dependencies for a standalone distribution.

---

## 🔬 Technical Details

### AI Models Used

**Speech-to-Text:**
- Model: Whisper Large-V3 (OpenAI)
- Function: Multilingual transcription with timestamps
- Size: ~3 GB

**Speaker Diarization:**
- Model: Pyannote Speaker Diarization Community v1
- Function: Speaker segmentation and turn detection
- Size: ~1.5 GB

**Speaker Embeddings:**
- Model: SpeechBrain ECAPA-VoxCeleb
- Function: Voice fingerprinting for identification
- Size: ~500 MB

**Report Generation:**
- Model: Llama 3.1 8B Instruct (Q4_K_M quantized)
- Function: Clinical report synthesis from transcripts
- Size: ~4.7 GB

### Processing Performance

**Typical Processing Times:**
- 5-minute consultation: ~30-60 seconds (GPU), ~2-3 minutes (CPU)
- 15-minute consultation: ~1-2 minutes (GPU), ~5-8 minutes (CPU)
- Real-time transcription: <2 second latency

**Memory Usage:**
- GPU VRAM: 4-6 GB recommended
- System RAM: 8-12 GB during processing

### Clinical NLP Features

**Semantic Disambiguation:**
- Clinical pattern recognition for role assignment
- Doctor vs Patient linguistic analysis
- Automatic correction of diarization errors

**Report Structure:**
- Chief Complaint (CC)
- History of Present Illness (HPI)
- Examination Findings
- Assessment & Diagnosis
- Diagnostic Tests Ordered
- Prescriptions & Medications
- Follow-up Plan
- Patient Instructions

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "CUDA out of memory"**
- Solution: Use CPU mode or reduce batch size in config
- Alternative: Use smaller Whisper model (`base` instead of `large-v3`)

**Issue: "Model loading failed"**
- Solution: Run `python setup_models.py` again
- Check internet connection for initial model download
- Verify HuggingFace authentication

**Issue: "No speech detected"**
- Solution: Check microphone permissions and audio input device
- Ensure audio file is not corrupted
- Verify audio format is supported

**Issue: "Speaker identification inaccurate"**
- Solution: Enroll doctor voice profile
- Ensure clear audio quality during enrollment
- Use longer enrollment samples (60+ seconds)

**Issue: "Import errors on startup"**
- Solution: Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Optimization

**For faster processing:**
- Use GPU acceleration (CUDA)
- Enable Fast Mode for file processing
- Use smaller Whisper model
- Close other applications to free memory

**For better accuracy:**
- Use larger Whisper model
- Ensure high-quality audio input
- Enroll doctor voice profiles
- Use appropriate sample rate (16 kHz)

---

## 📊 Project Structure

```
MedicalScribeLocal/
├── main.py                      # Application entry point
├── setup_models.py              # Model download script
├── requirements.txt             # Python dependencies
├── build.bat                    # Windows build script
├── build.sh                     # Linux/macOS build script
├── run.bat                      # Windows launch script
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── assets/                      # Application assets
│   └── README.txt               # Icon guidelines
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py               # Configuration and paths
│   ├── gui.py                  # Main GUI implementation
│   ├── pipeline.py             # Audio processing pipeline
│   ├── realtime_pipeline.py    # Real-time transcription
│   ├── report_generator.py     # Clinical report generation
│   └── semantic_disambiguator.py # Role assignment logic
├── doctor_profiles/            # Voice enrollment profiles
├── models/                     # AI models (auto-created)
├── output/                     # Generated reports (auto-created)
├── temp/                       # Temporary files (auto-created)
└── venv/                       # Virtual environment (auto-created)
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional language support
- Enhanced clinical templates
- Performance optimizations
- Additional export formats (PDF, EHR integration)
- Improved voice recognition accuracy
- Mobile application development

---

## 📝 License

[Specify your license here - e.g., MIT License]

This project is designed for healthcare professionals and researchers. Ensure compliance with local medical data regulations when using in production environments.

---

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech recognition model
- **Pyannote** for speaker diarization technology
- **SpeechBrain** for speaker embedding models
- **Meta** for Llama language models
- **HuggingFace** for model hosting and infrastructure
- **Hamad Medical Corporation** inspiration for clinical standards

---

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check troubleshooting section above
- Review documentation in code comments

---

## 🏥 Medical Disclaimer

This tool is designed to assist healthcare professionals with documentation. It does not provide medical advice, diagnosis, or treatment recommendations. All clinical decisions remain the responsibility of qualified healthcare providers. Generated reports should be reviewed and validated by medical professionals before use in patient care.

---

**Built with ❤️ for the Qatar healthcare community**
