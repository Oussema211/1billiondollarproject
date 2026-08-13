#!/bin/bash
set -e

echo "Cleaning old builds..."
rm -rf build dist *.spec

echo "Building executable..."
pyinstaller \
    --name "MedicalScribe" \
    --windowed \
    --onefile \
    --icon assets/icon.icns \
    --add-data "models:models" \
    --add-data "doctor_profiles:doctor_profiles" \
    --add-data "output:output" \
    --hidden-import torch \
    --hidden-import torchaudio \
    --hidden-import pyannote.audio \
    --hidden-import speechbrain \
    --hidden-import faster_whisper \
    --hidden-import llama_cpp \
    --hidden-import customtkinter \
    --collect-all pyannote.audio \
    --collect-all speechbrain \
    --collect-all faster_whisper \
    main.py

echo "Build complete. Check the dist/ folder."
