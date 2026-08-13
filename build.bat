@echo off
echo Cleaning old builds...
rmdir /s /q build dist 2>nul
del /f /q *.spec 2>nul

echo Building executable...
pyinstaller ^
    --name "MedicalScribe" ^
    --windowed ^
    --onefile ^
    --icon assets\icon.ico ^
    --add-data "models;models" ^
    --add-data "doctor_profiles;doctor_profiles" ^
    --add-data "output;output" ^
    --hidden-import torch ^
    --hidden-import torchaudio ^
    --hidden-import pyannote.audio ^
    --hidden-import speechbrain ^
    --hidden-import faster_whisper ^
    --hidden-import llama_cpp ^
    --hidden-import customtkinter ^
    --collect-all pyannote.audio ^
    --collect-all speechbrain ^
    --collect-all faster_whisper ^
    main.py

echo.
echo Build complete. Check the dist\ folder.
pause
