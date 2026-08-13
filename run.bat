@echo off
title Medical Scribe Local
cd /d "C:\Users\USER\Desktop\MedicalScribeLocal"
set PYTHONPATH=.
set PYTHONIOENCODING=utf-8
echo Launching Medical Scribe App...
".\venv\Scripts\python.exe" main.py
if errorlevel 1 pause
