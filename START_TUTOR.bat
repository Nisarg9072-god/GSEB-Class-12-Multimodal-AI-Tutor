@echo off
title GSEB Class 12 AI Tutor
echo.
echo  ================================================
echo   GSEB Class 12 AI Tutor - Starting...
echo  ================================================
echo.

cd /d "%~dp0"

echo  Opening browser at http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.

.venv\Scripts\streamlit.exe run app.py --server.port 8501

pause
