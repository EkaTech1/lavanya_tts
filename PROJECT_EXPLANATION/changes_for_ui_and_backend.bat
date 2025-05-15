@echo off
echo ====================================
echo UI and Backend Changes Documentation
echo ====================================

echo 1. Environment Setup:
echo - Created conda environment with Python 3.10
echo conda create -n tts_env python=3.10
echo conda activate tts_env

echo 2. Required Package Installation:
echo - Installing required packages
echo pip install torch torchaudio espnet pandas num2words indic-unified-parser flask nltk

echo 3. NLTK Data Setup:
echo - Downloaded required NLTK data
echo python setup_nltk.py

echo 4. Directory Structure:
echo - Created static/audio directory for output files
echo mkdir static
echo mkdir static/audio

echo 5. File Changes Made:
echo a) Created app.py:
echo   - Flask application setup
echo   - Routes for / and /synthesize
echo   - Integration with inference.py
echo   - Error handling and response formatting

echo b) Created setup_nltk.py:
echo   - NLTK data download script
echo   - Required for English TTS

echo c) Created templates/index.html:
echo   - Modern UI with Bootstrap
echo   - Language and gender selection
echo   - Speech speed control
echo   - Audio playback and download

echo 6. Working Languages:
echo - Marathi (male/female)
echo - Hindi (male/female)
echo - English (requires model.pth in english/male/model/)

echo 7. Known Issues:
echo - Flash Attention warning (can be ignored)
echo - English TTS requires model file in correct path

echo 8. Running the Application:
echo conda activate tts_env
echo python app.py
echo Access at http://localhost:4005

echo ====================================
echo End of Documentation
echo ==================================== 