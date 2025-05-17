# Multilingual Text-to-Speech (TTS) System Setup Guide

This guide will help you set up the Multilingual TTS system with FastSpeech2 and HiFi-GAN support.

## System Requirements

- Operating System: Windows/Linux/MacOS
- Python version: 3.10 (recommended)
- GPU: CUDA-compatible GPU (optional, for faster processing)
- Storage: At least 10GB free space (for models and dependencies)

## Step-by-Step Setup Guide

### 1. Set up Python Environment

First, install Conda if you haven't already. Download from: https://docs.conda.io/en/latest/miniconda.html

```bash
# Create a new conda environment
conda create -n tts_env python=3.10
conda activate tts_env
```

### 2. Install Core Dependencies

```bash
# Install PyTorch and TorchAudio
conda install pytorch torchaudio -c pytorch

# Install basic requirements
pip install flask==3.1.0
pip install numpy>=2.2.5
pip install scipy>=1.15.3
pip install PyYAML>=6.0.2
```

### 3. Install TTS-Specific Dependencies

```bash
# Install ESPnet and related packages
pip install espnet
pip install espnet2
pip install soundfile

# Install language processing tools
pip install indic-num2words
pip install nltk
pip install indic-unified-parser
pip install pandas

# Install utilities
pip install matplotlib
pip install requests
pip install gunicorn  # For production deployment
```

### 4. Download and Setup Project

```bash
# Clone the repository (if using git)
git clone [your-repository-url]
cd [project-directory]

# Or download and extract the project files to your desired location
```

### 5. Directory Structure Setup

Ensure you have the following directory structure:
```
.
├── app.py                  # Flask application
├── Fastspeech2_HS/        # Core TTS engine
│   ├── inference.py       # TTS inference script
│   ├── phone_dict/       # Phone dictionaries for all languages
│   └── [language]/       # Language-specific models
├── static/
│   └── audio/            # Generated audio files
└── templates/
    └── index.html        # Web interface
```

### 6. Model Setup

1. Create necessary directories:
```bash
mkdir -p static/audio
```

2. Ensure model files are in place:
- Phone dictionaries should be in `Fastspeech2_HS/phone_dict/[language]`
- Model files should be in `Fastspeech2_HS/[language]/[gender]/model/model.pth`

### 7. Running the Application

```bash
# Activate the environment (if not already activated)
conda activate tts_env

# Start the Flask application
python app.py
```

The application will be available at: `http://localhost:4005`

### 8. Verify Installation

1. Open your web browser and navigate to `http://localhost:4005`
2. Select a language from the dropdown
3. Choose gender and speed options
4. Enter some text
5. Click "Generate Speech" to test

## Supported Languages

- Hindi
- Marathi
- Punjabi
- Tamil
- Telugu
- Kannada
- Malayalam
- Gujarati
- Bengali
- Odia
- Assamese
- Manipuri
- Bodo
- Rajasthani
- Urdu
- English

## Troubleshooting

1. If you see Flash Attention warning:
   ```
   Failed to import Flash Attention, using ESPnet default: No module named 'flash_attn'
   ```
   This is normal and can be safely ignored.

2. If you encounter CUDA errors:
   - Ensure you have CUDA installed if using GPU
   - Try running on CPU by setting appropriate environment variables

3. If models are not loading:
   - Check if model files are in correct directories
   - Verify file permissions
   - Check console for specific error messages

4. For package conflicts:
   ```bash
   # Create a fresh environment
   conda deactivate
   conda env remove -n tts_env
   conda create -n tts_env python=3.10
   conda activate tts_env
   ```
   Then follow the installation steps again.

## Quick Commands Reference

```bash
# Start application
conda activate tts_env && python app.py

# Kill existing process and restart (if needed)
pkill -f "python app.py" && conda activate tts_env && python app.py

# Check installed packages
pip list
```

## Additional Notes

- Keep your conda environment activated while working with the project
- Regular updates to dependencies might be required for security patches
- Backup your model files before any major updates
- For production deployment, consider using gunicorn with Flask

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your Python and package versions
3. Ensure all model files are in place
4. Check system requirements are met 

# Lavanya TTS - Multilingual Text-to-Speech System

## Deployment on Render

### Prerequisites
- A Render account
- Git repository with your code

### Steps to Deploy

1. Fork or clone this repository
2. Connect your GitHub account to Render
3. Create a new Web Service on Render
4. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300`
   - Python Version: 3.10.0
   - Environment: Python
   - Region: Choose the closest to your users

### Environment Variables
No additional environment variables are required for basic setup.

### Important Notes
1. The application uses disk storage for audio files
2. Audio files are automatically cleaned up (keeping only the last 50 files)
3. Maximum request timeout is set to 300 seconds
4. The application uses 1 worker with 4 threads for optimal performance

### Troubleshooting
1. If you see "Error generating speech":
   - Check the logs in Render dashboard
   - Ensure all model files are properly uploaded
   - Verify phone dictionaries are present

2. If the application is slow:
   - Consider upgrading to a higher tier on Render
   - Reduce the number of concurrent users
   - Use shorter text inputs

3. If you get disk space errors:
   - The cleanup routine should handle this automatically
   - Consider increasing disk size in render.yaml
   - Monitor disk usage in Render dashboard

### Monitoring
- Use Render's built-in logging to monitor the application
- Check application logs for detailed error messages
- Monitor disk usage regularly

### Support
For issues and questions:
1. Check the troubleshooting section
2. Review Render logs
3. Verify model files are present
4. Contact support with log details 