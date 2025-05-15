# Multilingual Text-to-Speech System

A web-based Text-to-Speech (TTS) system supporting multiple Indian languages using FastSpeech2 and HiFi-GAN.

## Features

- Web interface for easy text-to-speech conversion
- Supports 16 languages:
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
- Male and female voice options
- Adjustable speech speed (0.5x to 2.0x)
- Sample texts for each language
- Audio playback and download functionality

## Prerequisites

- Python 3.10
- Conda environment manager
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Create and activate a conda environment:
```bash
conda create -n tts_env python=3.10
conda activate tts_env
```

2. Install required packages:
```bash
pip install torch torchaudio espnet pandas num2words indic-unified-parser flask nltk
```


## Directory Structure

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

## Usage
(pkill -f "python app.py" && conda activate tts_env && python app.py)
this will again run flask app:)
1. Activate the conda environment:
```bash
conda activate tts_env
```

2. Start the Flask application:
```bash
python app.py
```
3. Access the web interface at `http://localhost:4005`

4. Select:
   - Language
   - Gender (male/female)
   - Speech speed (0.5x to 2.0x)
   - Enter text or use sample text
   - Click "Generate Speech"

## Model Requirements

Each language requires:
1. Phone dictionary in `Fastspeech2_HS/phone_dict/[language]`
2. Model file in `Fastspeech2_HS/[language]/[gender]/model/model.pth`

## Known Issues

1. Flash Attention warning can be safely ignored:
   ```
   Failed to import Flash Attention, using ESPnet default: No module named 'flash_attn'
   ```

2. Some languages may require additional model files. Check the model path:
   ```
   Fastspeech2_HS/[language]/[gender]/model/model.pth
   ```

## Recent Updates

1. Added web interface with Bootstrap UI
2. Added sample texts for all languages
3. Improved error handling for missing models
4. Added speech speed control
5. Added audio download functionality
6. Fixed directory path issues for model loading
7. Added proper error messages for missing models/dictionaries

## Contributing

Feel free to submit issues and enhancement requests.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- ESPnet for the TTS models
- HiFi-GAN for the vocoder
- All contributors to the phone dictionaries and language models 
