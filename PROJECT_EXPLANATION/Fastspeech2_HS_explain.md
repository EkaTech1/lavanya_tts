# Fastspeech2_HS Directory Explanation

## Table of Contents
1. [Directory Structure](#directory-structure)
2. [Core Components](#core-components)
3. [Text Processing Pipeline](#text-processing-pipeline)
4. [Model Architecture](#model-architecture)
5. [Error Handling](#error-handling)
6. [Troubleshooting Guide](#troubleshooting-guide)

## Directory Structure

```
Fastspeech2_HS/
├── inference.py                    # Main inference script
├── text_preprocess_for_inference.py # Text preprocessing
├── phone_dict/                     # Phone dictionaries
│   ├── marathi
│   ├── hindi
│   └── ...
└── [language]/                     # Language models
    └── [gender]/                   # Gender-specific models
        └── model/
            ├── model.pth
            ├── config.yaml
            ├── feats_stats.npz
            ├── pitch_stats.npz
            └── energy_stats.npz
```

## Core Components

### 1. `inference.py`
Main script for text-to-speech conversion.

```python
# Key Functions
def load_fastspeech2_model(language, gender, device):
    """
    Loads the FastSpeech2 model for specific language and gender
    Example:
    model = load_fastspeech2_model("marathi", "male", "cuda")
    """

def text_synthesis(language, gender, text, vocoder, max_wav_value, device, alpha):
    """
    Converts text to speech
    Example:
    audio = text_synthesis("marathi", "male", "नमस्कार", vocoder, 32768, "cuda", 1.0)
    """
```

### 2. `text_preprocess_for_inference.py`
Handles text preprocessing and phoneme conversion.

```python
# Preprocessor Classes
class TTSPreprocessor:
    """
    Default preprocessor for most languages
    Handles:
    - Number conversion
    - Special characters
    - Punctuation
    """

class CharTextPreprocessor:
    """
    Special preprocessor for specific languages
    Used for: Urdu, Punjabi
    """

class TTSDurAlignPreprocessor:
    """
    Preprocessor with duration alignment
    Used for: Default fallback
    """
```

## Text Processing Pipeline

### 1. Text Normalization
```python
# Example: Number conversion in Marathi
Input: "१२३ विद्यार्थी"
Output: "एकशे तेविस विद्यार्थी"

# Example: Special character handling
Input: "नमस्कार!!!"
Output: "नमस्कार ।"
```

### 2. Phoneme Conversion
```python
# Example: Marathi phoneme mapping
Input: "नमस्कार"
Phonemes: ["n", "ə", "m", "ə", "s", "k", "aː", "r"]

# Example: Hindi phoneme mapping
Input: "नमस्ते"
Phonemes: ["n", "ə", "m", "ə", "s", "t", "eː"]
```

## Model Architecture

### 1. FastSpeech2 Model Configuration
```yaml
# config.yaml example
normalize_conf:
    stats_file: "feats_stats.npz"
pitch_normalize_conf:
    stats_file: "pitch_stats.npz"
energy_normalize_conf:
    stats_file: "energy_stats.npz"
```

### 2. Model Files Purpose
- `model.pth`: Trained model weights
- `feats_stats.npz`: Feature statistics for normalization
- `pitch_stats.npz`: Pitch information for natural speech
- `energy_stats.npz`: Energy information for volume control

## Error Handling

### 1. Model File Validation
```python
def check_model_exists(language, gender):
    """
    Validates model file existence
    Example:
    if not check_model_exists("english", "male"):
        raise FileNotFoundError("English male model not found")
    """
    model_path = os.path.join(
        language,
        gender,
        "model",
        "model.pth"
    )
    return os.path.exists(model_path)
```

### 2. Phone Dictionary Validation
```python
def check_phone_dict_exists(language):
    """
    Validates phone dictionary existence
    Example:
    if not check_phone_dict_exists("marathi"):
        raise FileNotFoundError("Marathi phone dictionary not found")
    """
    dict_path = os.path.join("phone_dict", language)
    return os.path.exists(dict_path)
```

## Troubleshooting Guide

### 1. Common Errors and Solutions

#### Missing Model Files
```
Error: FileNotFoundError: [Errno 2] No such file or directory: 'english/male/model/model.pth'
Solution: 
1. Check if model exists in correct path
2. Download/copy missing model files
3. Verify directory structure
```

#### NLTK Data Missing
```
Error: Resource averaged_perceptron_tagger_eng not found
Solution:
1. Run setup_nltk.py
2. Manually download NLTK data:
   import nltk
   nltk.download('averaged_perceptron_tagger')
```

#### Flash Attention Warning
```
Warning: Failed to import Flash Attention
Solution: 
- This is a non-critical warning
- System falls back to default attention mechanism
- No action needed
```

### 2. Language-Specific Issues

#### Hindi/Marathi
```python
# Working example
text = "नमस्कार"
language = "marathi"
gender = "male"
# Should work without issues
```

#### English
```python
# Requires additional setup
1. NLTK data installation
2. English phone dictionary
3. English model files
```

### 3. Performance Optimization

#### GPU Acceleration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

#### Memory Management
```python
# Clear GPU memory
torch.cuda.empty_cache()

# Use torch.no_grad for inference
with torch.no_grad():
    output = model(input_text)
```

## Usage Examples

### 1. Basic Text-to-Speech
```python
# Convert Marathi text to speech
text = "नमस्कार, कसे आहात?"
language = "marathi"
gender = "male"
alpha = 1.0  # Normal speed

audio = text_synthesis(language, gender, text, vocoder, MAX_WAV_VALUE, device, alpha)
```

### 2. Speed Modification
```python
# Faster speech
alpha = 1.5
fast_audio = text_synthesis(language, gender, text, vocoder, MAX_WAV_VALUE, device, alpha)

# Slower speech
alpha = 0.8
slow_audio = text_synthesis(language, gender, text, vocoder, MAX_WAV_VALUE, device, alpha)
```

### 3. Multiple Language Support
```python
# Hindi example
hindi_text = "नमस्ते"
hindi_audio = text_synthesis("hindi", "female", hindi_text, vocoder, MAX_WAV_VALUE, device, 1.0)

# Marathi example
marathi_text = "नमस्कार"
marathi_audio = text_synthesis("marathi", "male", marathi_text, vocoder, MAX_WAV_VALUE, device, 1.0)
```

## Best Practices

1. **Model Loading**
   - Load models only when needed
   - Keep models in memory if multiple conversions
   - Use appropriate device (GPU/CPU)

2. **Text Preprocessing**
   - Always validate input text
   - Handle special characters
   - Convert numbers to words
   - Check for language support

3. **Error Handling**
   - Validate model files before processing
   - Check phone dictionary existence
   - Handle missing NLTK data
   - Provide clear error messages

4. **Resource Management**
   - Clear GPU memory after large batches
   - Remove temporary audio files
   - Monitor memory usage
   - Use appropriate batch sizes 