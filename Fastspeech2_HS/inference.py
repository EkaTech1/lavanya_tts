import sys
import os
import requests
from pathlib import Path
#replace the path with your hifigan path to import Generator from models.py 
sys.path.append("hifigan")
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from env import AttrDict
import json
import yaml
from text_preprocess_for_inference import TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor
import gc
import concurrent.futures
import numpy as np
import re

SAMPLING_RATE = 22050
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 100))  # Process 100 words at a time
MAX_TEXT_LENGTH = int(os.environ.get('MAX_TEXT_LENGTH', 2000))  # Maximum 2000 characters

# Global cache for models
model_cache = {}
vocoder_cache = {}

# Model storage configuration
MODEL_BASE_URL = os.environ.get('MODEL_BASE_URL', 'https://your-storage-url.com/models')
LOCAL_MODEL_DIR = os.environ.get('LOCAL_MODEL_DIR', 'models_cache')

# Sentence boundary markers for different languages
SENTENCE_MARKERS = {
    'english': ['.', '!', '?', ';'],
    'hindi': ['।', '॥'],
    'punjabi': ['।', '॥'],
    'tamil': ['.|'],
    'telugu': ['.|'],
    'kannada': ['.|'],
    'malayalam': ['.|'],
    'gujarati': ['.|'],
    'bengali': ['।'],
    'odia': ['।'],
    'assamese': ['।'],
    'manipuri': ['।'],
    'bodo': ['।'],
    'rajasthani': ['।'],
    'urdu': ['۔'],
    'default': ['.', '।', '॥', '|', '۔', '!', '?']
}

def get_sentence_markers(language):
    """Get sentence boundary markers for a specific language."""
    return SENTENCE_MARKERS.get(language.lower(), SENTENCE_MARKERS['default'])

def split_into_chunks(text, language='default', max_length=MAX_TEXT_LENGTH, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks while preserving sentence boundaries and context.
    """
    # First check if text exceeds maximum length
    if len(text) > max_length:
        raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed length ({max_length})")
    
    # Get sentence markers for the language
    markers = get_sentence_markers(language)
    
    # Create regex pattern for sentence boundaries
    pattern = '|'.join(map(re.escape, markers))
    
    # Split text into sentences
    sentences = re.split(f'({pattern})', text)
    
    # Recombine sentences with their punctuation
    sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2] + [''] * (len(sentences[::2]) - len(sentences[1::2])))]
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If a single sentence is longer than chunk_size, split it
        if sentence_word_count > chunk_size:
            # If there are words in current_chunk, add them as a chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split long sentence into chunks
            for i in range(0, sentence_word_count, chunk_size):
                chunk = sentence_words[i:i + chunk_size]
                chunks.append(' '.join(chunk))
        
        # If adding this sentence would exceed chunk_size
        elif current_word_count + sentence_word_count > chunk_size:
            # Add current chunk and start new one
            chunks.append(' '.join(current_chunk))
            current_chunk = sentence_words
            current_word_count = sentence_word_count
        else:
            # Add sentence to current chunk
            current_chunk.extend(sentence_words)
            current_word_count += sentence_word_count
    
    # Add any remaining words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def download_if_needed(remote_path, local_path):
    """Download a file if it doesn't exist locally."""
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"{MODEL_BASE_URL}/{remote_path}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

def get_model_path(language, gender, model_type='model'):
    """Get the path for a model file, downloading it if necessary."""
    remote_path = f"{language}/{gender}/model/{model_type}.pth"
    local_path = os.path.join(LOCAL_MODEL_DIR, remote_path)
    
    try:
        download_if_needed(remote_path, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        # Fallback to local path in Fastspeech2_HS
        return os.path.join("Fastspeech2_HS", remote_path)

def load_hifigan_vocoder(language, gender, device):
    """Load HiFi-GAN vocoder."""
    cache_key = f"{language}_{gender}"
    if cache_key in vocoder_cache:
        return vocoder_cache[cache_key]

    try:
        # Determine if language is Aryan or Dravidian
        aryan_languages = ['hindi', 'marathi', 'punjabi', 'gujarati', 'bengali', 'odia', 'assamese', 'manipuri', 'bodo', 'rajasthani', 'urdu']
        language_family = 'aryan' if language.lower() in aryan_languages else 'dravidian'
        
        # Use local paths
        vocoder_config = os.path.join('vocoder', gender, language_family, 'hifigan', 'config.json')
        vocoder_generator = os.path.join('vocoder', gender, language_family, 'hifigan', 'generator')
        
        with open(vocoder_config, 'r') as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
            
        device = torch.device(device)
        generator = Generator(h).to(device)
        state_dict_g = torch.load(vocoder_generator, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()

        vocoder_cache[cache_key] = generator
        return generator
    except Exception as e:
        print(f"Error loading vocoder: {str(e)}")
        raise

def load_fastspeech2_model(language, gender, device):
    """Load FastSpeech2 model."""
    cache_key = f"{language}_{gender}"
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    try:
        model_dir = os.path.join(language, gender, 'model')
        config_path = os.path.join(model_dir, 'config.yaml')
        
        with open(config_path, "r") as file:      
            config = yaml.safe_load(file)
    
        current_working_directory = os.getcwd()
        feat = os.path.join(model_dir, 'feats_stats.npz')
        pitch = os.path.join(model_dir, 'pitch_stats.npz')
        energy = os.path.join(model_dir, 'energy_stats.npz')
    
        feat_path = os.path.join(current_working_directory, feat)
        pitch_path = os.path.join(current_working_directory, pitch)
        energy_path = os.path.join(current_working_directory, energy)
    
        config["normalize_conf"]["stats_file"] = feat_path
        config["pitch_normalize_conf"]["stats_file"] = pitch_path
        config["energy_normalize_conf"]["stats_file"] = energy_path
        
        with open(config_path, "w") as file:
            yaml.dump(config, file)
    
        tts_model = os.path.join(model_dir, 'model.pth')
        
        model = Text2Speech(train_config=config_path, model_file=tts_model, device=device)
        model_cache[cache_key] = model
        return model
    except Exception as e:
        print(f"Error loading FastSpeech2 model: {str(e)}")
        raise

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device, alpha):
    try:
        with torch.no_grad():
            # Load the FastSpeech2 model for the specified language and gender
            model = load_fastspeech2_model(language, gender, device)
           
            # Generate mel-spectrograms from the input text using the FastSpeech2 model
            out = model(sample_text, decode_conf={"alpha": alpha})
            x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
            x = x.to(device)
            
            # Use the HiFi-GAN vocoder to convert mel-spectrograms to raw audio waveforms
            y_g_hat = vocoder(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            
            # Clear some memory
            del out, x, y_g_hat
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            return audio
    except Exception as e:
        print(f"Error in text synthesis: {str(e)}")
        raise

def process_chunk(args):
    sample_text, language, gender, vocoder, device, alpha = args
    try:
        phone_dictionary = {}
        preprocessor = get_preprocessor(language)
        preprocessed_text, _ = preprocessor.preprocess(sample_text, language, gender, phone_dictionary)
        preprocessed_text = " ".join(preprocessed_text)
        return text_synthesis(language, gender, preprocessed_text, vocoder, MAX_WAV_VALUE, device, alpha)
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        raise

def get_preprocessor(language):
    if language in ["urdu", "punjabi"]:
        return CharTextPreprocessor()
    elif language == "english":
        return TTSPreprocessor()
    return TTSDurAlignPreprocessor()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-Speech Inference")
    parser.add_argument("--language", type=str, required=True, help="Language (e.g., hindi)")
    parser.add_argument("--gender", type=str, required=True, help="Gender (e.g., female)")
    parser.add_argument("--sample_text", type=str, required=True, help="Text to be synthesized")
    parser.add_argument("--output_file", type=str, default="", help="Output WAV file path")
    parser.add_argument("--alpha", type=float, default=1, help="Alpha Parameter")

    args = parser.parse_args()

    try:
        # Set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the HiFi-GAN vocoder with dynamic language and gender
        vocoder = load_hifigan_vocoder(args.language, args.gender, device)
        
        chunks = split_into_chunks(args.sample_text, args.language)
        audio_arr = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            chunk_args = [(chunk, args.language, args.gender, vocoder, device, args.alpha) for chunk in chunks]
            results = list(executor.map(process_chunk, chunk_args))
            audio_arr.extend(results)
        
        if audio_arr:
            result_array = np.concatenate(audio_arr, axis=0)
            output_file = args.output_file or f"{args.language}_{args.gender}_output.wav"
            write(output_file, SAMPLING_RATE, result_array)
            print(f"Successfully generated audio file: {output_file}")
        else:
            print("No audio generated")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)
    finally:
        # Clear caches
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
