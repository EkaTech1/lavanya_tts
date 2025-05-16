from flask import Flask, render_template, request, send_file, jsonify
import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'static/audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def check_model_exists(language, gender):
    """Check if the model file exists for the given language and gender."""
    model_path = os.path.join('Fastspeech2_HS', language, gender, 'model', 'model.pth')
    
    # Determine if language is Aryan or Dravidian
    aryan_languages = ['hindi', 'marathi', 'punjabi', 'gujarati', 'bengali', 'odia', 'assamese', 'manipuri', 'bodo', 'rajasthani', 'urdu']
    language_family = 'aryan' if language.lower() in aryan_languages else 'dravidian'
    
    vocoder_path = os.path.join('Fastspeech2_HS', 'vocoder', gender, language_family, 'hifigan', 'generator')
    
    model_exists = os.path.exists(model_path)
    vocoder_exists = os.path.exists(vocoder_path)
    
    if not model_exists:
        logger.error(f"Model not found at path: {model_path}")
    if not vocoder_exists:
        logger.error(f"Vocoder not found at path: {vocoder_path}")
        
    # Currently only these combinations are supported
    supported_models = {
        'hindi': ['male'],
        'marathi': ['male'],
        'sarjerao_ekalipi': ['male', 'female']
    }
    
    is_supported = language in supported_models and gender in supported_models[language]
    
    if not is_supported:
        logger.error(f"Language-gender combination {language}-{gender} is not yet supported. Available models: {supported_models}")
        
    return model_exists and vocoder_exists and is_supported

def check_phone_dict_exists(language):
    """Check if the phone dictionary exists for the given language."""
    dict_path = os.path.join('Fastspeech2_HS', 'phone_dict', language)
    exists = os.path.exists(dict_path) and os.path.getsize(dict_path) > 0
    if not exists:
        logger.error(f"Phone dictionary not found at path: {dict_path}")
    return exists

def cleanup_old_files(max_files=50):
    """Clean up old audio files to prevent disk space issues."""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        wav_files = [f for f in files if f.endswith('.wav')]
        if len(wav_files) > max_files:
            wav_files.sort(key=lambda x: os.path.getmtime(
                os.path.join(app.config['UPLOAD_FOLDER'], x)))
            for f in wav_files[:-max_files]:
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                except Exception as e:
                    logger.error(f"Error removing file {f}: {str(e)}")
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        logger.info("Starting speech synthesis request")
        
        # Get form data
        text = request.form['text']
        language = request.form['language']
        gender = request.form['gender']
        alpha = float(request.form.get('alpha', 1.0))
        
        logger.info(f"Received request - Language: {language}, Gender: {gender}, Text length: {len(text)}")
        
        # Input validation
        if not text or len(text.strip()) == 0:
            logger.error("Empty text input received")
            return jsonify({
                'status': 'error',
                'message': 'Text input is required.'
            }), 400
        
        # Check if model and dictionary exist
        logger.info("Checking model existence...")
        if not check_model_exists(language, gender):
            supported_models = {
                'hindi': ['male'],
                'marathi': ['male'],
                'sarjerao_ekalipi': ['male', 'female']
            }
            logger.error(f"Model not available for {language} ({gender})")
            return jsonify({
                'status': 'error',
                'message': f'TTS model for {language} ({gender}) is not available. Currently supported models: {supported_models}'
            }), 400
            
        logger.info("Checking phone dictionary...")
        if not check_phone_dict_exists(language):
            logger.error(f"Phone dictionary not available for {language}")
            return jsonify({
                'status': 'error',
                'message': f'Phone dictionary for {language} is not available.'
            }), 400
        
        # Clean up old files before generating new ones
        logger.info("Cleaning up old files...")
        cleanup_old_files()
        
        # Generate output filename with timestamp
        timestamp = str(int(os.path.getmtime(__file__)))
        filename = f'output_{language}_{gender}_{timestamp}.wav'
        output_file = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)
        
        logger.info(f"Will generate output to: {output_file}")
        
        # Get the absolute path to inference.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        inference_dir = os.path.join(current_dir, 'Fastspeech2_HS')
        
        logger.info(f"Running inference from directory: {inference_dir}")
        
        # Run inference.py with the provided parameters
        cmd = [
            sys.executable,  # Use the same Python interpreter
            'inference.py',
            '--sample_text', text,
            '--language', language,
            '--gender', gender,
            '--alpha', str(alpha),
            '--output_file', output_file
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the command from the Fastspeech2_HS directory
        try:
            process = subprocess.run(
                cmd,
                check=True,
                cwd=inference_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            # Log the process output
            if process.stdout:
                logger.info(f"Process output: {process.stdout}")
            if process.stderr:
                logger.warning(f"Process warnings: {process.stderr}")
            
            # Check if the output file was actually created
            if not os.path.exists(output_file):
                logger.error("Output file was not created")
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to generate audio file. Please try again.'
                }), 500
                
            # Check if the file size is valid (not empty)
            if os.path.getsize(output_file) == 0:
                logger.error("Generated audio file is empty")
                os.remove(output_file)  # Clean up empty file
                return jsonify({
                    'status': 'error',
                    'message': 'Generated audio file is empty. Please try again.'
                }), 500
            
            logger.info(f"Successfully generated audio file: {filename}")
            
            # Return success response
            return jsonify({
                'status': 'success',
                'audio_path': f'/static/audio/{filename}'
            })
            
        except subprocess.TimeoutExpired:
            logger.error("TTS generation timed out")
            return jsonify({
                'status': 'error',
                'message': 'Speech generation is taking too long. Please try with shorter text.'
            }), 500
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            logger.error(f"TTS generation process error: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': f'Speech generation failed: {error_msg}'
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in synthesize: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again later.'
    }), 500

@app.errorhandler(404)
def handle_404_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found.'
    }), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4005))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug) 