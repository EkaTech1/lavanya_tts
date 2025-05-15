# Pushing Fastspeech2_HS to GitHub: Process Documentation

## Initial State
- Repository: https://github.com/EkaTechKB/lavanya_tts.git
- Issue: Fastspeech2_HS directory was not being pushed to GitHub
- Root Cause: Directory was a Git submodule and large files were being blocked by `.gitignore`

## Steps Taken

### 1. Identified Issues
- Fastspeech2_HS was configured as a Git submodule
- Large model files (`.pth`, `.pt`, `.ckpt`, `.bin`) were being ignored
- Several files exceeded GitHub's recommended 50MB file size limit

### 2. Modified .gitignore
```diff
- # Large files and models
- *.pth
- *.pt
- *.ckpt
- *.bin
+ # Note: We are including model files (.pth, .pt, .ckpt, .bin) as they are essential for the TTS system
```

### 3. Removed Submodule Structure
```bash
git rm --cached Fastspeech2_HS
rm -rf Fastspeech2_HS/.git
```

### 4. Set Up Git LFS
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

### 5. Added and Committed Files
```bash
git add Fastspeech2_HS/
git commit -m "Add Fastspeech2_HS directory directly with model files"
git push origin main
```

## Errors Encountered and Solutions

### 1. Submodule Error
```
fatal: Pathspec 'Fastspeech2_HS/hindi/male/model/model.pth' is in submodule 'Fastspeech2_HS'
```
**Solution**: Removed the submodule structure and added files directly to the main repository.

### 2. Large File Warnings
```
warning: File Fastspeech2_HS/vocoder/female/aryan/hifigan/generator is 53.24 MB
warning: File Fastspeech2_HS/vocoder/male/aryan/hifigan/generator is 53.24 MB
warning: File Fastspeech2_HS/vocoder/male/dravidian/hifigan/generator is 53.20 MB
```
**Solution**: Implemented Git LFS to handle large files properly.

## Final Results
- Successfully pushed 520 files to the repository
- Large files were handled by Git LFS
- All model files and dependencies are now available in the repository
- Total size of pushed data: ~175.74 MiB

## Large Files Tracked
1. Model files (`.pth`)
   - `hindi/male/model/model.pth`
   - `marathi/male/model/model.pth`
   - `sarjerao_ekalipi/male/model/model.pth`
   - `sarjerao_ekalipi/female/model/model.pth`

2. Generator files
   - `vocoder/female/aryan/hifigan/generator` (53.24 MB)
   - `vocoder/male/aryan/hifigan/generator` (53.24 MB)
   - `vocoder/male/dravidian/hifigan/generator` (53.20 MB)

## Notes for Future Reference
1. Always use Git LFS for files larger than 50MB
2. Keep model files in the repository despite size for project completeness
3. Monitor `.gitignore` to ensure essential files aren't accidentally excluded
4. Use proper commit messages for better tracking
5. Handle submodules carefully to avoid repository structure issues 

## Adding New Language Models

### Steps to Push New model.pth Files

1. Verify Git LFS is tracking .pth files:
```bash
# Check if .pth files are being tracked
git lfs track | grep ".pth"

# If not tracked, set up tracking
git lfs track "*.pth"
git add .gitattributes
```

2. Add the new model file:
```bash
# Example for adding a new language model
git add Fastspeech2_HS/[language]/[gender]/model/model.pth

# For multiple files
git add Fastspeech2_HS/**/model/model.pth
```

3. Commit and push:
```bash
# Commit with descriptive message
git commit -m "Add [language] [gender] TTS model"

# Push to GitHub
git push origin main
```

### Example for Adding Multiple Language Models
```bash
# Add all new model files
git add Fastspeech2_HS/bengali/*/model/model.pth
git add Fastspeech2_HS/gujarati/*/model/model.pth

# Commit changes
git commit -m "Add Bengali and Gujarati TTS models for both genders"

# Push to repository
git push origin main
```

### Troubleshooting New Model Pushes

1. If you get size warnings:
   - GitHub LFS should handle it automatically since we're tracking .pth files
   - No additional setup needed for new models

2. If push fails with LFS errors:
```bash
# Verify LFS installation
git lfs install

# Retry failed LFS push
git lfs push --all origin main
```

3. If files are not being tracked by LFS:
```bash
# Force LFS to track existing .pth files
git lfs migrate import --include="*.pth"
git push --force origin main
```

### Best Practices for New Models
1. Always verify file size before pushing (large models should use LFS)
2. Keep consistent directory structure: `Fastspeech2_HS/[language]/[gender]/model/`
3. Include all necessary configuration files along with model.pth
4. Add appropriate documentation for new language support
5. Test the model locally before pushing 

## Runtime Troubleshooting Guide

### Common Runtime Errors

#### 1. JSON Response Error
```
Error generating speech: Failed to execute 'json' on 'Response': Unexpected end of JSON input
```
**Causes**:
- Server timeout while processing large text
- Memory issues with model loading
- Incomplete response from the TTS service

**Solutions**:
1. Modify Gunicorn configuration:
```bash
# Add to start.sh or command line
gunicorn --timeout 300 --workers 1 --threads 4 app:app
```

2. Update Flask configuration in app.py:
```python
app.config['TIMEOUT'] = 300  # 5 minutes timeout
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-content-length
```

#### 2. Worker Timeout Error
```
[CRITICAL] WORKER TIMEOUT (pid:109)
[INFO] Worker exiting (pid: 109)
[ERROR] Worker (pid:109) exited with code 1
```
**Causes**:
- Default worker timeout (30s) is too short for model inference
- Resource constraints (CPU/RAM)
- Multiple concurrent requests

**Solutions**:
1. Increase worker timeout:
```bash
# In your start script or command
gunicorn --timeout 300 --graceful-timeout 300 app:app
```

2. Adjust worker configuration:
```bash
# For CPU-intensive tasks
gunicorn --workers 1 --threads 4 --worker-class gthread app:app

# For memory-intensive tasks
gunicorn --workers 1 --worker-class sync app:app
```

3. Add memory management:
```python
# In app.py
import gc

@app.after_request
def cleanup(response):
    gc.collect()
    return response
```

#### 3. Port Binding Issues
```
Detected service running on port 10000
```
**Solutions**:
1. Specify port explicitly:
```bash
# In start.sh
export PORT=10000
gunicorn --bind 0.0.0.0:$PORT app:app
```

2. Add port configuration in app.py:
```python
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
```

### Best Practices for Deployment
1. Memory Management:
   - Clear GPU cache between inferences
   - Implement garbage collection
   - Monitor memory usage

2. Request Handling:
   - Implement request queuing
   - Add timeout handlers
   - Include proper error responses

3. Configuration Settings:
```python
# Recommended app.py configurations
app.config.update(
    TIMEOUT = 300,
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024,
    JSON_SORT_KEYS = False,
    JSONIFY_PRETTYPRINT_REGULAR = False
)
```

4. Monitoring:
   - Add logging for model loading times
   - Track inference times
   - Monitor memory usage
   - Log all errors with stack traces

### Quick Fix for Current Error
1. Stop the current service
2. Update your start command:
```bash
gunicorn --timeout 300 --workers 1 --threads 4 --worker-class gthread app:app
```
3. Add error handling in app.py:
```python
@app.errorhandler(500)
def handle_timeout(e):
    return jsonify({"error": "Processing timeout. Please try with shorter text or contact support."}), 500
``` 