# Arabic Speech-to-Text with Speaker Diarization

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/HARISHSENTHIL/Arabic_model.git
   cd Arabic_model
   ```

2. **Set up environment**
   ```bash
   # Create an environment file
   nano .env
   
   # Edit .env and add your Hugging Face token
   # Get your token from: https://huggingface.co/settings/tokens
   
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

3. **Build and run**
   ```bash
   sudo docker-compose up --build -d
   ```
4. **Test the service**
   ```bash
   # The API will be available at http://localhost:8000/docs
   
   # Example using curl:
   curl -X POST "http://localhost:8000/transcribe-diarize" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/audio.wav"
   ```
