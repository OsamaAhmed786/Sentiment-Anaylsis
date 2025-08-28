# Lightweight Python image
FROM python:3.10-slim

# System deps: ffmpeg (for Whisper), git + g++ (some wheels may need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
 && rm -rf /var/lib/apt/lists/*

# For faster HF model caching inside the container
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Optional: choose Whisper size via env (tiny/base/small/medium)
# You can override this in Space settings too.
ENV WHISPER_MODEL=base

# Copy files
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY README.md .  

# Hugging Face Spaces expects the app to listen on port 7860
ENV PORT=7860

# (Optional) Pre-download models on build to reduce cold-start
# Comment out if build time becomes too long
# RUN python -c "import nltk; \
#     import whisper; \
#     from nltk.sentiment.vader import SentimentIntensityAnalyzer as _; \
#     import nltk; \
#     nltk.download('vader_lexicon'); \
#     whisper.load_model('${WHISPER_MODEL}')"

# Start FastAPI with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
