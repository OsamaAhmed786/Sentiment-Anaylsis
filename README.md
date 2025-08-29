# 🎙️ AI Sentiment, Emotion & Tone Analyzer  

An **AI-powered sentiment, emotion, and tone analysis system** that works with both **text and audio inputs**.  
The project combines **multiple NLP models** to provide deep insights into human communication.  

---

## ✨ Features  
- 🔹 **Text & Audio Support** → Analyze both written text and uploaded voice recordings.  
- 🔹 **Sentiment Analysis (VADER)** → Detects polarity (Positive, Negative, Neutral).  
- 🔹 **Emotion Detection (Hugging Face Model)** → Multilingual emotion classifier.  
- 🔹 **Tone Classification (FinBERT)** → Identifies Positive, Negative, or Neutral tones.  
- 🔹 **JSON Output** → Clean, normalized, developer-friendly results.  
- 🔹 **Rounded Scores** → Confidence values rounded to 2 decimal places.  

---

## ⚙️ Tech Stack  
- **Python 3.9+**  
- [NLTK (VADER)](https://www.nltk.org/) – Rule-based sentiment analyzer  
- [Transformers (Hugging Face)](https://huggingface.co/) – Pretrained models for sentiment & emotion  
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) – Speech-to-text transcription  
- [Torch](https://pytorch.org/) – Model inference  

---

## 🚀 Installation  

1. Clone the repository  
   ```
   git clone https://github.com/OsamaAhmed786/Sentiment-Anaylsis.git
   cd sentiment-analyzer

2. Create & activate virtual environment (recommended)
    ```
    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

3. Install dependencies
    ```  
    pip install -r requirements.txt


4. (Optional) Download NLTK data manually
    ```
    python -m nltk.downloader vader_lexicon -d ./nltk_data


## 📊 Usage

🔹 1. Analyze Text

    from app import text_to_json
    
    text = "This is the worst experience I've ever had. Completely disappointed."
    result = text_to_json(text)
    print(result)


  Output (JSON):
       
      [
          {
              "sentiment": -0.82,
              "emotion": -0.92,
              "tone": -1.0
          }
      ]

🔹 2. Analyze Audio File

    from app import analyze_audio_file
    
    result = analyze_audio_file("sample.wav")
    print(result)


Output Example:
    
    [
        {
            "sentiment": 0.65,
            "emotion": 0.71,
            "tone": 0.88
        }
    ]


## 📂 Project Structure
      ```
      .
      ├── app.py               # Main script
      ├── requirements.txt     # Dependencies
      ├── nltk_data/           # Pre-downloaded NLTK lexicon (optional)
      ├── Dockerfile           # Image of Docker
      └── README.md            # Project documentation
      

## Local Run (FastAPI/Uvicorn)
    ```
    uvicorn app:app --reload



 ##  💡 Use Cases

🛡️ Safety Apps → Detect red/green flags in conversations

📞 Call Center Analytics → Understand customer emotions

💬 Chatbots & Virtual Assistants → Add empathy & tone-awareness

📊 Customer Feedback Analysis → Monitor product sentiment

📓 Journaling Apps → Help users track emotions over time


## 📜 License

MIT License © 2025

## 👨‍💻 Author

Developed by Osama Ahmed 🚀

