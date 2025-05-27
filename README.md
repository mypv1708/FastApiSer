# FastAPI Emotion Detection from Audio

## Introduction

This project provides a REST API using FastAPI to detect emotions from voice audio files (wav). The API uses a trained deep learning model to classify emotions into 5 categories: Angry, Cold, Tired, Friendly, and Happy.

## Project Structure

```
FastApiSer/
├── app/
│   ├── api/
│       └── endpoints.py      # API endpoints
│       └── emotion_model.py # Model handling
│       └── audio_processor.py # Audio processing
├── saved_models/
│   ├── emotion_detection_cnn.json
│   └── emotion_detection_cnn.weights.h5
├── temp/
│   ├── output_chunks/      # Temporary chunk files
│   └── processed/          # Processed audio files
├── main.py                 # Main application file
├── run.py                  # Server runner
└── requirements.txt        # Project dependencies
```

## System Requirements

- Python 3.8+
- Libraries listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd FastApiSer
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Model Preparation

- Ensure the `saved_models/` directory contains the following files:
  - `emotion_detection_cnn.json`
  - `emotion_detection_cnn.weights.h5`

## Running the Server

You can start the server in two ways:

1. Using the run script:

   ```bash
   python run.py
   ```

2. Using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8386 --reload
   ```

The server will start at `http://localhost:8386`

## Using the API

- Endpoint: `POST /predict-emotion/`
- Parameters: Send a `wav` audio file via form-data with the key `file`
- Example using curl:
  ```bash
  curl -X POST "http://localhost:8386/predict-emotion/" -F "file=@audio/AudioRecord.wav"
  ```

### Response Format

```json
{
  "original_file": "example.wav",
  "original_duration": 10.5,
  "original_file_path": "/audio/example.wav",
  "predictions_details": [
    {
      "file": "example_chunk_1.wav",
      "file_path": "/audio/processed/example_chunk_1.wav",
      "emotion": "Friendly",
      "duration": 2.5,
      "probability": 85.5,
      "probability_positive": 85.5,
      "probability_negative": 14.5
    }
  ],
  "emotion_percentages": {
    "Friendly": 72.22,
    "Angry": 27.78
  },
  "overview_percentage": {
    "positive_percentage": 72.22,
    "negative_percentage": 27.78
  }
}
```

## Features

- Audio file processing and chunking
- Emotion detection using deep learning
- Support for WAV format audio files
- Automatic file cleanup
- Static file serving for processed audio
- Detailed emotion analysis with probabilities
- Positive/negative emotion overview

## Notes

- Only supports WAV format audio files
- Processed audio files are available via `/audio/` endpoint
- API documentation is available at `/docs` when server is running
- The server automatically reloads when code changes (in development mode)
