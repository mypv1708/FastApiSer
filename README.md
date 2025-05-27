# FastAPI Emotion Detection from Audio

## Introduction

This project provides a REST API using FastAPI to detect emotions from voice audio files (wav). The API uses a trained deep learning model to classify emotions into 5 categories: Angry, Cold, Tired, Friendly, and Happy.

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

Run the following command to start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8386
```

## Using the API

- Endpoint: `POST /predict-emotion/`
- Parameters: Send a `wav` audio file via form-data with the key `file`.
- Example using curl:
  ```bash
  curl -X POST "http://localhost:8386/predict-emotion/" -F "file=@audio/AudioRecord.wav"
  ```
- Returns JSON containing:
  - Original filename and duration
  - Emotion predictions for each audio segment
  - Percentage duration for each emotion
  - Overview of positive/negative emotion percentages

## Directory Structure

- `main.py`: Main FastAPI code file.
- `requirements.txt`: List of required libraries.
- `saved_models/`: Contains trained model files.
- `audio/`: (Optional) Contains sample audio files.
- `temp/`: Temporary directory for file processing (automatically created/deleted).

## Notes

- Only supports WAV format audio files.
- Temporary files are automatically deleted after processing.
- If encountering model errors, check the files in `saved_models/`.
- Endpoints and models can be extended or modified as needed.
