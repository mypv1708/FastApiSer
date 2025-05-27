import os
import librosa
import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.audio_processor import AudioProcessor
from app.models.emotion_model import EmotionModel
from app.core.config import EMOTIONS

router = APIRouter()
audio_processor = AudioProcessor()
emotion_model = EmotionModel()

@router.post("/predict-emotion/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        temp_input_path = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Get original file duration
        y, sr = librosa.load(temp_input_path, sr=16000)
        original_duration = librosa.get_duration(y=y, sr=sr)

        # Process audio
        chunk_paths = audio_processor.split_audio(temp_input_path)
        processed_paths = audio_processor.process_chunks(chunk_paths)

        # Predict emotions
        predictions = []
        durations = []
        for file_path in processed_paths:
            # Get duration
            y, sr = librosa.load(file_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)

            # Extract features and predict
            features = audio_processor.extract_features(file_path)
            features_df = pd.DataFrame([features])
            prediction = emotion_model.predict(features_df)
            predicted_class = prediction.argmax(axis=1)[0]
            predicted_probability = prediction[0][predicted_class]

            # Calculate probabilities
            probability_negative = sum(float(round(p * 100, 2)) for p in prediction[0][:3])

            predictions.append({
                "file": os.path.basename(file_path),
                "file_path": f"/audio/processed/{os.path.basename(file_path)}",
                "emotion": EMOTIONS[predicted_class],
                "duration": round(duration, 2),
                "probability": round(float(predicted_probability * 100), 2),
                "probability_positive": round(100 - probability_negative, 2),
                "probability_negative": round(probability_negative, 2)
            })

        # Calculate percentages
        emotion_percentages, positive_percentage, negative_percentage = audio_processor.calculate_emotion_percentages(
            [pred["emotion"] for pred in predictions],
            durations
        )

        return JSONResponse(content={
            "original_file": file.filename,
            "original_duration": original_duration,
            "original_file_path": f"/audio/{file.filename}",
            "predictions_details": predictions,
            "emotion_percentages": emotion_percentages,
            "overview_percentage": {
                "positive_percentage": positive_percentage,
                "negative_percentage": negative_percentage
            }
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500) 