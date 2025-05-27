from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from keras.models import model_from_json
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router

# Tạo ứng dụng FastAPI
app = FastAPI()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc bạn có thể chỉ định các nguồn cụ thể
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các tiêu đề
)

# Thêm vào sau phần khởi tạo app
app.mount("/audio", StaticFiles(directory="temp"), name="audio")

# Load model đã huấn luyện
json_file_path = r'saved_models/emotion_detection_cnn.json'
weights_file_path = r'saved_models/emotion_detection_cnn.weights.h5'

with open(json_file_path, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weights_file_path)

# Thông số xử lý âm thanh
res_type_s = 'kaiser_best'
duration_s = None
sample_rate_s = 16000
sample_rate_mfcc = 16000
offset_s = 0.0
n_mfcc = 40
axis_mfcc = 1
emotions = ['Cáu Giận', 'Lạnh Lùng', 'Mệt Mỏi', 'Thân Thiện', 'Vui Vẻ']

# Hàm trích xuất đặc trưng
def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, 
                                  res_type=res_type_s, 
                                  duration=duration_s, 
                                  sr=sample_rate_s, 
                                  offset=offset_s)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate_mfcc, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=axis_mfcc)
    return mfccs_mean

# Dự đoán cảm xúc từ file WAV
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)
        features_df = pd.DataFrame([features])
        prediction = loaded_model.predict(features_df)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_probability = prediction[0][predicted_class[0]]  # Lấy xác suất của lớp dự đoán
        return emotions[predicted_class[0]], predicted_probability
    except Exception as e:
        return f"Error: {e}", None

# API nhận file WAV và xử lý
@app.post("/predict-emotion/")
async def process_audio(file: UploadFile = File(...)):
    try:
        temp_input_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(await file.read())
        
        original_file_name = file.filename
        y, sr = librosa.load(temp_input_path, sr=sample_rate_s)
        original_duration = librosa.get_duration(y=y, sr=sr)

        # Chia file thành các đoạn âm thanh
        temp_output_dir = "temp/output_chunks"
        os.makedirs(temp_output_dir, exist_ok=True)
        audio = AudioSegment.from_wav(temp_input_path)
        chunks = split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=-58
        )

        chunk_paths = []
        for i, chunk in enumerate(chunks):
            # Lấy tên file gốc không có phần mở rộng
            original_name = os.path.splitext(original_file_name)[0]
            chunk_path = os.path.join(temp_output_dir, f"{original_name}_chunk_{i+1}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)

        # Xử lý cắt đoạn âm thanh không còn khoảng lặng
        processed_output_dir = "temp/processed"
        os.makedirs(processed_output_dir, exist_ok=True)
        processed_chunk_paths = []  # Lưu đường dẫn của các file đã xử lý
        for chunk_path in chunk_paths:
            y, sr = librosa.load(chunk_path, sr=sample_rate_s)
            y_trimmed, _ = librosa.effects.trim(y, top_db=18)
            output_path = os.path.join(processed_output_dir, os.path.basename(chunk_path))
            sf.write(output_path, y_trimmed, sr)
            processed_chunk_paths.append(output_path)

        # Dự đoán cảm xúc từng file
        predictions = []
        emotion_durations = {emotion: 0 for emotion in emotions}
        total_duration = 0

        for file_path in processed_chunk_paths:  # Chỉ xử lý các file chunk mới
            if file_path.endswith(".wav"):
                file_name = os.path.basename(file_path)

                # Tính thời lượng của đoạn âm thanh
                y, sr = librosa.load(file_path, sr=sample_rate_s)
                duration = librosa.get_duration(y=y, sr=sr)
                total_duration += duration

                features = extract_features(file_path)
                features_df = pd.DataFrame([features])

                # Dự đoán cảm xúc
                prediction = loaded_model.predict(features_df)
                predicted_class = np.argmax(prediction, axis=1)
                predicted_probability = prediction[0][predicted_class[0]]

                # Tính toán xác suất tích cực và tiêu cực
                probability_negative = sum(float(round(p * 100, 2)) for p in prediction[0][:3])  

                predictions.append({
                    "file": file_name,
                    "file_path": f"/audio/processed/{file_name}",  # Thay đổi đường dẫn
                    "emotion": emotions[predicted_class[0]],
                    "duration": round(duration, 2),
                    "probability": round(float(predicted_probability * 100), 2),
                    "probability_positive": round(100 - probability_negative, 2),
                    "probability_negative": round(probability_negative, 2)
                })

                emotion_durations[emotions[predicted_class[0]]] += duration
        
        # Tính phần trăm cảm xúc
        emotion_percentages = {
        emotion: round((duration / total_duration) * 100, 2)
            for emotion, duration in emotion_durations.items()
            if duration > 0
        }

        # Tính phần trăm cảm xúc tích cực và tiêu cực
        positive_emotions = ['Thân Thiện', 'Vui Vẻ']
        negative_emotions = ['Cáu Giận', 'Lạnh Lùng', 'Mệt Mỏi']

        positive_duration = sum(emotion_durations[emotion] for emotion in positive_emotions)
        negative_duration = sum(emotion_durations[emotion] for emotion in negative_emotions)

        positive_percentage = round((positive_duration / total_duration) * 100, 2) if total_duration > 0 else 0
        negative_percentage = round((negative_duration / total_duration) * 100, 2) if total_duration > 0 else 0

        # Trả về kết quả JSON
        return JSONResponse(content={
            "original_file": original_file_name,
            "original_duration": original_duration,
            "original_file_path": f"/audio/{original_file_name}",  # Thay đổi đường dẫn
            "predictions_details": predictions,
            "emotion_percentages": emotion_percentages,
            "overview_percentage": {
                "positive_percentage": positive_percentage,
                "negative_percentage": negative_percentage
            }
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        pass
