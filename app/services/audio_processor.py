import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from app.core.config import (
    RES_TYPE, DURATION, SAMPLE_RATE, SAMPLE_RATE_MFCC,
    OFFSET, N_MFCC, AXIS_MFCC, EMOTIONS, POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS, OUTPUT_CHUNKS_DIR, PROCESSED_DIR
)

class AudioProcessor:
    def __init__(self):
        os.makedirs(OUTPUT_CHUNKS_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)

    def extract_features(self, file_path):
        """Extract MFCC features from audio file"""
        X, sample_rate = librosa.load(
            file_path,
            res_type=RES_TYPE,
            duration=DURATION,
            sr=SAMPLE_RATE,
            offset=OFFSET
        )
        mfccs = librosa.feature.mfcc(y=X, sr=SAMPLE_RATE_MFCC, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs, axis=AXIS_MFCC)
        return mfccs_mean

    def split_audio(self, input_path):
        """Split audio into chunks based on silence"""
        audio = AudioSegment.from_wav(input_path)
        chunks = split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=-58
        )
        
        chunk_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = os.path.join(OUTPUT_CHUNKS_DIR, f"chunk_{i+1}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)
        
        return chunk_paths

    def process_chunks(self, chunk_paths):
        """Process audio chunks by trimming silence"""
        processed_paths = []
        for chunk_path in chunk_paths:
            y, sr = librosa.load(chunk_path, sr=SAMPLE_RATE)
            y_trimmed, _ = librosa.effects.trim(y, top_db=18)
            output_path = os.path.join(PROCESSED_DIR, os.path.basename(chunk_path))
            sf.write(output_path, y_trimmed, sr)
            processed_paths.append(output_path)
        return processed_paths

    def calculate_emotion_percentages(self, predictions, durations):
        """Calculate emotion percentages from predictions"""
        total_duration = sum(durations)
        emotion_durations = {emotion: 0 for emotion in EMOTIONS}
        
        for pred, duration in zip(predictions, durations):
            emotion_durations[EMOTIONS[pred]] += duration

        emotion_percentages = {
            emotion: round((duration / total_duration) * 100, 2)
            for emotion, duration in emotion_durations.items()
            if duration > 0
        }

        positive_duration = sum(emotion_durations[emotion] for emotion in POSITIVE_EMOTIONS)
        negative_duration = sum(emotion_durations[emotion] for emotion in NEGATIVE_EMOTIONS)

        positive_percentage = round((positive_duration / total_duration) * 100, 2) if total_duration > 0 else 0
        negative_percentage = round((negative_duration / total_duration) * 100, 2) if total_duration > 0 else 0

        return emotion_percentages, positive_percentage, negative_percentage 