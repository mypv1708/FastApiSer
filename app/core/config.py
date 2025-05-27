import os

# Audio processing parameters
RES_TYPE = 'kaiser_best'
DURATION = None
SAMPLE_RATE = 16000
SAMPLE_RATE_MFCC = 16000
OFFSET = 0.0
N_MFCC = 40
AXIS_MFCC = 1

# Emotions
EMOTIONS = ['Cáu Giận', 'Lạnh Lùng', 'Mệt Mỏi', 'Thân Thiện', 'Vui Vẻ']
POSITIVE_EMOTIONS = ['Thân Thiện', 'Vui Vẻ']
NEGATIVE_EMOTIONS = ['Cáu Giận', 'Lạnh Lùng', 'Mệt Mỏi']

# Model paths
MODEL_DIR = 'saved_models'
JSON_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_detection_cnn.json')
WEIGHTS_MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_detection_cnn.weights.h5')

# Temporary directories
TEMP_DIR = 'temp'
OUTPUT_CHUNKS_DIR = os.path.join(TEMP_DIR, 'output_chunks')
PROCESSED_DIR = os.path.join(TEMP_DIR, 'processed') 