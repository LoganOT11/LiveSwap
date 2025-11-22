import os
DEVICE_INDEX = 2         # Your VB-Cable Output ID
SAMPLE_RATE = 44100       # Standard for video
CHANNELS = 2              # Stereo is required for spatial detection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Now we build paths relative to the root
RAW_RECORDINGS_DIR = os.path.join(BASE_DIR, "dataset", "raw_recordings")
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fast_1.0s.pth")
MODEL_PATH2 = os.path.join(BASE_DIR, "models", "main_4.3s.pth")
