import time
import numpy as np
import sounddevice as sd
import torch
import collections
import torchaudio
import queue
from training.model_arch import AdDetectorCNN
from settings import SAMPLE_RATE, SMOOTHING_WINDOW


def find_stereo_mix():
    for i, dev in enumerate(sd.query_devices()):
        if "stereo mix" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def predict_live(
    model_path,
    thread_name,
    device_index=2,
    sample_rate=44100,
    smoothing_window=5,
    chunk_duration=1.0,
    step_duration=0.2,
    confidence_threshold=0.85,
    prediction_queue=None,
    input_audio_queue=None
):
    """
    Live ad detection. 
    If input_audio_queue is provided, it reads from there (Passive Mode).
    If NOT, it opens the microphone itself (Active Mode).
    """
    # 1. Load model
    device = torch.device("cpu")
    model = AdDetectorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Audio transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=64
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    # 3. Buffers
    raw_buffer_len = int(sample_rate * chunk_duration)
    audio_buffer = collections.deque(maxlen=raw_buffer_len)
    prediction_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)

    # Pre-fill buffer to avoid startup crash
    audio_buffer.extend(np.zeros(raw_buffer_len))

    # --- CORE LOGIC ---
    def run_inference():
        # A. Prepare Snapshot
        snapshot = np.array(audio_buffer)
        waveform = torch.from_numpy(snapshot).float().unsqueeze(0)

        # B. Predict
        with torch.no_grad():
            spec = mel_transform(waveform)
            spec = db_transform(spec)
            spec = spec.unsqueeze(0).to(device)
            raw_pred = model(spec).item()

        # C. Smooth
        prediction_buffer.append(raw_pred)
        smoothed_score = sum(prediction_buffer)/len(prediction_buffer)

        # D. Output
        if prediction_queue is not None:
            prediction_queue.append(smoothed_score)

    # --- MODE A: PASSIVE (Read from Queue) ---
    if input_audio_queue is not None:
        print(f"AI Thread started ({thread_name}) - Context: {chunk_duration}s")
        while True:
            try:
                # Blocking wait for audio
                new_audio = input_audio_queue.get(timeout=5) 
                audio_buffer.extend(new_audio)
                run_inference()
            except queue.Empty:
                continue

    # --- MODE B: ACTIVE (Open Microphone) ---
    else:
        # Device Setup
        device_index = device_index if device_index is not None else find_stereo_mix() 
        device_info = sd.query_devices(device_index)
        channels = device_info['max_input_channels']

        def callback(indata, frames, time_info, status):
            if status: print(status)
            mono = np.mean(indata, axis=1)
            audio_buffer.extend(mono)

        print(f"AI Thread started (Active) - Device: {device_index}")
        with sd.InputStream(device=device_index, channels=channels, 
                            samplerate=sample_rate, callback=callback):
            while True:
                time.sleep(step_duration)
                run_inference()
    
