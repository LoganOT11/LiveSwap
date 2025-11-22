import time
import numpy as np
import sounddevice as sd
import torch
import collections
import torchaudio
import queue
import sys
import os

# Dynamic Import: Handles finding model_arch whether running from root or subfolder
try:
    from training.model_arch import AdDetectorCNN
except ImportError:
    # Fallback if running relative
    from ..training.model_arch import AdDetectorCNN

def find_stereo_mix():
    for i, dev in enumerate(sd.query_devices()):
        if "stereo mix" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def predict_live(
    model_path,
    thread_name="AI-Thread", # Optional default
    device_index=None,       # Default None allows auto-find
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
    # CORRECTED: Uses the local 'sample_rate' argument, not a global constant
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=64
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    # 3. Buffers
    # CORRECTED: Uses local 'sample_rate' and 'smoothing_window'
    raw_buffer_len = int(sample_rate * chunk_duration)
    audio_buffer = collections.deque(maxlen=raw_buffer_len)
    prediction_buffer = collections.deque(maxlen=smoothing_window)

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
        
        if len(prediction_buffer) > 0:
            smoothed_score = sum(prediction_buffer)/len(prediction_buffer)
        else:
            smoothed_score = 0.0

        # D. Output
        if prediction_queue is not None:
            prediction_queue.append(smoothed_score)

    # --- MODE A: PASSIVE (Read from Queue) ---
    if input_audio_queue is not None:
        print(f"{thread_name} started (Passive) - Context: {chunk_duration}s")
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

        print(f"{thread_name} started (Active) - Device: {device_index}")
        with sd.InputStream(device=device_index, channels=channels, 
                            samplerate=sample_rate, callback=callback):
            while True:
                time.sleep(step_duration)
                run_inference()
