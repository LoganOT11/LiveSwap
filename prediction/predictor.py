import time
import numpy as np
import sounddevice as sd
import torch
import collections
import cv2
import torchaudio
from .model_arch import AdDetectorCNN

def find_stereo_mix():
    for i, dev in enumerate(sd.query_devices()):
        if "stereo mix" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    return None

def predict_live(
    model_path,
    device_index=None,
    sample_rate=44100,
    smoothing_window=5,
    chunk_duration=1.0,
    step_duration=0.2,
    confidence_threshold=0.85,
    show_hud=False,
    prediction_queue=None 
):
    """
    Live ad detection. Only device_index and model_path need to be provided.
    Other values have defaults.
    """
    # Load model
    device = torch.device("cpu")
    model = AdDetectorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Audio device info
    device_index = device_index if device_index is not None else find_stereo_mix() 
    device_info = sd.query_devices(device_index)
    channels = device_info['max_input_channels']

    # Audio transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=64
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    # Buffers
    raw_buffer_len = int(sample_rate * chunk_duration)
    audio_buffer = collections.deque(maxlen=raw_buffer_len)
    prediction_buffer = collections.deque(maxlen=smoothing_window)

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        mono = np.mean(indata, axis=1)
        audio_buffer.extend(mono)

    def draw_hud(spec_tensor, score, raw_audio):
        UI_WIDTH = 600
        spec_img = spec_tensor.squeeze().cpu().numpy()
        spec_img = (spec_img + 80) / 80.0
        spec_img = np.clip(spec_img, 0, 1) * 255
        spec_img = spec_img.astype(np.uint8)
        spec_color = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
        spec_resized = cv2.resize(spec_color, (UI_WIDTH, 250), interpolation=cv2.INTER_LINEAR)

        wave_h = 100
        wave_canvas = np.zeros((wave_h, UI_WIDTH, 3), dtype=np.uint8)
        step = max(1, len(raw_audio) // UI_WIDTH)
        points = [(i, int(50 - raw_audio[i*step]*40)) for i in range(len(raw_audio)//step)]
        cv2.polylines(wave_canvas, [np.array(points)], isClosed=False, color=(0,255,255), thickness=1)
        cv2.line(wave_canvas, (0,50),(UI_WIDTH,50),(50,50,50),1)

        is_ad = score > confidence_threshold
        bar_color = (0,0,255) if is_ad else (0,255,0)
        bar_canvas = np.zeros((80, UI_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(bar_canvas, (0,0),(int(score*UI_WIDTH),80), bar_color, -1)
        text = "ADVERTISEMENT" if is_ad else "CONTENT"
        cv2.putText(bar_canvas, f"{text} ({score:.1%})", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        final_ui = np.vstack((spec_resized, wave_canvas, bar_canvas))
        cv2.imshow("Ad Detector Dashboard", final_ui)
        cv2.waitKey(1)

    audio_buffer.extend(np.zeros(raw_buffer_len))
    required_samples = raw_buffer_len

    print(f"Listening on device {device_index}: {device_info['name']}, averaging last {smoothing_window} guesses...")
    with sd.InputStream(device=device_index, channels=channels, samplerate=sample_rate, callback=callback):
        while True:
            if len(audio_buffer) < required_samples:
                time.sleep(0.1)
                continue

            snapshot = np.array(audio_buffer)
            waveform = torch.from_numpy(snapshot).float().unsqueeze(0)

            with torch.no_grad():
                spec = mel_transform(waveform)
                spec = db_transform(spec)
                spec_for_ui = spec.clone()
                spec = spec.unsqueeze(0).to(device)
                raw_pred = model(spec).item()

            prediction_buffer.append(raw_pred)
            smoothed_score = sum(prediction_buffer)/len(prediction_buffer)

            if show_hud:
                draw_hud(spec_for_ui, smoothed_score, snapshot)

            if prediction_queue is not None:
                prediction_queue.append(smoothed_score)

            time.sleep(step_duration)

# python -m prediction.predictor
if __name__ == "__main__":
    try:
        import settings

        predict_live(
            # device_index=settings.DEVICE_INDEX,
            model_path=settings.MODEL_PATH,
            show_hud=True
        )
    except KeyboardInterrupt:
        print("\nStopped.")
        cv2.destroyAllWindows()

    
    
