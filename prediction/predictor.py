import torch
import torchaudio
import collections
import numpy as np
import cv2
from .model_arch import AdDetectorCNN

# Constants
SAMPLE_RATE = 44100
CHUNK_DURATION = 1.0
SMOOTHING_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.85

class AdPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cpu")
        
        # Load Model
        self.model = AdDetectorCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Audio Transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=64
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

        # Buffers
        self.raw_buffer_len = int(SAMPLE_RATE * CHUNK_DURATION)
        self.audio_buffer = collections.deque(maxlen=self.raw_buffer_len)
        self.prediction_buffer = collections.deque(maxlen=SMOOTHING_WINDOW)
        
        # Pre-fill buffer with zeros
        self.audio_buffer.extend(np.zeros(self.raw_buffer_len))
        
        # State for HUD
        self.last_spec = None
        self.last_snapshot = None

    def predict(self, new_audio_chunk):
        """
        Ingests a numpy array of audio samples (mono).
        Returns smoothed prediction score (0.0 - 1.0).
        """
        # 1. Update Buffer
        self.audio_buffer.extend(new_audio_chunk)

        # 2. Prepare Tensor
        snapshot = np.array(self.audio_buffer)
        self.last_snapshot = snapshot # Save for HUD
        waveform = torch.from_numpy(snapshot).float().unsqueeze(0)

        # 3. Inference
        with torch.no_grad():
            spec = self.mel_transform(waveform)
            spec = self.db_transform(spec)
            
            self.last_spec = spec.clone() # Save for HUD
            
            spec = spec.unsqueeze(0).to(self.device)
            raw_pred = self.model(spec).item()

        # 4. Smoothing
        self.prediction_buffer.append(raw_pred)
        smoothed_score = sum(self.prediction_buffer) / len(self.prediction_buffer)
        
        return smoothed_score

def draw_hud(predictor, score):
    """Helper to visualize the predictor's internal state."""
    if predictor.last_spec is None or predictor.last_snapshot is None:
        return

    UI_WIDTH = 600
    
    # Draw Spectrogram
    spec_img = predictor.last_spec.squeeze().cpu().numpy()
    spec_img = (spec_img + 80) / 80.0
    spec_img = np.clip(spec_img, 0, 1) * 255
    spec_img = spec_img.astype(np.uint8)
    spec_color = cv2.applyColorMap(spec_img, cv2.COLORMAP_INFERNO)
    spec_resized = cv2.resize(spec_color, (UI_WIDTH, 250), interpolation=cv2.INTER_LINEAR)

    # Draw Waveform
    wave_h = 100
    raw_audio = predictor.last_snapshot
    wave_canvas = np.zeros((wave_h, UI_WIDTH, 3), dtype=np.uint8)
    step = max(1, len(raw_audio) // UI_WIDTH)
    points = [(i, int(50 - raw_audio[i*step]*40)) for i in range(len(raw_audio)//step)]
    if points:
        cv2.polylines(wave_canvas, [np.array(points)], isClosed=False, color=(0,255,255), thickness=1)
    cv2.line(wave_canvas, (0,50),(UI_WIDTH,50),(50,50,50),1)

    # Draw Score Bar
    is_ad = score > CONFIDENCE_THRESHOLD
    bar_color = (0,0,255) if is_ad else (0,255,0)
    bar_canvas = np.zeros((80, UI_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(bar_canvas, (0,0),(int(score*UI_WIDTH),80), bar_color, -1)
    text = "ADVERTISEMENT" if is_ad else "CONTENT"
    cv2.putText(bar_canvas, f"{text} ({score:.1%})", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    final_ui = np.vstack((spec_resized, wave_canvas, bar_canvas))
    cv2.imshow("Ad Detector Diagnostic", final_ui)

# python -m prediction.predictor
if __name__ == "__main__":
    import sounddevice as sd
    import settings
    import sys

    print("Initializing passive predictor...")
    predictor = AdPredictor(settings.MODEL_PATH)
    
    # Queue to bridge the audio callback and the main loop
    audio_queue = collections.deque()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # Mix to mono and add to queue
        mono = np.mean(indata, axis=1)
        audio_queue.extend(mono)

    # Use default input device (or Stereo Mix if you prefer)
    print("Starting audio stream for testing...")
    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=callback, blocksize=2048):
            while True:
                # Process whatever is in the queue
                if len(audio_queue) > 0:
                    # Consume all available audio
                    chunk = []
                    while len(audio_queue) > 0:
                        chunk.append(audio_queue.popleft())
                    
                    # Feed to predictor
                    score = predictor.predict(np.array(chunk))
                    
                    # Visualize
                    draw_hud(predictor, score)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()