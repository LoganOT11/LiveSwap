import os
import torch
import torchaudio
from torch.utils.data import Dataset

import soundfile as sf

class AdDataset(Dataset):
    def __init__(self, root_dir = None, target_sample_rate=44100, duration=4.3):
        """
        root_dir: Path to the sliced dataset (e.g., "dataset/main_4.3s")
        """
        
        if root_dir is None:
            # 1. Get folder where loader.py lives (prediction/training)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            project_root = os.path.dirname(current_dir)
            
            # 2. Build path to default dataset (training/dataset/fast_1.0s)
            root_dir = os.path.join(project_root, "dataset", "fast_1.0s")
            
            print(f"No path provided. Defaulting to: {root_dir}")
            
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        
        # Spectrogram Transformer
        # We perform the conversion here to ensure consistency
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=64  # Must match model_arch.py input
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Load CONTENT (Label = 0)
        content_path = os.path.join(root_dir, "content")
        if os.path.exists(content_path):
            for f in os.listdir(content_path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(content_path, f))
                    self.labels.append(0.0) # 0 = Content

        # Load ADS (Label = 1)
        ads_path = os.path.join(root_dir, "ads")
        if os.path.exists(ads_path):
            for f in os.listdir(ads_path):
                if f.endswith(".wav"):
                    self.files.append(os.path.join(ads_path, f))
                    self.labels.append(1.0) # 1 = Ad

        print(f"Loaded {len(self.files)} files from {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]

        # sf.read returns tuple: (data_numpy, sample_rate)
        data_numpy, sr = sf.read(audio_path)
        
        # Convert Numpy -> PyTorch Tensor
        waveform = torch.from_numpy(data_numpy).float()
        
        # Fix Dimensions: 
        # Soundfile gives: [Time, Channels] (e.g. 1000, 2)
        # PyTorch wants:   [Channels, Time] (e.g. 2, 1000)
        if waveform.ndim == 2:
            waveform = waveform.t() # Transpose
        else:
            waveform = waveform.unsqueeze(0) # Handle Mono case

        # Force Mono (Average stereo channels)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert to Spectrogram Image
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        return spec, torch.tensor(label, dtype=torch.float32)
