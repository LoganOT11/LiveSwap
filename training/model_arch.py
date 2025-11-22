import torch
import torch.nn as nn
import torch.nn.functional as F

class AdDetectorCNN(nn.Module):
    def __init__(self):
        super(AdDetectorCNN, self).__init__()
        
        # Low-Level Features
        # Input: [Batch, 1, 64, Time] (1 Channel, 64 Mel bins, Variable Time)
        # We use 1 channel because spectrograms are grayscale (intensity only).
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Normalizes data to speed up training
        self.pool = nn.MaxPool2d(2, 2) # Cuts dimensions in half

        # Shapes & Patterns
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # We pool again to keep shrinking the data size
        
        # Complex Textures
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Deep Features 
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Adaptive layer - output size is (4,4) regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Input size is fixed thanks to adaptive pool: 
        # 128 channels * 4 height * 4 width = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5) # Randomly disconnects neurons to prevent memorization
        self.fc2 = nn.Linear(512, 1)   # Output: Single Probability Score (0.0 to 1.0)

    def forward(self, x):
        # Pass through the convolutional blocks with Activation (ReLU) and Pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 4th block
        x = F.relu(self.bn4(self.conv4(x)))

        # Force to fixed size map (4x4)
        x = self.adaptive_pool(x)

        # Flatten: Turn 3D cube into 1D vector
        x = x.view(x.size(0), -1)

        # Decision
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid squashes output to be between 0 (Content) and 1 (Ad)
        return torch.sigmoid(x)

# Debugging Block
if __name__ == "__main__":
    print("Testing Model Architecture...")
    
    # Test 1: Simulate the "Main Model" (4.3 seconds)
    # 4.3s * 44100Hz / 512 hop_length ≈ 370 time steps
    input_4s = torch.randn(1, 1, 64, 370) 
    
    # Test 2: Simulate the "Fast Model" (1.0 second)
    # 1.0s * 44100Hz / 512 hop_length ≈ 86 time steps
    input_1s = torch.randn(1, 1, 64, 86)

    model = AdDetectorCNN()

    print(f"   Input 4.3s Shape: {input_4s.shape}")
    out_1 = model(input_4s)
    print(f"   Output: {out_1.item():.4f} Check")

    print(f"   Input 1.0s Shape: {input_1s.shape}")
    out_2 = model(input_1s)
    print(f"   Output: {out_2.item():.4f} Check")
    
    print("\nSuccess! The same model handles both durations.")
