import sounddevice as sd

from .settings import DEVICE_INDEX

try:
    info = sd.query_devices(DEVICE_INDEX)
    print(f"Device {DEVICE_INDEX}: {info['name']}")
    print(f"Max Output Channels: {info['max_output_channels']}")
    
    if info['max_input_channels'] < 2:
        print("ERROR: Windows still thinks this device is MONO.")
    else:
        print("SUCCESS: Device supports Stereo.")
        
except Exception as e:
    print(f"Could not query device: {e}")
