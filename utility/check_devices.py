import sounddevice as sd

# Lists all audio devices visible on device
# Look for your VB-Cable here and note its ID
def list_devices():
    print("--- Available Audio Devices ---")
    print(sd.query_devices())

if __name__ == "__main__":
    list_devices()
