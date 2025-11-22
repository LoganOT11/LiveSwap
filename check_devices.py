import sounddevice as sd
import cv2

# Lists all audio devices visible on device
# Look for your VB-Cable here and note its ID
def list_devices():
    print("--- Available Audio Devices ---")
    print(sd.query_devices())

def list_video_devices(): 
    print("--- Available Video Capture Devices ---")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        if cap.isOpened():
            print(f"Camera found at Index {i}: Trying to read frame...")
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"   -> Successfully captured a frame! (This is likely the OBS Virtual Camera or a physical webcam)")
            else:
                print(f"   -> Opened but could not read frame. (May be a non-video device or already in use)")
        else:
            print(f"❌ No camera found at Index {i}")

if __name__ == "__main__":
    list_devices()
    # list_video_devices()



