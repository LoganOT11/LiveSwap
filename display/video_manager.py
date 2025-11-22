import cv2
import os
import random
import numpy as np

class VideoManager:
    def __init__(self, content_folder, stream_device_index=1, drain_buffer=True):
        """
        drain_buffer (bool): If True, we keep reading from the capture card (discarding frames)
                             while showing ads. This prevents the 30-second lag when switching back.
        """
        self.content_folder = content_folder
        self.stream_index = stream_device_index
        self.width = 800
        self.height = 600
        self.drain_buffer = drain_buffer # <--- NEW CONTROL VARIABLE
        
        # State Variables
        self.cap_passthrough = None
        self.ad_cap = None
        self.current_ad_image = None
        self.content_files = self._load_content_files()
        
        # Start the Passthrough immediately
        self._setup_passthrough()

    def _load_content_files(self):
        """Scans folder for valid images/videos."""
        if not os.path.exists(self.content_folder):
            print(f"Content folder '{self.content_folder}' missing.")
            return []
            
        files = [os.path.join(self.content_folder, f) 
                 for f in os.listdir(self.content_folder)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov"))]
        
        if not files:
            print("No content files found.")
        return files

    def _setup_passthrough(self):
        """Attempts to connect to OBS/Capture Card."""
        try:
            # CAP_DSHOW is highly recommended for Windows Capture Cards to reduce latency
            self.cap_passthrough = cv2.VideoCapture(self.stream_index, cv2.CAP_DSHOW)
            if not self.cap_passthrough.isOpened():
                print(f"Error: Could not open device {self.stream_index}.")
                self.cap_passthrough = None
            else:
                self.cap_passthrough.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap_passthrough.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                # Try to set hardware buffer size to 1 (minimal latency)
                self.cap_passthrough.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"Passthrough active on device {self.stream_index}")
        except Exception as e:
            print(f"Video Error: {e}")

    def get_frame(self, score, threshold=0.85):
        """
        The Master Logic: Returns the correct image frame based on the AI score.
        """
        is_ad_detected = score > threshold

        if is_ad_detected and self.content_files:
            return self._get_ad_frame(score)
        else:
            return self._get_passthrough_frame(score)

    def _get_passthrough_frame(self, score):
        """Handles showing the live game."""
        # 1. Clean up any running ad
        if self.ad_cap:
            self.ad_cap.release()
            self.ad_cap = None
        self.current_ad_image = None

        # 2. Read Live Frame
        frame = None
        if self.cap_passthrough:
            ret, frame = self.cap_passthrough.read()
            if not ret: frame = None

        # 3. Fallback if stream is dead
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(frame, "NO SIGNAL", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        else:
            # Optional: Resize to ensure consistency
            frame = cv2.resize(frame, (self.width, self.height))
            # Overlay status
            cv2.putText(frame, f"LIVE GAME ({score:.1%})", (30, self.height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def _get_ad_frame(self, score):
        """Handles playing the filler content."""
        
        # --- CRITICAL BUFFER IMPLEMENTATION ---
        # While showing an ad, we MUST grab the live frame and throw it away.
        # If we don't, the hardware buffer fills up, and when we switch back, 
        # we see the game from 30 seconds ago.
        if self.drain_buffer:
            if self.cap_passthrough and self.cap_passthrough.isOpened():
                self.cap_passthrough.grab() # Fastest way to clear buffer
        # --------------------------------------

        # 1. If we aren't playing anything yet, pick a new file
        if self.ad_cap is None and self.current_ad_image is None:
            file_path = random.choice(self.content_files)
            
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                # It's an Image
                img = cv2.imread(file_path)
                if img is not None:
                    self.current_ad_image = cv2.resize(img, (self.width, self.height))
            else:
                # It's a Video
                self.ad_cap = cv2.VideoCapture(file_path)
        
        # 2. Return the content
        if self.current_ad_image is not None:
            # Return the static image
            frame = self.current_ad_image.copy()
        elif self.ad_cap is not None:
            # Read next video frame
            ret, frame = self.ad_cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
            else:
                # Video ended -> Restart loop or Pick new (Recursive call)
                self.ad_cap.release()
                self.ad_cap = None
                return self._get_ad_frame(score) # Pick a new one immediately
        else:
            # Fallback (Shouldn't happen)
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Overlay Status
        cv2.putText(frame, f"AD BLOCKED ({score:.1%})", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def release(self):
        if self.cap_passthrough: self.cap_passthrough.release()
        if self.ad_cap: self.ad_cap.release()
