import cv2
import os
import random
import numpy as np
import json

class VideoManager:
    def __init__(self, content_folder, stream_index=0):
        self.content_folder = content_folder
        self.cap = cv2.VideoCapture(stream_index, cv2.CAP_DSHOW) 
        # actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        
        self.ad_cap, self.ad_img = None, None
        self.drain_buffer = True 

    def _get_active_files(self):
        """Reads content_state.json. If missing, creates it with all current files enabled."""
        try:
            all_files = [f for f in os.listdir(self.content_folder) 
                         if f.lower().endswith(('.png','.jpg','.jpeg','.mp4','.avi'))]
        except FileNotFoundError:
            return []

        state_path = 'content_state.json'
        
        # Auto-Create State File if Missing
        if not os.path.exists(state_path):
            initial_state = {f: True for f in all_files}
            try:
                with open(state_path, 'w') as f:
                    json.dump(initial_state, f, indent=4)
            except Exception as e:
                print(f"Error creating state file: {e}")
            return [os.path.join(self.content_folder, f) for f in all_files]

        # Read Existing State
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            active_files = [f for f in all_files if state.get(f, True)]
            return [os.path.join(self.content_folder, f) for f in active_files]
        except Exception as e:
            print(f"State Read Error: {e}")
            return [os.path.join(self.content_folder, f) for f in all_files]

    def get_frame(self, is_ad):
        # Ad detected
        if is_ad:
            if self.drain_buffer and self.cap.isOpened():
                self.cap.grab()
            
            if not self.ad_cap and self.ad_img is None:
                files = self._get_active_files()
                
                if files:
                    f = random.choice(files)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(f)
                        if img is not None: self.ad_img = img.copy()
                    else:
                        self.ad_cap = cv2.VideoCapture(f)

            if self.ad_img is not None: 
                return self.ad_img.copy()
            
            if self.ad_cap:
                ret, frame = self.ad_cap.read()
                if ret: return frame
                
                self.ad_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.ad_cap.read()
                if ret: return frame
                
                self.ad_cap.release()
                self.ad_cap = None
        
        # Passthrough mode
        if self.ad_cap: self.ad_cap.release(); self.ad_cap = None
        self.ad_img = None
        
        # buffer above, this .read() will fetch a FRESH frame immediately
        ret, frame = self.cap.read()
        return frame if ret else np.zeros((600, 800, 3), np.uint8)

    def release(self):
        self.cap.release()
        if self.ad_cap: self.ad_cap.release()
