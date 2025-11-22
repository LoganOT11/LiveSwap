import cv2
import os
import random
import numpy as np
import json

class VideoManager:
    def __init__(self, content_folder, stream_index=0):
        self.content_folder = content_folder
        self.cap = cv2.VideoCapture(stream_index)
        # Set hardware resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        
        self.ad_cap, self.ad_img = None, None

    def _get_active_files(self):
        """Reads content_state.json. If missing, creates it with all current files enabled."""
        # 1. Scan directory for physical files
        try:
            all_files = [f for f in os.listdir(self.content_folder) 
                         if f.lower().endswith(('.png','.jpg','.jpeg','.mp4','.avi'))]
        except FileNotFoundError:
            return []

        state_path = 'content_state.json'

        # 2. Auto-Create State File if Missing
        if not os.path.exists(state_path):
            print("Creating new content_state.json...")
            initial_state = {f: True for f in all_files}
            try:
                with open(state_path, 'w') as f:
                    json.dump(initial_state, f, indent=4)
            except Exception as e:
                print(f"Error creating state file: {e}")
            
            # Return all files since we just initialized them as Active
            return [os.path.join(self.content_folder, f) for f in all_files]

        # 3. Read Existing State
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            # Filter: Include file if it's NOT in state (default True) or value is True
            active_files = [f for f in all_files if state.get(f, True)]
            return [os.path.join(self.content_folder, f) for f in active_files]
        
        except Exception as e:
            print(f"State Read Error: {e}")
            # Fallback: Return all files if JSON is corrupt
            return [os.path.join(self.content_folder, f) for f in all_files]

    def get_frame(self, is_ad):
        # --- AD MODE ---
        if is_ad:
            if not self.ad_cap and self.ad_img is None:
                # Get valid files (will create json if missing)
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
        
        # --- PASSTHROUGH MODE ---
        if self.ad_cap: self.ad_cap.release(); self.ad_cap = None
        self.ad_img = None
        
        ret, frame = self.cap.read()
        return frame if ret else np.zeros((600, 800, 3), np.uint8)

    def release(self):
        self.cap.release()
        if self.ad_cap: self.ad_cap.release()