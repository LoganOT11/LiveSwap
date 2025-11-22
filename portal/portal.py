import os
import shutil
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# --- CONFIGURATION ---
app = FastAPI()
UPLOAD_FOLDER = 'content'
STATE_FILE = 'content_state.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CRITICAL STEP: Mount the content folder to serve files (images/videos)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

def get_state():
    """Reads the current active/disabled state from the JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: 
                return json.load(f)
        except json.JSONDecodeError:
            print("WARNING: Corrupt JSON state file. Returning empty state.")
            return {}
    return {}

def save_state(state):
    """Writes the current active/disabled state to the JSON file."""
    with open(STATE_FILE, 'w') as f: 
        json.dump(state, f, indent=4)

@app.get("/", response_class=HTMLResponse)
async def home():
    state = get_state()
    files = sorted(os.listdir(UPLOAD_FOLDER))
    
    list_items = ""
    for f in files:
        is_active = state.get(f, True)
        status_color = "green" if is_active else "gray"
        status_text = "Active" if is_active else "Disabled"
        
        is_image = f.lower().endswith(('.png', '.jpg', '.jpeg'))
        is_video = f.lower().endswith(('.mp4', '.avi', '.mov'))
        preview_tag = ""
        
        if is_image:
            preview_tag = f'<img src="/static/{f}" class="preview-img" loading="lazy">'
        elif is_video:
            preview_tag = f'<div class="preview-video-placeholder">&#9658; VIDEO</div>'
        else:
            preview_tag = f'<div class="preview-placeholder">FILE</div>'

        list_items += f"""
        <li class="file-item">
            <div class="preview-area">{preview_tag}</div>
            <span class="file-name">{f}</span>
            <span class="file-status" style="color: {status_color};">{status_text}</span>
            
            <form action="/toggle" method="post" style="margin:0;">
                <input type="hidden" name="filename" value="{f}">
                <input type="submit" value="{'Disable' if is_active else 'Enable'}" class="btn-toggle">
            </form>
            
            <form action="/delete" method="post" style="margin:0;" onsubmit="return confirm('Delete {f}?');">
                <input type="hidden" name="filename" value="{f}">
                <input type="submit" value="Delete" class="btn-delete">
            </form>
        </li>
        """

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ad Replacement Portal</title>
        <style>
            /* Base Styling */
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 0; background: #f4f4f9; }}
            .container {{ max-width: 900px; margin: 2rem auto; background: white; padding: 2rem; box-shadow: 0 6px 12px rgba(0,0,0,0.15); border-radius: 12px; }}
            h1 {{ color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; margin-bottom: 20px; }}
            
            /* Upload Box */
            .upload-box {{ background: #e8eaf6; padding: 20px; border-radius: 8px; margin-bottom: 30px; display: flex; gap: 15px; align-items: center; }}
            
            /* File List (Desktop Grid) */
            .file-list {{ list-style: none; padding: 0; }}
            .file-item {{
                display: grid;
                /* Preview | Name | Status | Toggle | Delete */
                grid-template-columns: 100px 1fr 100px 100px 80px; 
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
                gap: 15px;
            }}
            .file-item:last-child {{ border-bottom: none; }}
            
            /* Preview Area */
            .preview-area {{ width: 100%; height: 60px; overflow: hidden; }}
            .preview-img {{ width: 100%; height: 100%; object-fit: cover; border-radius: 4px; display: block; }}
            .preview-video-placeholder {{
                width: 100%; height: 100%; background: #607D8B; color: white; font-size: 0.8em;
                display: flex; justify-content: center; align-items: center; border-radius: 4px;
            }}

            /* Buttons */
            input[type="submit"] {{
                padding: 8px 12px; cursor: pointer; border: none; border-radius: 4px; font-weight: 600;
                transition: background 0.2s; white-space: nowrap;
            }}
            .btn-toggle {{ background: #4caf50; color: white; }}
            .btn-delete {{ background: #f44336; color: white; }}
            
            /* --- Mobile Layout Override --- */
            @media (max-width: 600px) {{
                .container {{ margin: 1rem; padding: 1rem; }}
                
                /* FIX 2: Upload Box Stack */
                .upload-box {{
                    flex-direction: column;
                    gap: 10px;
                    align-items: stretch; 
                }}
                .upload-box input[type="file"], .upload-box input[type="submit"] {{
                    width: 100%;
                    box-sizing: border-box;
                }}

                /* FIX 1: File Item Grid Layout */
                .file-item {{
                    grid-template-columns: 70px 1fr 1fr; /* Preview | Name/Status | Toggle/Delete */
                    grid-template-rows: auto auto; /* Two functional rows */
                    padding: 10px 0;
                    gap: 10px 15px;
                }}
                .preview-area {{ grid-row: 1 / 3; }} /* Preview spans both rows */
                
                .file-name {{ grid-column: 2 / 4; font-size: 1em; }} /* Name spans remaining width */
                .file-status {{ grid-column: 2 / 4; grid-row: 1 / 2; text-align: right; font-size: 0.9em; }}
                
                /* Buttons: Positioned on the second row */
                .file-item form {{
                    grid-row: 2 / 3;
                    display: block;
                    width: 100%;
                }}
                /* Toggle button (Form 1) */
                .file-item form:nth-of-type(1) {{ grid-column: 2 / 3; }}
                /* Delete button (Form 2) - RESTORED and placed in third column */
                .file-item form:nth-of-type(2) {{ 
                    grid-column: 3 / 4; 
                    display: block !important; /* Ensure it is not hidden */
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ad Replacement Portal</h1>
            
            <form action="/upload" method="post" enctype="multipart/form-data" class="upload-box">
                <input type="file" name="file" required>
                <input type="submit" value="Upload New Content" class="btn-upload">
            </form>
            
            <h3>Manage Content:</h3>
            <ul class="file-list">{list_items}</ul>
        </div>
    </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Enable new files by default
    state = get_state()
    state[file.filename] = True
    save_state(state)
    return RedirectResponse(url="/", status_code=303)

@app.post("/toggle")
async def toggle_file(filename: str = Form(...)):
    state = get_state()
    state[filename] = not state.get(filename, True)
    save_state(state)
    return RedirectResponse(url="/", status_code=303)

@app.post("/delete")
async def delete_file(filename: str = Form(...)):
    # Remove physical file
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path): os.remove(path)
    
    # Remove from state
    state = get_state()
    if filename in state: 
        del state[filename]
        save_state(state)
        
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)