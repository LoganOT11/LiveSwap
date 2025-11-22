import os
import shutil
import uvicorn
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI()
UPLOAD_FOLDER = 'content'
STATE_FILE = 'content_state.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f)

@app.get("/", response_class=HTMLResponse)
async def home():
    state = get_state()
    files = sorted(os.listdir(UPLOAD_FOLDER))
    
    # Build list items with Toggle and Delete controls
    list_items = ""
    for f in files:
        is_active = state.get(f, True) # Default to True if new
        status_color = "green" if is_active else "gray"
        status_text = "Active" if is_active else "Disabled"
        
        list_items += f"""
        <li style="margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
            <span style="width: 300px; font-weight: bold; color: {status_color};">{f} ({status_text})</span>
            
            <form action="/toggle" method="post" style="margin:0;">
                <input type="hidden" name="filename" value="{f}">
                <input type="submit" value="{'Disable' if is_active else 'Enable'}">
            </form>
            
            <form action="/delete" method="post" style="margin:0;" onsubmit="return confirm('Delete {f}?');">
                <input type="hidden" name="filename" value="{f}">
                <input type="submit" value="Delete" style="background: #ffcccc; border: 1px solid red;">
            </form>
        </li>
        """

    return f"""
    <html>
        <body style="font-family: sans-serif; padding: 2rem;">
            <h1>Ad Replacement Portal</h1>
            <form action="/upload" method="post" enctype="multipart/form-data" style="margin-bottom: 20px; padding: 15px; background: #eee;">
                <input type="file" name="file" required>
                <input type="submit" value="Upload New Content">
            </form>
            <h3>Manage Content:</h3>
            <ul style="list-style: none; padding: 0;">{list_items}</ul>
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