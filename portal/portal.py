import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI()
UPLOAD_FOLDER = 'content'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home():
    files_html = "".join([f"<li>{f}</li>" for f in os.listdir(UPLOAD_FOLDER)])
    return f"""
    <html>
        <body style="font-family: sans-serif; padding: 2rem;">
            <h1>Ad Replacement Portal</h1>
            <form action="/upload" method="post" enctype="multipart/form-data" style="margin-bottom: 20px;">
                <input type="file" name="file" required>
                <input type="submit" value="Upload New Content">
            </form>
            <h3>Active Content Files:</h3>
            <ul>{files_html}</ul>
        </body>
    </html>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)