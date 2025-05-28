from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import requests
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

class VideoURL(BaseModel):
    url: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/extract-frames/")
async def extract_frames(video_url: VideoURL):
    # دانلود ویدیو
    video_response = requests.get(video_url.url, stream=True)
    if video_response.status_code != 200:
        return {"error": "Unable to download video"}

    output_dir = tempfile.mkdtemp()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        for chunk in video_response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        video_path = tmp_file.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.rmdir(output_dir)
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scene_threshold = 30.0
        SKIP_FRAMES = 5
        prev_gray = None
        frame_index = 0
        last_saved_index = -15
        output_frames_info = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % SKIP_FRAMES != 0:
                frame_index += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            save_frame = False
            scene_diff = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)

                if scene_diff > scene_threshold:
                    if frame_index - last_saved_index >= 10:
                        save_frame = True

            if save_frame:
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)

                output_frames_info.append({
                    "frame": int(frame_index),
                    "timestamp": float(round(frame_index / fps, 2)),
                    "scene_diff": float(round(scene_diff, 2)),
                    "image_path": frame_filepath
                })
                last_saved_index = frame_index

            prev_gray = gray
            frame_index += 1

        cap.release()
        os.unlink(video_path)

        return {
            "frames": output_frames_info,
            "total": len(output_frames_info),
            "output_directory": output_dir
        }

    except Exception as e:
        if 'output_dir' in locals() and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        return {"error": str(e)}

# 🔍 اندپوینت برای نمایش یا دانلود فایل تصویری با دادن مسیر کامل
@app.get("/frame")
def get_frame(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}
