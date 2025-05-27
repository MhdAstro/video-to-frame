from fastapi import FastAPI
from pydantic import BaseModel
import requests
import cv2
import numpy as np
import tempfile
import os
import base64

app = FastAPI()

class VideoURL(BaseModel):
    url: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/extract-frames/")
async def extract_frames(video_url: VideoURL):
    # 1. دانلود ویدیو
    video_response = requests.get(video_url.url, stream=True)
    if video_response.status_code != 200:
        return {"error": "Unable to download video"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        for chunk in video_response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        video_path = tmp_file.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scene_threshold = 30.0
        motion_threshold = 1.5
        prev_gray = None
        frame_index = 0
        last_saved_index = -15
        output_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            save_frame = False
            scene_diff = 0.0
            motion = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)

                if scene_diff < scene_threshold:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

                if scene_diff > scene_threshold or motion > motion_threshold:
                    if frame_index - last_saved_index >= 10:
                        save_frame = True

            if save_frame:
                _, buffer = cv2.imencode(".jpg", frame)
                encoded = base64.b64encode(buffer).decode("utf-8")
                output_frames.append({
                    "frame": int(frame_index),
                    "timestamp": float(round(frame_index / fps, 2)),
                    "scene_diff": float(round(scene_diff, 2)),
                    "motion": float(round(motion, 2)),
                    "image_base64": encoded
                })
                last_saved_index = frame_index

            prev_gray = gray
            frame_index += 1

        cap.release()
        os.unlink(video_path)

        return {"frames": output_frames, "total": len(output_frames)}

    except Exception as e:
        return {"error": str(e)}
