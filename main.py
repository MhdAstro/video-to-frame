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

# ✅ توکن ثابت برای بررسی محتوا
CONTENT_API_TOKEN = "YOUR_STATIC_TOKEN_HERE"  # ← اینجا توکن رو بذار

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/extract-frames/")
async def extract_frames(video_url: VideoURL):
    return extract_frames_internal(video_url.url)

@app.post("/analyze-video/")
async def analyze_video(video_url: VideoURL):
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    images = []
    url_prefix = "https://video-check.darkube.app/frame?path="

    for idx, frame in enumerate(result["frames"]):
        image_url = url_prefix + frame["image_path"]
        images.append({"file_id": idx, "url": image_url})

    payload = {"images": images}
    moderation_api_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {
        "api-token": "YwrdzYgYnMAGWyE18LVu1B4sbOz2qzpeo0g3dzKslFiCI0EMSdA0rxPue4YKDaYT",
        "Content-Type": "application/json"
    }

    try:
        print(payload)
        response = requests.post(moderation_api_url, json=payload, headers=headers)
        moderation_result = response.json()

        forbidden_images = [
            image["url"]
            for image in moderation_result.get("images", [])
            if image.get("is_forbidden") == True
        ]

        return {
            "is_forbidden": len(forbidden_images) > 0,
            "forbidden_images": forbidden_images
        }

    except Exception as e:
        return {"error": f"Failed to call moderation API: {str(e)}"}

def extract_frames_internal(video_url):
    video_response = requests.get(video_url, stream=True)
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
        scene_threshold = 20
        SKIP_FRAMES = 3
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
            scene_diff = 0.0
            save_frame = False

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)
                if scene_diff > scene_threshold :
                    save_frame = True

            if save_frame:
                frame_filename = f"frame_{frame_index:06d}.jpg"
                frame_filepath = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_filepath, frame)

                output_frames_info.append({
                    "frame": frame_index,
                    "timestamp": round(frame_index / fps, 2),
                    "scene_diff": round(scene_diff, 2),
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
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        return {"error": str(e)}

@app.get("/frame")
def get_frame(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}
