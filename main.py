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

# ✅ توکن ثابت برای API بررسی محتوا
REVISION_API_TOKEN = "YwrdzYgYnMAGWyE18LVu1B4sbOz2qzpeo0g3dzKslFiCI0EMSdA0rxPue4YKDaYT"
UPLOAD_API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI1NTAiLCJqdGkiOiJjN2FiYTE2NTBkNzA3ZTg4Mjc5YzI4MTY4ZTczYzc4NzJkMzY1NWIwMGFjYjRkZGJhMDA1ZWE2MTU0ODhjZjFjZjExZjJjZWU1MmNhNzcxNyIsImlhdCI6MTc0NzY0MTkxNC42ODg0MjIsIm5iZiI6MTc0NzY0MTkxNC42ODg0MjYsImV4cCI6MTc3OTE3NzkxNC42NzIxNzQsInN1YiI6IjE2NTExNjc5Iiwic2NvcGVzIjpbIm9yZGVyLXByb2Nlc3NpbmciLCJ2ZW5kb3IucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2ZpbGUud3JpdGUiLCJjdXN0b21lci5wcm9maWxlLndyaXRlIiwiY3VzdG9tZXIucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2R1Y3Qud3JpdGUiLCJ2ZW5kb3IucHJvZHVjdC5yZWFkIiwiY3VzdG9tZXIub3JkZXIucmVhZCIsImN1c3RvbWVyLm9yZGVyLndyaXRlIiwidmVuZG9yLnBhcmNlbC5yZWFkIiwidmVuZG9yLnBhcmNlbC53cml0ZSIsImN1c3RvbWVyLndhbGxldC5yZWFkIiwiY3VzdG9tZXIud2FsbGV0LndyaXRlIiwiY3VzdG9tZXIuY2hhdC5yZWFkIiwiY3VzdG9tZXIuY2hhdC53cml0ZSJdLCJ1c2VyX2lkIjoxNjUxMTY3OX0.EbrqOaUaGI6wORC446IDclq4gg8j2mWhuVzHA82tph2PZ6Fnx2sPMMqCuhOSavSXX6Vuk6Pmfh_qMIl_zcAPXvBvgmi62or1BRPCqOZ9E-L0DUWdDIiY8tpU6Rxl5QkISCjS-K5dpgpj6aBwQYadYQKUxUN0JJ_usgNSeSXYfAUJvVxOO3ZhpSjZ9O4jEu2vPZSiS5gkOIw-Q8Erz9GHB21m_3h2r2XJvJEwJ6GfPuYVubfMlNFMfufpqHQUpRyov0OAS_wCMGJmA5jBYHxlt3GEAb-hU0eWP6Tg44y5XO65gaIF1vyLKu5tHZ1j-d6Oue3wolxb3NgTwZHTVsxR6pUrA6j90vunHLVSlE4uVD0QYB3R2PUKOA5tM6LWgu72d3ynnSRrBBXEpBMy-SFk0iESyOLKD2qCXcetRfRlDPBoKVjotavp2W0hU9GDthVzopsKQaD8YrQW1zSWXPKRxgflod455bRZdeJo2dvJMhZAX8C7wTcGyJddcLO4Eq-bT7w7yJnBeSUEZtycqHVCD6mIZ_gq4jlVtYir4tnU5IKHpeMITkCaA1H9QcOr1VdGVngfrjqRSduATGA-IxW9VKeiHNYowZ6JQrbbXi0GDCKluPbGxji5WYV-kvRt3afzEiYx1vuDns58Xu8hkac8rdVrXbbpkIZyv46V6R1z4-o"
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

    local_images = []
    for idx, frame in enumerate(result["frames"]):
        local_images.append({
            "file_id": idx,
            "local_path": frame["image_path"]
        })

    # ✅ مرحله 1: آپلود هر تصویر به باسلام
    uploaded_images = []
    for image in local_images:
        try:
            with open(image["local_path"], "rb") as f:
                files = {"file": f}
                headers = {
                    "Authorization": UPLOAD_API_TOKEN
                }
                upload_response = requests.post(
                    "https://uploadio.basalam.com/v3/files",
                    headers=headers,
                    files=files
                )
                upload_response.raise_for_status()
                upload_result = upload_response.json()
                uploaded_images.append({
                    "file_id": image["file_id"],
                    "url": upload_result["url"]
                })
        except Exception as e:
            return {
                "error": f"Failed to upload image {image['local_path']}",
                "details": str(e)
            }

    # ✅ مرحله 2: ارسال به revision برای بررسی محتوای نامناسب
    payload = {"images": uploaded_images}
    moderation_api_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {
        "api-token": REVISION_API_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(moderation_api_url, json=payload, headers=headers)

        try:
            moderation_result = response.json()
            if not isinstance(moderation_result, list):
                raise ValueError("API response is not a list")
        except Exception:
            return {
                "error": "Invalid response from moderation API",
                "status_code": response.status_code,
                "raw_response": response.text
            }

        forbidden_images = []
        for result in moderation_result:
            try:
                if result.get("is_forbidden") is True:
                    file_id = result.get("file_id")
                    if file_id is not None and 0 <= file_id < len(uploaded_images):
                        forbidden_images.append(uploaded_images[file_id]["url"])
            except Exception:
                continue

        return {
            "is_forbidden": len(forbidden_images) > 0,
            "forbidden_images": forbidden_images
        }

    except Exception as e:
        return {
            "error": f"Failed to call moderation API: {str(e)}"
        }

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
        scene_threshold = 20.0
        SKIP_FRAMES = 3
        prev_gray = None
        frame_index = 0
        last_saved_index = -15
        output_frames_info = []

        url_prefix = "https://video-check.darkube.app/frame?path="  # 🔁 لینک نهایی دسترسی

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

                image_url = url_prefix + frame_filepath  # ⬅️ لینک نهایی رو می‌سازیم

                output_frames_info.append({
                    "file_id": len(output_frames_info),
                    "frame": frame_index,
                    "timestamp": round(frame_index / fps, 2),
                    "scene_diff": round(scene_diff, 2),
                    "url": image_url  # فقط URL می‌دیم
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
