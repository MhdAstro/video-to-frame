from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import requests
import cv2
import numpy as np
import tempfile
import os
import shutil

app = FastAPI()

# --- بخش ۱: خواندن متغیرهای محیطی ---
# مقادیر ضروری که باید حتما در محیط تنظیم شوند (بدون مقدار پیش‌فرض)
REVISION_API_TOKEN = os.getenv("REVISION_API_TOKEN")
UPLOAD_API_TOKEN = os.getenv("UPLOAD_API_TOKEN") # این توکن در کد شما استفاده نشده بود اما خوانده می‌شود
APP_BASE_URL = os.getenv("APP_BASE_URL")
MODERATION_API_URL = os.getenv("MODERATION_API_URL")

# بررسی وجود متغیرهای ضروری
if not REVISION_API_TOKEN or not APP_BASE_URL or not MODERATION_API_URL:
    raise ValueError("Missing one or more required environment variables: REVISION_API_TOKEN, APP_BASE_URL, MODERATION_API_URL")

# پارامترهای پیکربندی (با مقادیر پیش‌فرض امن)
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "30"))
SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", "3"))
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "20.0"))
FRAME_RESIZE = (int(os.getenv("FRAME_RESIZE_WIDTH", "224")), int(os.getenv("FRAME_RESIZE_HEIGHT", "224")))


class VideoURL(BaseModel):
    url: str

@app.get("/")
def health_check():
    return {"status": "ok"}

# بقیه کد بدون هیچ تغییری باقی می‌ماند
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

        prev_gray = None
        frame_index = 0
        changes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % SKIP_FRAMES != 0:
                frame_index += 1
                continue

            resized = cv2.resize(frame, FRAME_RESIZE)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            scene_diff = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)
                if scene_diff > SCENE_THRESHOLD:
                    changes.append((frame_index, scene_diff, frame.copy()))

            prev_gray = gray
            frame_index += 1

        cap.release()
        os.unlink(video_path)

        top_changes = sorted(changes, key=lambda x: -x[1])[:MAX_FRAMES]
        output_frames_info = []
        
        url_prefix = f"{APP_BASE_URL}/frame?path="

        for i, (frame_idx, score, frame_original) in enumerate(top_changes):
            frame_filename = f"frame_{frame_idx:06d}.jpeg"
            frame_filepath = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame_original)
            image_url = url_prefix + frame_filepath

            output_frames_info.append({
                "file_id": i,
                "frame": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "scene_diff": round(score, 2),
                "url": image_url,
                "image_path": frame_filepath
            })

        return {
            "frames": output_frames_info,
            "total": len(output_frames_info),
            "output_directory": output_dir
        }

    except Exception as e:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return {"error": str(e)}

@app.post("/extract-frames/")
async def extract_frames(video_url: VideoURL):
    return extract_frames_internal(video_url.url)

@app.post("/analyze-video/")
async def analyze_video(video_url: VideoURL):
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    uploaded_images = []
    for idx, frame in enumerate(result["frames"]):
        frame_url = frame["url"]
        if not frame_url.startswith('http'):
            frame_url = f"{APP_BASE_URL}/frame?path={frame['image_path']}"

        uploaded_images.append({
            "file_id": idx,
            "url": frame_url
        })

    payload = {"images": uploaded_images}
    headers = {
        "api-token": REVISION_API_TOKEN,
        "Content-Type": "application/json"
    }

    print("Revision API Request Payload URLs:")
    for img in uploaded_images:
        print(f"file_id: {img['file_id']}, URL: {img['url']}")

    try:
        response = requests.post(MODERATION_API_URL, json=payload, headers=headers)
        print("Revision API Response:", response.text)

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
        is_video_forbidden = False
        frames_with_urls = []
        
        print("\nProcessing each frame result:")
        for result in moderation_result:
            try:
                file_id = result.get("file_id")
                if file_id is not None and 0 <= file_id < len(uploaded_images):
                    result_with_url = result.copy()
                    result_with_url["frame_url"] = uploaded_images[file_id]["url"]
                    frames_with_urls.append(result_with_url)

                    print(f"Frame {file_id}:")
                    print(f"  URL: {uploaded_images[file_id]['url']}")
                    print(f"  Is Forbidden: {result.get('is_forbidden')}")

                    if result.get("is_forbidden") is True:
                        forbidden_images.append(uploaded_images[file_id]["url"])
                        is_video_forbidden = True
            except Exception as e:
                print(f"Error processing frame {file_id}: {str(e)}")
                continue

        return {
            "final_result": {
                "is_forbidden": is_video_forbidden,
                "forbidden_images": forbidden_images,
                "total_forbidden_images": len(forbidden_images),
                "total_processed_images": len(moderation_result)
            },
            "frames_results": frames_with_urls,
            "debug_info": {
                "original_urls": [img["url"] for img in uploaded_images],
                "raw_revision_response": moderation_result
            }
        }

    except Exception as e:
        return {
            "error": f"Failed to call moderation API: {str(e)}"
        }

@app.get("/frame")
def get_frame(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}

@app.post("/check-video/")
async def check_video(video_url: VideoURL):
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    frame_urls = []
    revision_input = []

    for idx, frame in enumerate(result["frames"]):
        url = frame["url"]
        if not url.startswith("http"):
            url = f"{APP_BASE_URL}/frame?path={frame['image_path']}"

        frame_urls.append(url)
        revision_input.append({
            "file_id": idx,
            "url": url
        })

    headers = {
        "api-token": REVISION_API_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            MODERATION_API_URL,
            json={"images": revision_input},
            headers=headers
        )
        if response.status_code != 200:
            return {
                "error": "Revision API call failed",
                "status_code": response.status_code,
                "response": response.text
            }

        revision_results = response.json()

        is_video_forbidden = any(f.get("is_forbidden", False) for f in revision_results)

        for item in revision_results:
            fid = item.get("file_id")
            if fid is not None and fid < len(frame_urls):
                item["frame_url"] = frame_urls[fid]

        return {
            "is_video_forbidden": is_video_forbidden,
            "frame_count": len(frame_urls),
            "frame_urls": frame_urls,
            "revision_results": revision_results
        }

    except Exception as e:
        return {"error": f"Failed to call revision API: {str(e)}"}