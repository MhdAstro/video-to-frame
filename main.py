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

class VideoURL(BaseModel):
    url: str

# ✅ توکن ثابت برای API بررسی محتوا
REVISION_API_TOKEN = "YwrdzYgYnMAGWyE18LVu1B4sbOz2qzpeo0g3dzKslFiCI0EMSdA0rxPue4YKDaYT"
UPLOAD_API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI1NTAiLCJqdGkiOiJjN2FiYTE2NTBkNzA3ZTg4Mjc5YzI4MTY4ZTczYzc4NzJkMzY1NWIwMGFjYjRkZGJhMDA1ZWE2MTU0ODhjZjFjZjExZjJjZWU1MmNhNzcxNyIsImlhdCI6MTc0NzY0MTkxNC42ODg0MjIsIm5iZiI6MTc0NzY0MTkxNC42ODg0MjYsImV4cCI6MTc3OTE3NzkxNC42NzIxNzQsInN1YiI6IjE2NTExNjc5Iiwic2NvcGVzIjpbIm9yZGVyLXByb2Nlc3NpbmciLCJ2ZW5kb3IucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2ZpbGUud3JpdGUiLCJjdXN0b21lci5wcm9maWxlLndyaXRlIiwiY3VzdG9tZXIucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2R1Y3Qud3JpdGUiLCJ2ZW5kb3IucHJvZHVjdC5yZWFkIiwiY3VzdG9tZXIub3JkZXIucmVhZCIsImN1c3RvbWVyLm9yZGVyLndyaXRlIiwidmVuZG9yLnBhcmNlbC5yZWFkIiwidmVuZG9yLnBhcmNlbC53cml0ZSIsImN1c3RvbWVyLndhbGxldC5yZWFkIiwiY3VzdG9tZXIud2FsbGV0LndyaXRlIiwiY3VzdG9tZXIuY2hhdC5yZWFkIiwiY3VzdG9tZXIuY2hhdC53cml0ZSJdLCJ1c2VyX2lkIjoxNjUxMTY3OX0.EbrqOaUaGI6wORC446IDclq4gg8j2mWhuVzHA82tph2PZ6Fnx2sPMMqCuhOSavSXX6Vuk6Pmfh_qMIl_zcAPXvBvgmi62or1BRPCqOZ9E-L0DUWdDIiY8tpU6Rxl5QkISCjS-K5dpgpj6aBwQYadYQKUxUN0JJ_usgNSeSXYfAUJvVxOO3ZhpSjZ9O4jEu2vPZSiS5gkOIw-Q8Erz9GHB21m_3h2r2XJvJEwJ6GfPuYVubfMlNFMfufpqHQUpRyov0OAS_wCMGJmA5jBYHxlt3GEAb-hU0eWP6Tg44y5XO65gaIF1vyLKu5tHZ1j-d6Oue3wolxb3NgTwZHTVsxR6pUrA6j90vunHLVSlE4uVD0QYB3R2PUKOA5tM6LWgu72d3ynnSRrBBXEpBMy-SFk0iESyOLKD2qCXcetRfRlDPBoKVjotavp2W0hU9GDthVzopsKQaD8YrW1zSWXPKRxgflod455bRZdeJo2dvJMhZAX8C7wTcGyJddcLO4Eq-bT7w7yJnBeSUEZtycqHVCD6mIZ_gq4jlVtYir4tnU5IKHpeMITkCaA1H9QcOr1VdGVngfrjqRSduATGA-IxW9VKeiHNYowZ6JQrbbXi0GDCKluPbGxji5WYV-kvRt3afzEiYx1vuDns58Xu8hkac8rdVrXbbpkIZyv46V6R1z4-o"

@app.get("/")
def health_check():
    return {"status": "ok"}

# تابع extract_frames_internal جدید که منطق انتخاب فریم‌ها را تغییر می‌دهد
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
        MAX_FRAMES = 30
        SKIP_FRAMES = 3
        SCENE_THRESHOLD = 20.0
        FRAME_RESIZE = (224, 224)

        prev_gray = None
        frame_index = 0
        changes = [] # لیست برای ذخیره فریم‌هایی با تغییرات صحنه

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % SKIP_FRAMES != 0:
                frame_index += 1
                continue

            resized = cv2.resize(frame, FRAME_RESIZE) # تغییر اندازه فریم برای تحلیل سریعتر
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            scene_diff = 0.0

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)
                if scene_diff > SCENE_THRESHOLD:
                    changes.append((frame_index, scene_diff, frame.copy())) # ذخیره فریم اصلی

            prev_gray = gray
            frame_index += 1

        cap.release()
        os.unlink(video_path)

        # انتخاب ۳۰ فریم با بیشترین تغییر
        # اگر تعداد فریم‌های با تغییر کمتر از 30 باشد، همه آن‌ها انتخاب می‌شوند.
        top_changes = sorted(changes, key=lambda x: -x[1])[:MAX_FRAMES]
        output_frames_info = []
        url_prefix = "https://video-check.darkube.app/frame?path="

        for i, (frame_idx, score, frame_original) in enumerate(top_changes):
            frame_filename = f"frame_{frame_idx:06d}.jpeg"
            frame_filepath = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame_original) # ذخیره فریم اصلی با اندازه کامل
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
            import shutil
            shutil.rmtree(output_dir)
        return {"error": str(e)}

@app.post("/extract-frames/")
async def extract_frames(video_url: VideoURL):
    # این اندپوینت از تابع extract_frames_internal جدید استفاده می‌کند
    return extract_frames_internal(video_url.url)

@app.post("/analyze-video/")
async def analyze_video(video_url: VideoURL):
    # این اندپوینت از تابع extract_frames_internal جدید استفاده می‌کند
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    # Prepare the images data directly from the frame URLs
    uploaded_images = []
    for idx, frame in enumerate(result["frames"]):
        # Ensure the URL is properly formatted
        frame_url = frame["url"]
        if not frame_url.startswith('http'):
            frame_url = f"https://video-check.darkube.app/frame?path={frame['image_path']}"

        uploaded_images.append({
            "file_id": idx,
            "url": frame_url
        })

    # Send to revision for content moderation
    payload = {"images": uploaded_images}
    moderation_api_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {
        "api-token": REVISION_API_TOKEN,
        "Content-Type": "application/json"
    }

    # Debug print with full URLs
    print("Revision API Request Payload URLs:")
    for img in uploaded_images:
        print(f"file_id: {img['file_id']}, URL: {img['url']}")

    try:
        response = requests.post(moderation_api_url, json=payload, headers=headers)
        print("Revision API Response:", response.text)  # Debug print raw response

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

        # Add URLs to moderation results and debug prints
        frames_with_urls = []
        print("\nProcessing each frame result:")
        for result in moderation_result:
            try:
                file_id = result.get("file_id")
                if file_id is not None and 0 <= file_id < len(uploaded_images):
                    # Add URL to the result
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
    # این اندپوینت از تابع extract_frames_internal جدید استفاده می‌کند
    # استخراج فریم‌ها
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    frame_urls = []
    revision_input = []

    for idx, frame in enumerate(result["frames"]):
        url = frame["url"]
        if not url.startswith("http"):
            url = f"https://video-check.darkube.app/frame?path={frame['image_path']}"

        frame_urls.append(url)
        revision_input.append({
            "file_id": idx,
            "url": url
        })

    # ارسال فریم‌ها به API بررسی
    revision_api_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {
        "api-token": REVISION_API_TOKEN ,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            revision_api_url,
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

        # آیا حتی یک فریم هم ممنوع بوده؟
        is_video_forbidden = any(f.get("is_forbidden", False) for f in revision_results)

        # اضافه کردن آدرس فریم‌ها به خروجی نهایی
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