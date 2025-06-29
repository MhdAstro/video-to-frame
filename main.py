# main.py

import asyncio
import cv2
import logging
import numpy as np
import os
import requests
import shutil
import tempfile
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import FileResponse

# --- بخش ۱: راه‌اندازی لاگر ---
# برای نمایش پیام‌های مربوط به فرآیندها در کنسول
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- بخش ۲: خواندن متغیرهای محیطی و پیکربندی ---
app = FastAPI(
    title="Video Analysis Service",
    description="سرویسی برای استخراج فریم‌های کلیدی از ویدیو و تحلیل آن‌ها",
    version="1.1.0"
)

# مقادیر ضروری (برنامه بدون اینها اجرا نمی‌شود)
REVISION_API_TOKEN = os.getenv("REVISION_API_TOKEN")
APP_BASE_URL = os.getenv("APP_BASE_URL")
MODERATION_API_URL = os.getenv("MODERATION_API_URL")

# بررسی وجود متغیرهای ضروری
if not all([REVISION_API_TOKEN, APP_BASE_URL, MODERATION_API_URL]):
    raise ValueError("Missing required environment variables: REVISION_API_TOKEN, APP_BASE_URL, MODERATION_API_URL")

# پارامترهای قابل تنظیم (با مقادیر پیش‌فرض)
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "30"))
SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", "3"))
SCENE_THRESHOLD = float(os.getenv("SCENE_THRESHOLD", "20.0"))
FRAME_RESIZE = (int(os.getenv("FRAME_RESIZE_WIDTH", "224")), int(os.getenv("FRAME_RESIZE_HEIGHT", "224")))
CLEANUP_DELAY_SECONDS = int(os.getenv("CLEANUP_DELAY_SECONDS", "600")) # <-- جدید: زمان تأخیر برای پاک‌سازی

class VideoURL(BaseModel):
    url: str

# --- بخش ۳: تابع پاک‌سازی پس‌زمینه ---
async def cleanup_files(directory_path: str):
    """
    پس از یک تأخیر مشخص، پوشه و محتویات آن را حذف می‌کند.
    """
    logger.info(f"زمان‌بندی برای پاک‌سازی پوشه: '{directory_path}' تا {CLEANUP_DELAY_SECONDS} ثانیه دیگر.")
    await asyncio.sleep(CLEANUP_DELAY_SECONDS)
    try:
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)
            logger.info(f"پوشه با موفقیت پاک شد: {directory_path}")
        else:
            logger.warning(f"پوشه برای پاک‌سازی یافت نشد (ممکن است قبلاً حذف شده باشد): {directory_path}")
    except Exception as e:
        logger.error(f"خطا در حین پاک‌سازی پوشه {directory_path}: {e}")

# --- بخش ۴: منطق اصلی استخراج فریم ---
# این تابع بدون تغییر باقی می‌ماند اما توسط اندپوینت‌های async فراخوانی می‌شود
def extract_frames_internal(video_url: str):
    """
    ویدیو را دانلود، فریم‌های کلیدی را استخراج و اطلاعات آن‌ها را برمی‌گرداند.
    """
    try:
        video_response = requests.get(video_url, stream=True, timeout=30)
        video_response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"خطا در دانلود ویدیو: {e}")
        return {"error": f"Unable to download video: {e}"}

    # ساخت پوشه موقت برای ذخیره فریم‌ها
    output_dir = tempfile.mkdtemp()

    # دانلود ویدیو در یک فایل موقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        try:
            for chunk in video_response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            video_path = tmp_file.name
        except Exception as e:
            shutil.rmtree(output_dir)
            logger.error(f"خطا در ذخیره ویدیو در فایل موقت: {e}")
            return {"error": f"Failed to save video chunk: {e}"}
            
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30 # مقدار پیش‌فرض برای جلوگیری از تقسیم بر صفر
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

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                scene_diff = np.mean(diff)
                if scene_diff > SCENE_THRESHOLD:
                    changes.append((frame_index, scene_diff, frame.copy()))
            
            prev_gray = gray
            frame_index += 1

        cap.release()

        top_changes = sorted(changes, key=lambda x: -x[1])[:MAX_FRAMES]
        output_frames_info = []
        url_prefix = f"{APP_BASE_URL}/frame?path="

        for i, (frame_idx, score, frame_original) in enumerate(top_changes):
            frame_filename = f"frame_{frame_idx:06d}.jpeg"
            frame_filepath = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_filepath, frame_original)
            
            output_frames_info.append({
                "file_id": i,
                "frame": frame_idx,
                "timestamp": round(frame_idx / fps, 2),
                "scene_diff": round(score, 2),
                "url": url_prefix + frame_filepath, # URL برای دسترسی خارجی
                "image_path": frame_filepath        # مسیر داخلی فایل
            })
        
        logger.info(f"استخراج {len(output_frames_info)} فریم کلیدی از ویدیو در پوشه {output_dir}")
        return {
            "frames": output_frames_info,
            "total": len(output_frames_info),
            "output_directory": output_dir
        }

    except Exception as e:
        logger.error(f"خطا در پردازش ویدیو: {e}")
        shutil.rmtree(output_dir) # در صورت خطا، پوشه موقت را پاک کن
        return {"error": str(e)}
    finally:
        # فایل ویدیوی دانلود شده همیشه حذف می‌شود
        if os.path.exists(video_path):
            os.unlink(video_path)
            logger.info(f"فایل ویدیوی موقت حذف شد: {video_path}")


# --- بخش ۵: اندپوینت‌های API ---

@app.get("/", summary="Health Check")
def health_check():
    """بررسی سلامت و فعال بودن سرویس."""
    return {"status": "ok", "message": "Video Analysis Service is running."}

@app.post("/extract-frames/", summary="استخراج فریم‌ها (با پاک‌سازی خودکار)")
async def extract_frames(video_url: VideoURL, background_tasks: BackgroundTasks):
    """
    فقط فریم‌ها را استخراج کرده و اطلاعاتشان را برمی‌گرداند.
    فریم‌ها پس از مدتی به‌صورت خودکار پاک خواهند شد.
    """
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    # زمان‌بندی پاک‌سازی پوشه فریم‌ها در پس‌زمینه
    if output_dir := result.get("output_directory"):
        background_tasks.add_task(cleanup_files, output_dir)
    
    return result

@app.post("/analyze-video/", summary="تحلیل کامل ویدیو (با پاک‌سازی خودکار)")
async def analyze_video(video_url: VideoURL, background_tasks: BackgroundTasks):
    """
    فریم‌ها را استخراج کرده، به سرویس تحلیل ارسال می‌کند و نتیجه نهایی را برمی‌گرداند.
    فایل‌های موقت پس از مدتی به‌صورت خودکار پاک خواهند شد.
    """
    result = extract_frames_internal(video_url.url)
    if "error" in result:
        return result

    # زمان‌بندی پاک‌سازی پوشه فریم‌ها در پس‌زمینه
    if output_dir := result.get("output_directory"):
        background_tasks.add_task(cleanup_files, output_dir)

    # آماده‌سازی داده برای ارسال به سرویس تحلیل
    payload = {
        "images": [{"file_id": f["file_id"], "url": f["url"]} for f in result["frames"]]
    }
    headers = {"api-token": REVISION_API_TOKEN, "Content-Type": "application/json"}

    try:
        response = requests.post(MODERATION_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        moderation_result = response.json()
    except requests.RequestException as e:
        logger.error(f"خطا در ارتباط با Moderation API: {e}")
        return {"error": f"Failed to call moderation API: {e}"}
    except ValueError: # JSONDecodeError
        return {"error": "Invalid JSON response from moderation API", "raw_response": response.text}

    # پردازش نتیجه تحلیل
    is_video_forbidden = any(r.get("is_forbidden") for r in moderation_result)
    forbidden_images = [
        f["url"] for r in moderation_result if r.get("is_forbidden")
        for f in result["frames"] if f["file_id"] == r.get("file_id")
    ]

    # اضافه کردن URL هر فریم به نتیجه برای راحتی دیباگ
    for res in moderation_result:
        if (file_id := res.get("file_id")) is not None:
            for frame in result["frames"]:
                if frame["file_id"] == file_id:
                    res["frame_url"] = frame["url"]
                    break

    return {
        "final_result": {
            "is_video_forbidden": is_video_forbidden,
            "forbidden_images_count": len(forbidden_images),
            "forbidden_images": forbidden_images,
        },
        "frames_results": moderation_result,
    }

@app.get("/frame", summary="ارائه یک فریم ذخیره شده")
def get_frame(path: str):
    """
    این اندپوینت یک فریم را از روی مسیر داخلی سرور برمی‌گرداند.
    سرویس تحلیل از این اندپوینت برای دسترسی به تصاویر استفاده می‌کند.
    """
    # جلوگیری از دسترسی به مسیرهای خارج از پوشه‌های موقت (برای امنیت)
    if not os.path.basename(os.path.dirname(path)).startswith("tmp"):
         return {"error": "Access denied"}, 403
    
    if os.path.exists(path):
        return FileResponse(path)
    
    logger.warning(f"درخواست برای فایلی که وجود ندارد: {path}")
    return {"error": "File not found"}, 404

# اندپوینت check-video حذف شد چون عملکرد آن کاملاً توسط analyze-video پوشش داده می‌شود
# و باعث تکرار کد شده بود. اگر همچنان به آن نیاز دارید، می‌توانید با منطقی مشابه
# analyze-video و افزودن تسک پس‌زمینه، آن را مجدداً اضافه کنید.