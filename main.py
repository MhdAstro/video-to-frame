# main.py

# --- Dependencies ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import requests
import cv2
import numpy as np
import tempfile
import os
import shutil

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Analysis Service",
    description="Extracts keyframes from a video and sends them for content moderation.",
    version="1.0.1",
)

# --- Pydantic Input Models ---
class VideoURL(BaseModel):
    url: str

# --- Constants & Tokens ---
# TODO: Move tokens to environment variables for better security.
REVISION_API_TOKEN = "YwrdzYgYnMAGWyE18LVu1B4sbOz2qzpeo0g3dzKslFiCI0EMSdA0rxPue4YKDaYT"
UPLOAD_API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI1NTAiLCJqdGkiOiJjN2FiYTE2NTBkNzA3ZTg4Mjc5YzI4MTY4ZTczYzc4NzJkMzY1NWIwMGFjYjRkZGJhMDA1ZWE2MTU0ODhjZjFjZjExZjJjZWU1MmNhNzcxNyIsImlhdCI6MTc0NzY0MTkxNC42ODg0MjIsIm5iZiI6MTc0NzY0MTkxNC42ODg0MjYsImV4cCI6MTc3OTE3NzkxNC42NzIxNzQsInN1YiI6IjE2NTExNjc5Iiwic2NvcGVzIjpbIm9yZGVyLXByb2Nlc3NpbmciLCJ2ZW5kb3IucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2ZpbGUud3JpdGUiLCJjdXN0b21lci5wcm9maWxlLndyaXRlIiwiY3VzdG9tZXIucHJvZmlsZS5yZWFkIiwidmVuZG9yLnByb2R1Y3Qud3JpdGUiLCJ2ZW5kb3IucHJvZHVjdC5yZWFkIiwiY3VzdG9tZXIub3JkZXIucmVhZCIsImN1c3RvbWVyLm9yZGVyLndyaXRlIiwidmVuZG9yLnBhcmNlbC5yZWFkIiwidmVuZG9yLnBhcmNlbC53cml0ZSIsImN1c3RvbWVyLndhbGxldC5yZWFkIiwiY3VzdG9tZXIud2FsbGV0LndyaXRlIiwiY3VzdG9tZXIuY2hhdC5yZWFkIiwiY3VzdG9tZXIuY2hhdC53cml0ZSJdLCJ1c2VyX2lkIjoxNjUxMTY3OX0.EbrqOaUaGI6wORC446IDclq4gg8j2mWhuVzHA82tph2PZ6Fnx2sPMMqCuhOSavSXX6Vuk6Pmfh_qMIl_zcAPXvBvgmi62or1BRPCqOZ9E-L0DUWdDIiY8tpU6Rxl5QkISCjS-K5dpgpj6aBwQYadYQKUxUN0JJ_usgNSeSXYfAUJvVxOO3ZhpSjZ9O4jEu2vPZSiS5gkOIw-Q8Erz9GHB21m_3h2r2XJvJEwJ6GfPuYVubfMlNFMfufpqHQUpRyov0OAS_wCMGJmA5jBYHxlt3GEAb-hU0eWP6Tg44y5XO65gaIF1vyLKu5tHZ1j-d6Oue3wolxb3NgTwZHTVsxR6pUrA6j90vunHLVSlE4uVD0QYB3R2PUKOA5tM6LWgu72d3ynnSRrBBXEpBMy-SFk0iESyOLKD2qCXcetRfRlDPBoKVjotavp2W0hU9GDthVzopsKQaD8YrW1zSWXPKRxgflod455bRZdeJo2dvJMhZAX8C7wTcGyJddcLO4Eq-bT7w7yJnBeSUEZtycqHVCD6mIZ_gq4jlVtYir4tnU5IKHpeMITkCaA1H9QcOr1VdGVngfrjqRSduATGA-IxW9VKeiHNYowZ6JQrbbXi0GDCKluPbGxji5WYV-kvRt3afzEiYx1vuDns58Xu8hkac8rdVrXbbpkIZyv46V6R1z4-o"


# --- API Endpoints ---

@app.get("/", summary="Health Check", tags=["Public"])
def health_check():
    """A simple endpoint to check if the service is running."""
    return {"status": "ok"}


def extract_frames_internal(video_url: str):
    """
    Core video processing function.
    Downloads a video, extracts keyframes based on scene changes, and saves them.
    """
    try:
        # Stream download to avoid high memory usage
        video_response = requests.get(video_url, stream=True, timeout=30)
        video_response.raise_for_status()  # Raise exception for bad status codes
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")

    output_dir = tempfile.mkdtemp()
    
    # Use a temporary file to store the downloaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        for chunk in video_response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        video_path = tmp_file.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        # --- Frame Extraction Algorithm Params ---
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        MAX_FRAMES = 30         # Max number of frames to extract
        SKIP_FRAMES = 3         # Process every Nth frame for performance
        SCENE_THRESHOLD = 20.0  # Min difference to be considered a scene change

        prev_gray = None
        frame_index = 0
        changes = []

        # --- Main Frame Processing Loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video

            # Skip frames for performance
            if frame_index % SKIP_FRAMES != 0:
                frame_index += 1
                continue

            # Calculate scene change score
            gray = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                scene_diff = np.mean(cv2.absdiff(gray, prev_gray))
                if scene_diff > SCENE_THRESHOLD:
                    # Store the original, full-resolution frame
                    changes.append((frame_index, scene_diff, frame.copy()))

            prev_gray = gray
            frame_index += 1

        # Get top N frames with the highest change score
        top_changes = sorted(changes, key=lambda item: -item[1])[:MAX_FRAMES]
        
        output_frames_info = []
        url_prefix = "https://video-analysis.darkube.app/frame?path="

        # Save selected frames and build the output data
        for i, (frame_idx, score, frame_original) in enumerate(top_changes):
            frame_filepath = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpeg")
            cv2.imwrite(frame_filepath, frame_original)

            output_frames_info.append({
                "file_id": i,
                "timestamp": round(frame_idx / fps, 2),
                "url": url_prefix + frame_filepath,
                "image_path": frame_filepath
            })

        return {
            "frames": output_frames_info,
            "output_directory": output_dir
        }

    except Exception as e:
        # If any error occurs, clean up the created directory
        if 'output_dir' in locals() and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise HTTPException(status_code=500, detail=f"Internal error during video processing: {e}")

    finally:
        # --- Cleanup ---
        # This block always runs to ensure temp files are deleted
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)


@app.post("/check-video/", summary="Full Video Analysis Workflow", tags=["Video Analysis"])
async def check_video(video_url: VideoURL):
    """
    Main endpoint: extracts frames and sends them for content moderation.
    """
    extraction_result = extract_frames_internal(video_url.url)
    
    # If no significant frames were found, no need to call the API
    if not extraction_result["frames"]:
        shutil.rmtree(extraction_result["output_directory"])
        return {
            "is_video_forbidden": False,
            "message": "No significant frames found for analysis."
        }
    
    # Prepare payload for the moderation API
    revision_input = [{"file_id": frame["file_id"], "url": frame["url"]} for frame in extraction_result["frames"]]
    
    revision_api_url = "https://revision.basalam.com/api_v1.0/validation/image/hijab-detector/bulk"
    headers = {"api-token": REVISION_API_TOKEN, "Content-Type": "application/json"}

    try:
        # Call the external moderation API
        response = requests.post(revision_api_url, json={"images": revision_input}, headers=headers, timeout=45)
        response.raise_for_status()
        revision_results = response.json()

        # Check if any frame was marked as forbidden
        is_video_forbidden = any(item.get("is_forbidden", False) for item in revision_results)

        # Add frame URLs back to the results for easier debugging
        for item in revision_results:
            file_id = item.get("file_id")
            if file_id is not None and file_id < len(revision_input):
                item["frame_url"] = revision_input[file_id]["url"]

        return {
            "is_video_forbidden": is_video_forbidden,
            "frame_count": len(revision_input),
            "revision_results": revision_results
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Moderation API call failed: {e}")
    finally:
        # Cleanup the temporary frame directory after processing
        shutil.rmtree(extraction_result["output_directory"])


@app.get("/frame", summary="Serve a saved frame", tags=["Utilities"])
def get_frame(path: str):
    """
    Serves a frame image from a temporary path. Used by the moderation API.
    NOTE: In production, access to this should be secured.
    """
    # Security check: prevent directory traversal attacks
    if not os.path.exists(path) or not path.startswith(tempfile.gettempdir()):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    
    return FileResponse(path)