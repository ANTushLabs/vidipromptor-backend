from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import uuid
import json
import threading
import time
import cv2
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from core.engine import (
    generate_prompt_for_video, APP_DATA_DIR, API_KEY_FILE, GROQ_KEY_FILE, 
    CUSTOM_PROMPT_FILE, CUSTOM_CONFIG_FILE, CUSTOM_KEY_FILE, load_settings, save_settings, configure_proxy,
    get_instructional_prompt, get_default_prompt
)
# tkinter removed for cloud deployment compatibility

app = FastAPI(title="VidiPromptor Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DIRECTORIES ---
UPLOAD_DIR = os.path.join(APP_DATA_DIR, "web_uploads")
DESKTOP_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "VidiPromptor_Output")
THUMBS_DIR = os.path.join(APP_DATA_DIR, "thumbs")

if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)
if not os.path.exists(DESKTOP_OUTPUT_DIR): os.makedirs(DESKTOP_OUTPUT_DIR)
if not os.path.exists(THUMBS_DIR): os.makedirs(THUMBS_DIR)

# --- STATE ---
TASKS = {}
TASKS = {}
# Initialize Executor with saved setting or default to 1 (safe default)
_init_settings = load_settings()
_init_threads = _init_settings.get("thread_limit", 1)
EXECUTOR = ThreadPoolExecutor(max_workers=_init_threads)

class TaskStatus:
    def __init__(self, name=""):
        self.name = name
        self.status = "queued"
        self.result = None
        self.error = None
        self.output_file = None
        self.log = [] 
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()

# --- ROUTES ---

@app.get("/config")
def get_config():
    return {
        "gemini_configured": os.path.exists(API_KEY_FILE) and os.path.getsize(API_KEY_FILE) > 0,
        "groq_configured": os.path.exists(GROQ_KEY_FILE) and os.path.getsize(GROQ_KEY_FILE) > 0,
        "custom_configured": os.path.exists(CUSTOM_KEY_FILE) and os.path.getsize(CUSTOM_KEY_FILE) > 0,
        "settings": load_settings(),
        "output_dir": DESKTOP_OUTPUT_DIR # Just as a fallback or info
    }

@app.post("/update-threads")
def update_threads(threads: int = Body(..., embed=True)):
    global EXECUTOR
    EXECUTOR._max_workers = threads
    
    # Persist setting
    s = load_settings()
    s["thread_limit"] = threads
    save_settings(s)
    
    return {"status": "updated", "threads": threads}

@app.post("/save-settings")
def save_app_settings(payload: dict = Body(...)):
    s = load_settings()
    s.update({
        "proxy_enabled": payload.get("proxy_enabled"),
        "proxy_url": payload.get("proxy_url"),
        "output_folder": payload.get("output_folder"),
        "export_format": payload.get("export_format")
    })
    save_settings(s)
    configure_proxy(s)
    return {"status": "saved"}

@app.get("/browse-folder")
def browse_folder():
    # Folder browsing not available in cloud deployment
    # Users should use the download feature instead
    return {"path": "", "error": "Folder browsing is not available in web deployment. Files are saved automatically and can be downloaded."}

@app.get("/get-keys")
def get_existing_keys(provider: str):
    target_file = API_KEY_FILE
    if provider.lower() == "gemini": target_file = API_KEY_FILE
    elif provider.lower() == "groq": target_file = GROQ_KEY_FILE
    elif provider.lower() == "custom": target_file = CUSTOM_KEY_FILE
    print(f"DEBUG GET KEYS: Provider={provider}, Path={target_file}, Exists={os.path.exists(target_file)}")
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            content = f.read().strip()
            print(f"DEBUG CONTENT LEN: {len(content)}")
            return {"keys": content}
    return {"keys": ""}

@app.post("/save-keys")
def save_keys(payload: dict = Body(...)):
    target_file = API_KEY_FILE
    if payload.get("provider").lower() == "gemini": target_file = API_KEY_FILE
    elif payload.get("provider").lower() == "groq": target_file = GROQ_KEY_FILE
    elif payload.get("provider").lower() == "custom": target_file = CUSTOM_KEY_FILE
    with open(target_file, "w") as f:
        f.write(payload.get("key").strip())
    return {"status": "saved"}

@app.get("/custom-config")
def get_custom_config():
    if os.path.exists(CUSTOM_CONFIG_FILE):
        try:
            with open(CUSTOM_CONFIG_FILE, "r") as f: return json.load(f)
        except: pass
    return {"base_url": "https://api.openai.com/v1", "model_id": "gpt-4-turbo"}

@app.post("/custom-config")
def save_custom_config(payload: dict = Body(...)):
    with open(CUSTOM_CONFIG_FILE, "w") as f:
        json.dump(payload, f)
    return {"status": "saved"}

@app.get("/prompt")
def get_sys_prompt():
    return {"prompt": get_instructional_prompt()}

@app.post("/prompt")
def save_sys_prompt(payload: dict = Body(...)):
    with open(CUSTOM_PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(payload.get("prompt").strip())
    return {"status": "saved"}

@app.post("/reset-prompt")
def reset_sys_prompt():
    d = get_default_prompt()
    with open(CUSTOM_PROMPT_FILE, "w", encoding="utf-8") as f: f.write(d)
    return {"prompt": d}

def generate_thumb(video_path, thumb_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Resize to reduce size (e.g., width 320)
            h, w = frame.shape[:2]
            scale = 320 / w
            frame = cv2.resize(frame, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(thumb_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    except Exception as e:
        print(f"Thumb Gen Error: {e}")

@app.post("/upload")
async def upload_v(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, f"{task_id}_{safe_name}")
    with open(path, "wb") as b: shutil.copyfileobj(file.file, b)
    
    TASKS[task_id] = TaskStatus(name=file.filename)
    TASKS[task_id].video_path = path
    TASKS[task_id].status = "ready"
    
    # Generate Thumb
    thumb_path = os.path.join(THUMBS_DIR, f"{task_id}.jpg")
    background_tasks.add_task(generate_thumb, path, thumb_path)
    
    return {"task_id": task_id}

@app.get("/thumbnail/{task_id}")
def get_thumbnail(task_id: str):
    thumb_path = os.path.join(THUMBS_DIR, f"{task_id}.jpg")
    if os.path.exists(thumb_path):
        return FileResponse(thumb_path)
    # Return 404 or a clear fallback
    return HTTPException(404)

@app.post("/start-task/{task_id}")
def start_t(task_id: str, background_tasks: BackgroundTasks, provider: str = Body(..., embed=True), model: str = Body(..., embed=True)):
    task = TASKS.get(task_id)
    if not task: raise HTTPException(404)
    
    # Load ALL keys
    key_file = API_KEY_FILE
    if provider == "Gemini": key_file = API_KEY_FILE
    elif provider == "Groq": key_file = GROQ_KEY_FILE
    elif provider == "Custom": key_file = CUSTOM_KEY_FILE
    api_keys = []
    if os.path.exists(key_file):
        with open(key_file) as f:
            api_keys = [l.strip() for l in f if l.strip()]
    
    if not api_keys:
        task.status = "failed"
        task.error = "No Keys Configured"
        return {"status": "failed"}

    task.status = "queued"
    # Pass LIST of keys to engine
    background_tasks.add_task(run_wrapper, task_id, task.video_path, api_keys, provider, model)
    return {"status": "queued"}

def run_wrapper(task_id, path, keys, prov, mod):
    EXECUTOR.submit(run_engine_logic, task_id, path, keys, prov, mod)

def run_engine_logic(task_id, video_path, api_keys, provider, model):
    if task_id not in TASKS:
        return # Task was likely cleared/reset
    task = TASKS[task_id]
    try:
        # Update status to processing ONLY when thread actually starts
        task.status = "processing"
        
        task.log.append("Starting engine...")
        # Engine now handles rotation internally
        result = generate_prompt_for_video(video_path, api_keys, provider, model, task.pause_event, task.cancel_event)
        
        if result.get("status") == "success":
            task.result = result
            task.status = "completed"
            task.log.append("Generation Successful.")
            
            # --- AUTO SAVE LOGIC (UPDATED) ---
            try:
                settings = load_settings()
                out_folder = settings.get("output_folder", "").strip()
                fmt = settings.get("export_format", "txt")
                
                # Determine Format
                ext = ".json" if fmt == "json" else ".txt"
                content = result["prompt"] # this is JSON string
                
                if fmt != "json": 
                    # If TXT, maybe we want it exactly as is (which is JSON string). 
                    pass 

                original_base = os.path.splitext(task.name)[0]
                save_name = f"{original_base}_prompt{ext}"
                
                # Determine Directory
                target_dir = out_folder if (out_folder and os.path.exists(out_folder)) else DESKTOP_OUTPUT_DIR
                if not os.path.exists(target_dir): os.makedirs(target_dir, exist_ok=True)
                
                final_path = os.path.join(target_dir, save_name)
                
                # Avoid Overwrite (Simple Timestamp)
                if os.path.exists(final_path):
                    save_name = f"{original_base}_prompt_{int(time.time())}{ext}"
                    final_path = os.path.join(target_dir, save_name)

                with open(final_path, "w", encoding="utf-8") as f:
                    f.write(content)
                
                task.output_file = final_path
                task.log.append(f"Saved to: {final_path}")
            except Exception as save_err:
                task.error = f"Saved locally failed: {save_err}"
                
        elif result.get("status") == "cancelled":
            task.status = "cancelled"
            task.log.append("Cancelled by user.")
        else:
            task.status = "failed"
            task.error = result.get("prompt", "Error")
            task.log.append(f"Failed: {task.error}")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.log.append(f"Critical Error: {str(e)}")
    finally:
        pass

@app.post("/task/{task_id}/retry")
def retry_task(task_id: str):
    task = TASKS.get(task_id)
    if not task: raise HTTPException(404)
    task.status = "ready"
    task.error = None
    task.result = None
    task.log = []
    task.cancel_event.clear()
    return {"status": "ready"}

@app.get("/task/{task_id}")
def status(task_id: str):
    t = TASKS.get(task_id)
    if not t: raise HTTPException(404)
    return {
        "status": t.status, 
        "result": t.result, 
        "error": t.error, 
        "logs": t.log[-1] if t.log else "",
        "download_ready": t.output_file is not None
    }

@app.get("/tasks")
def get_all_tasks():
    return [
        {
            "task_id": tid,
            "name": t.name,
            "status": t.status,
            "video_path": t.video_path,
            "result": t.result,
            "error": t.error,
            "logs": t.log[-1] if t.log else "",
            "download_ready": t.output_file is not None
        }
        for tid, t in TASKS.items()
    ]

@app.get("/tasks/summary")
def get_summary():
    total = len(TASKS)
    completed = len([t for t in TASKS.values() if t.status == "completed"])
    failed = len([t for t in TASKS.values() if t.status == "failed"])
    processing = len([t for t in TASKS.values() if t.status == "processing"])
    queued = len([t for t in TASKS.values() if t.status == "queued" or t.status == "ready"])
    
    # Check if any task is paused (naive check: if first processing task is paused)
    is_paused = False
    for t in TASKS.values():
        if t.status == "processing" and t.pause_event.is_set():
            is_paused = True
            break
            
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "queued": queued,
        "is_paused": is_paused
    }

@app.post("/tasks/global/pause")
def global_pause():
    # Toggle based on first active task or just toggle all
    should_pause = True
    # If any is paused, we resume all. If none paused, we pause all.
    any_paused = any(t.pause_event.is_set() for t in TASKS.values() if t.status == "processing")
    
    for t in TASKS.values():
        if t.status == "processing":
            if any_paused: t.pause_event.clear() # Resume
            else: t.pause_event.set() # Pause
            
    return {"status": "resumed" if any_paused else "paused"}

@app.post("/tasks/global/reset")
def global_reset():
    # Cancel all
    for t in TASKS.values():
        t.cancel_event.set()
    
    # Wait briefly? No, fire and forget cleanup
    TASKS.clear()
    
    # Clear Upload Dir
    for f in os.listdir(UPLOAD_DIR):
        fp = os.path.join(UPLOAD_DIR, f)
        try:
            if os.path.isfile(fp): os.unlink(fp)
        except: pass
        
    return {"status": "reset"}

@app.get("/download/{task_id}")
def dl(task_id: str):
    t = TASKS.get(task_id)
    if not t or not t.output_file: raise HTTPException(404)
    return FileResponse(t.output_file, filename=os.path.basename(t.output_file))

@app.post("/task/{task_id}/pause")
def pause(task_id: str):
    t = TASKS.get(task_id) 
    if t: 
        if t.pause_event.is_set(): t.pause_event.clear()
        else: t.pause_event.set()
    return {"status": "ok"}

@app.post("/task/{task_id}/cancel")
def cancel(task_id: str):
    t = TASKS.get(task_id)
    if t: t.cancel_event.set()
    return {"status": "ok"}

@app.delete("/task/{task_id}")
def delete_task(task_id: str):
    if task_id in TASKS:
        t = TASKS[task_id]
        # Try to stop if running
        t.cancel_event.set()
        
        # Cleanup files
        try:
            if t.video_path and os.path.exists(t.video_path):
                os.remove(t.video_path)
            thumb_path = os.path.join(THUMBS_DIR, f"{task_id}.jpg")
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
        except Exception as e:
            print(f"Cleanup error: {e}")

        del TASKS[task_id]
        return {"status": "deleted"}
    raise HTTPException(404, "Task not found")


