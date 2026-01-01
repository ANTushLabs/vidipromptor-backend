import os
import time
import json
import base64
import logging
import cv2
import numpy as np
from .utils import APP_DATA_DIR, LOG_QUEUE, setup_logging

# Load API keys and settings paths
API_KEY_FILE = os.path.join(APP_DATA_DIR, "api_keys.txt")
GROQ_KEY_FILE = os.path.join(APP_DATA_DIR, "groq_keys.txt")
CUSTOM_PROMPT_FILE = os.path.join(APP_DATA_DIR, "custom_prompt.txt")
CUSTOM_CONFIG_FILE = os.path.join(APP_DATA_DIR, "custom_config.json")
CUSTOM_KEY_FILE = os.path.join(APP_DATA_DIR, "custom_key.txt")
SETTINGS_FILE = os.path.join(APP_DATA_DIR, "settings.json")

# Constants
PROVIDERS = ["Gemini", "Groq", "Custom"]
MODELS = {
    "Gemini": ["gemini-2.5-flash"],
    "Groq": ["llama-4-scout", "llama-4-maverick"]
}
MODEL_IDS = {
    "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"proxy_enabled": False, "proxy_url": ""}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f)

def configure_proxy(settings):
    if settings.get("proxy_enabled") and settings.get("proxy_url"):
        os.environ["HTTP_PROXY"] = settings["proxy_url"]
        os.environ["HTTPS_PROXY"] = settings["proxy_url"]
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

def get_default_prompt():
    return """
# SYSTEM ROLE & OBJECTIVE
You are an expert Video Replication Specialist and Commercial Cinematographer acting as the prompt engine for Google Veo 3.1.
Your goal is to accept a raw video concept from the user and convert it into a hyper-detailed, physically accurate JSON prompt (Layer 2) based on the "Veo3_1_Universal_Commercial_Framework".

# CORE DIRECTIVES (NON-NEGOTIABLE)
1. **Output Format:** You must output **ONLY valid, raw JSON**. Do not include markdown code blocks (like ```json), do not include introductory text, and do not include explanations.
2. **IP Sanitization:** Replace ALL specific Brands, Logos, and Faces with generic, high-end equivalents (e.g., "Nike" -> "High-performance athletic gear").
3. **Physics Engine:** Apply "High-Velocity Physics". Heavy objects must move with explosive power and snap. No slow-motion floating unless specified.
4. **Audio Hallucination:** You MUST invent specific internal audio cues (e.g., "Heavy bass thud at 0:03") inside the `audio_hallucination_guidance` field to guide the visual rhythm.
5. **Biological Lock:** Ensure strictly accurate anatomy (2 arms, 2 legs, 5 fingers). No morphing.

# JSON TEMPLATE STRUCTURE
Fill out the following JSON structure strictly based on the user's input. Replace the bracketed text `[...]` with your generated details:

{
  "_context_metadata": {
    "source_analysis": {
      "medium": "Identify the core medium (e.g., Live-Action Photorealistic, CGI, Stop Motion).",
      "style": "Define the production level (e.g., High-End Commercial, Cinematic Documentary).",
      "mood": "[Describe the emotional tone: e.g., Intense, Energetic, Powerful]"
    },
    "narrative_summary": "[CRITICAL: A single sentence summarizing the action and conflict, sanitized of any specific brands.]"
  },
  
  "technical_specifications": {
    "audio_hallucination_guidance": "[Internal Logic: Describe the SOUND the AI should 'imagine' to time the visuals accurately. e.g., 'Heavy bass impact at 0:02 synchronizing with the slam'.]"
  },

  "single_shot_spec": {
    "camera_systems": {
      "shot_type_and_angle": "[Describe exact angle: e.g., Low Angle, Drone Top-Down]",
      "movement": "[CRITICAL FOR ACTION: Use 'Reactive Handheld' for high-energy scenes. The camera must shake slightly with every impact.]",
      "stability": "Low stability (Action Mode). Avoid tripod/locked-off look.",
      "lens_characteristics": "[Specify lens: e.g., '35mm Wide Angle' or '85mm Portrait lens']",
      "focus_and_dof": "[Describe Depth of Field: e.g., 'Shallow DoF, subject sharp, background creamy bokeh']",
      "frame_rate_and_shutter": "24fps, 180-degree shutter"
    },
    
    "lighting_and_color": {
      "lighting_scheme": "[Describe quality: e.g., 'High-key outdoor', 'Rembrandt lighting', 'Neon Cyberpunk']",
      "source_motivation": "[Where is the light coming from? e.g., 'Sunlight (Backlight)', 'Stadium lights']",
      "contrast_ratio": "[Light vs Dark: e.g., 'High contrast / Chiaroscuro']",
      "color_palette": "[Dominant colors: e.g., 'Teal and Orange', 'Vibrant Green and Blue']",
      "color_grade": "[Final look: e.g., 'Bleach bypass', 'Commercial High-Contrast']",
      "image_texture": "[Grain/Noise level: e.g., 'Clean digital', 'Subtle Film Grain']"
    },

    "physics_and_simulation": {
      "weight_and_gravity": "[CRITICAL: Objects have mass but are moved with HIGH VELOCITY. Subject overpowers gravity. Do NOT make motion slow/sluggish.]",
      "contact_dynamics": "[How objects hit surfaces: e.g., 'Violent impact causing dust rise', 'Water splashes explosively']",
      "momentum_and_inertia": "[Fast, snapping momentum. e.g., 'Whip-like motion, no floating']"
    },
    
    "environment_and_set": {
      "location": "[Generic description of setting. Avoid specific landmarks.]",
      "set_dressing": "[Generic props: e.g., 'Weights, mats', 'Street signs, trash cans']"
    },
    
    "principal_elements": [
      {
        "element_id": "Main_Subject",
        "description": "[Physical details: Age, Gender, Build, Color]",
        "attire_or_surface": "[Detailed material description. NO LOGOS.]",
        "surface_fidelity": "[CRITICAL: Define exact material physics. e.g., 'Matte carbon fiber', 'Porous skin with sweat'. Use 'Imperfect Realism'.]",
        "micro_imperfections": "[CRITICAL FOR REALISM: Add flaws. e.g., 'Visible sweat beads, heavy breathing', 'Road grime']",
        "action_and_expression": "[Use explosive, active verbs. e.g., 'Slamming', 'Sprinting', 'Drifting'. Avoid passive states.]"
      }
    ],
    
    "interaction_dynamics": {
      "element_relationships": "[How elements affect each other]",
      "spatial_composition": {
        "foreground_layer": "[Elements closest to lens (e.g., 'Blurred sparks/dust') for parallax]",
        "background_separation": "Deep depth of field ensuring the subject pops."
      },
      "foreground_background_link": "[Connection between layers]",
      "blocking": "[Positioning in frame]"
    },
    
    "pacing": {
      "internal_pacing": "Frenetic and fast. High BPM rhythm. No unintended slow motion."
    }
  },

  "generation_control": {
    "master_instruction": "Generate a photorealistic video with high-end commercial production value. Combine REALISTIC WEIGHT with EXPLOSIVE SPEED. Ensure the camera reacts to the energy of the scene. Keep content generic and free of specific IP.",
    "structural_integrity": {
      "biological_lock": "CRITICAL: Enforce strict anatomical correctness. Humans must have exactly 2 arms, 2 legs, and 5 distinct fingers per hand. No morphing.",
      "identity_consistency": "Generic High-End Look only."
    },
    "negative_prompt": {
        "exclude_ip_and_content": "Brand logos, text overlays, watermarks, trademarked designs, copyrighted faces.",
        "exclude_physics_errors": "Slow motion, floaty objects, weightlessness, sliding feet, clipping textures, disappearing limbs, morphing objects.",
        "exclude_aesthetic_errors": "Oversaturated colors, cartoonish look, distorted anatomy, bad hands, blurry main subject, static camera."
    }
  }
}
"""

def get_instructional_prompt():
    if os.path.exists(CUSTOM_PROMPT_FILE) and os.path.getsize(CUSTOM_PROMPT_FILE) > 0:
        with open(CUSTOM_PROMPT_FILE, "r", encoding='utf-8') as f: return f.read().strip()
    return get_default_prompt()

# Updated to accept LIST of keys
def generate_prompt_for_video(video_path, api_keys, provider, model_name, pause_event, cancel_event):
    if cancel_event.is_set(): return {"status": "cancelled", "video_path": video_path}
    
    # Ensure api_keys is a list
    if isinstance(api_keys, str): api_keys = [api_keys]
    if not api_keys: return {"status": "failed", "prompt": "No API Keys Provided"}

    vidcap = None
    final_frames = []

    try:
        # --- PHASE 1: FRAME EXTRACTION (Once per video) ---
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened(): raise IOError("Cannot open video")
        
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        BLUR_THRESHOLD = 110.0
        SCENE_THRESHOLD = 0.35
        SKIP_CHECK = 5
        target_count = 5 if provider == "Groq" else 20
        
        candidates = []
        fallback_candidates = []
        prev_hist = None
        curr = 0
        
        while curr < total_frames:
            if cancel_event.is_set(): break
            while pause_event.is_set(): time.sleep(0.5)
            
            # vidcap.set(cv2.CAP_PROP_POS_FRAMES, curr) # optimization: removed valid seek
            success, img = vidcap.read()
            if not success: break
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            is_sharp = False

            # Track how many frames we physically advanced validly in this iteration
            frames_advanced = 1 # We did one success, img = vidcap.read()

            if sharpness < BLUR_THRESHOLD:
                found_sharp = False
                best_neighbor_sharpness = sharpness
                best_neighbor_img = img
                for _ in range(3):
                    success_n, next_img = vidcap.read()
                    if not success_n: break
                    frames_advanced += 1
                    
                    gray_next = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
                    next_sharp = cv2.Laplacian(gray_next, cv2.CV_64F).var()
                    if next_sharp > best_neighbor_sharpness:
                        best_neighbor_sharpness = next_sharp
                        best_neighbor_img = next_img
                    if next_sharp > BLUR_THRESHOLD:
                        img = next_img
                        sharpness = next_sharp
                        found_sharp = True
                        is_sharp = True
                        break
                if not found_sharp:
                     h, w = best_neighbor_img.shape[:2]
                     if max(h, w) > 640:
                        scale = 640 / max(h, w)
                        best_neighbor_img = cv2.resize(best_neighbor_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                     _, buf = cv2.imencode(".jpg", best_neighbor_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                     fallback_candidates.append((best_neighbor_sharpness, buf.tobytes()))
                     
                     # Compensation: We advanced `frames_advanced`. We want to STEP by SKIP_CHECK (e.g. 5).
                     # So we need to skip (SKIP_CHECK - frames_advanced).
                     skips_needed = max(0, SKIP_CHECK - frames_advanced)
                     for _ in range(skips_needed): vidcap.grab()
                     curr += SKIP_CHECK
                     continue
            else:
                is_sharp = True

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            should_capture = prev_hist is None or (1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)) > SCENE_THRESHOLD
            
            if should_capture and is_sharp:
                h, w = img.shape[:2]
                if max(h, w) > 640:
                    scale = 640 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                candidates.append(buf.tobytes())
                prev_hist = hist
            
            # Optimization: Use grab() for skipping instead of set()
            # We want to be at Start + SKIP_CHECK. We currently are at Start + frames_advanced.
            # So grab (SKIP_CHECK - frames_advanced) more.
            skips_needed = max(0, SKIP_CHECK - frames_advanced)
            for _ in range(skips_needed):
                vidcap.grab()
            curr += SKIP_CHECK

        if vidcap: vidcap.release()
        
        if candidates:
             if len(candidates) > target_count:
                indices = np.linspace(0, len(candidates)-1, target_count, dtype=int)
                final_frames = [candidates[i] for i in indices]
             else:
                final_frames = candidates
        elif fallback_candidates:
            fallback_candidates.sort(key=lambda x: x[0], reverse=True)
            final_frames = [item[1] for item in fallback_candidates[:target_count]]
        else:
             raise ValueError("No valid frames extracted.")

        if cancel_event.is_set(): return {"status": "cancelled", "video_path": video_path}

        # --- PHASE 2: API CALL WITH KEY ROTATION ---
        last_error = None
        
        for index, key in enumerate(api_keys):
            try:
                if cancel_event.is_set(): return {"status": "cancelled"}
                
                logging.info(f"Attempting API call with Key #{index+1}")
                text = ""

                if provider == "Groq":
                    import groq
                    client = groq.Groq(api_key=key)
                    image_contents = []
                    for frame_bytes in final_frames:
                        b64 = base64.b64encode(frame_bytes).decode('utf-8')
                        image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    
                    messages = [{"role": "user", "content": [{"type": "text", "text": get_instructional_prompt()}] + image_contents}]
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=MODEL_IDS.get(model_name, model_name),
                        response_format={"type": "json_object"}
                    )
                    text = chat_completion.choices[0].message.content
                
                elif provider == "Gemini": # Gemini
                    import google.generativeai as genai
                    genai.configure(api_key=key)
                    
                    # Desktop Parity: Ensure 'models/' prefix or use hardcoded if specific text matches
                    # The Desktop app explicitly uses "models/gemini-2.5-flash"
                    target_model = model_name
                    if target_model == "gemini-2.5-flash" or target_model == "gemini-1.5-pro": 
                         if not target_model.startswith("models/"): 
                             target_model = f"models/{target_model}"
                    
                    model = genai.GenerativeModel(target_model)
                    content = [get_instructional_prompt()] + [{"mime_type": "image/jpeg", "data": f} for f in final_frames]
                    response = model.generate_content(content, generation_config={"response_mime_type": "application/json"})
                    
                    # Robust Parsing
                    raw_text = response.text
                    import re
                    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                    if match:
                        text = match.group(0)
                    else:
                        text = raw_text # Fallback to raw if no JSON structure found

                elif provider == "Custom":
                    import openai
                    
                    # Load Custom Config
                    base_url = "https://api.openai.com/v1" # Default
                    if os.path.exists(CUSTOM_CONFIG_FILE):
                        try:
                            with open(CUSTOM_CONFIG_FILE, 'r') as f:
                                cfg = json.load(f)
                                base_url = cfg.get("base_url", base_url)
                        except: pass
                    
                    client = openai.OpenAI(api_key=key, base_url=base_url)
                    
                    image_contents = []
                    for frame_bytes in final_frames:
                        b64 = base64.b64encode(frame_bytes).decode('utf-8')
                        image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    
                    messages = [{"role": "user", "content": [{"type": "text", "text": get_instructional_prompt()}] + image_contents}]
                    
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model=model_name, # User provided model ID
                        response_format={"type": "json_object"}
                    )
                    text = chat_completion.choices[0].message.content

                # If successful, return immediately
                # Desktop Parity: Returns 'api_key'
                return {
                    "video_path": video_path, 
                    "prompt": json.dumps(json.loads(text), indent=2), 
                    "status": "success", 
                    "used_key": key[:5]+"..."
                }

            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "quota" in err_msg or "exhausted" in err_msg or "rate limit" in err_msg
                last_error = str(e)
                
                if is_rate_limit:
                    logging.warning(f"Key #{index+1} Rate Limited. Switch to next.")
                    continue # Try next key
                else:
                    logging.error(f"Key #{index+1} Error: {e}")
                    # For reliability, we still try next key even on other errors, 
                    # unless it's a specific 'invalid argument' that would fail everywhere.
                    continue

        # If loop finishes without success, map the LAST error to a status code
        stat = "failed"
        if last_error:
            err = last_error.lower()
            if "api_key" in err or "unauthorized" in err:
                stat = "invalid_key"
            elif "quota" in err or "exhausted" in err or "rate" in err or "429" in err:
                stat = "rate_limit" # Desktop maps 'quota' to 'rate_limit' effectively in this context if all fail
            
        return {"status": stat, "video_path": video_path, "prompt": f"All keys exhausted. Last error: {last_error}"}

    except Exception as e:
        return {"status": "failed", "video_path": video_path, "prompt": str(e)}
    finally:
         if vidcap and vidcap.isOpened(): vidcap.release()
