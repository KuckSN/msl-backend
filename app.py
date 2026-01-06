#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSL sign language backend API

- /health           : health check
- /predict          : uploaded video -> sequence of tokens + sentence
- /predict_camera   : streaming single frames -> one token at a time
"""
import logging
import os
import tempfile
import base64
from io import BytesIO
import json
import random
from typing import List, Dict, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---- IMPORT FROM YOUR ML ENGINEER'S MODULE ----
from sign_language_recognition import SignLanguageLSTM, MediaPipeProcessor, CONFIG

# ============================================================
# CONFIG
# ============================================================

# Demo vs real model
DEV_MODE = False  # <-- set True for dummy demo, False to use real model

# Model path resolution
MODEL_PATH = (
    os.environ.get("MSL_MODEL_PATH")
    or os.environ.get("MODEL_PATH")
    or CONFIG.get("model_save_path", "sign_language_model.pth")
)

app = Flask(__name__)
CORS(app)

# ============================================================
# REAL-TIME CAMERA CONSTANTS (DEV_MODE only)
# ============================================================

CAMERA_DEMO_SEQUENCE = [
    {"gloss": "hi", "translation": "hi"},
    {"gloss": "hari", "translation": "hari"},
    {"gloss": "hujan", "translation": "hujan"},
    {"gloss": "jangan", "translation": "jangan"},
]

# Which glosses are considered "temporal" (need motion / GIF)
CAMERA_TEMPORAL_GLOSSES = {
    "ambil", "hari", "hi", "hujan", "jangan",
    "kakak", "keluarga", "kereta", "lemak", "lupa",
    "marah", "minum", "pergi", "pukul", "tanya",
}

# Gloss → human-friendly translation (you can extend this)
GLOSS_TRANSLATIONS: Dict[str, str] = {
    "ambil": "ambil (take)",
    "hari": "hari (day)",
    "hi": "hi",
    "hujan": "hujan (rain)",
    "jangan": "jangan (don't)",
    "kakak": "kakak (sister)",
    "keluarga": "keluarga (family)",
    "kereta": "kereta (car)",
    "lemak": "lemak (oil)",
    "lupa": "lupa (forget)",
    "marah": "marah (angry)",
    "minum": "minum (drink)",
    "pergi": "pergi (go)",
    "pukul": "pukul (hit)",
    "tanya": "tanya (ask)",
}

# Per-session sliding buffers (for real-time camera)
SESSION_BUFFERS: Dict[str, List[np.ndarray]] = {}
SESSION_STATE: Dict[str, Dict] = {}

MAX_BUFFER_FRAMES = 150          # keep last ~1–2 seconds worth
FRAMES_PER_TOKEN_MIN = 30        # DEMO: min frames before emitting token

# ============================================================
# GENERIC VIDEO/IMAGE HELPERS (for thumbnails & GIFs)
# ============================================================

def get_translation(gloss: str) -> str:
    """Map gloss to translation, fallback to gloss itself."""
    return GLOSS_TRANSLATIONS.get(gloss.lower(), gloss)


def get_video_metadata(video_path: str) -> Tuple[int, float]:
    """Return (frame_count, fps) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return frame_count, fps


def frame_to_data_url(frame: np.ndarray) -> Optional[str]:
    """Encode a single BGR frame (OpenCV) to a JPEG data URL."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_static_thumbnail(video_path: str, frame_index: int) -> Optional[str]:
    """Grab a single frame from the video and return it as a JPEG data URL."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        cap.release()
        return None

    idx = max(0, min(frame_index, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None
    return frame_to_data_url(frame)


def extract_gif_for_segment(
    video_path: str,
    start_frame: int,
    end_frame: int,
    fps: float | None = None,
    max_frames: int = 40,
    target_seconds: float = 2.5,
) -> Optional[str]:
    """
    Build an animated GIF for a sign segment [start_frame, end_frame].
    - max_frames: upper bound on frames used in GIF
    - target_seconds: approximate loop duration
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        cap.release()
        return None

    start = max(0, int(start_frame))
    end = min(frame_count - 1, int(end_frame))
    if end <= start:
        cap.release()
        return None

    total = end - start + 1
    step = max(1, total // max_frames)

    frames_rgb: List[np.ndarray] = []
    for idx in range(start, end + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)

    cap.release()
    if not frames_rgb:
        return None

    buf = BytesIO()
    duration = target_seconds / len(frames_rgb)

    imageio.mimsave(
        buf,
        frames_rgb,
        format="GIF",
        duration=duration,  # seconds per frame
        loop=0,             # infinite loop
    )
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{b64}"


def add_visuals_to_tokens(video_path: str, tokens: List[Dict]) -> List[Dict]:
    """
    For uploaded video predictions:
    Given tokens (with gloss, translation, confidence, temporal, start_frame,
    end_frame, fps), attach:
      - thumbnail_type: 'gif' or 'image'
      - thumbnail_url: data URL (GIF or JPG)
    """
    frame_count, vid_fps = get_video_metadata(video_path)
    if frame_count <= 0:
        return tokens

    n = len(tokens) or 1
    default_fps = vid_fps or 30.0
    segment_len = max(1, frame_count // n)

    for i, t in enumerate(tokens):
        # 1. Determine start/end frames
        if "start_frame" in t and "end_frame" in t:
            start = int(t["start_frame"])
            end = int(t["end_frame"])
        else:
            start = i * segment_len
            end = start + segment_len - 1

        start = max(0, min(start, frame_count - 1))
        end = max(start, min(end, frame_count - 1))

        t["start_frame"] = start
        t["end_frame"] = end

        # 2. Determine fps for this token
        fps_token = float(t.get("fps") or default_fps)
        t["fps"] = fps_token

        # 3. Build visuals
        thumb_type = "image"
        thumb_url = None

        if t.get("temporal"):
            gif_url = extract_gif_for_segment(
                video_path,
                start_frame=start,
                end_frame=end,
                fps=fps_token,
            )
            if gif_url:
                thumb_type = "gif"
                thumb_url = gif_url
            else:
                center = (start + end) // 2
                thumb_url = extract_static_thumbnail(video_path, center)
        else:
            center = (start + end) // 2
            thumb_url = extract_static_thumbnail(video_path, center)

        t["thumbnail_type"] = thumb_type
        if thumb_url:
            t["thumbnail_url"] = thumb_url

    return tokens


def gif_from_frames(
    frames_bgr: List[np.ndarray],
    target_seconds: float = 2.5,
    max_frames: int = 40,
) -> Optional[str]:
    """
    Create a looping GIF from a list of BGR frames (OpenCV images).
    We subsample if too many frames, to control GIF size.
    """
    if not frames_bgr:
        return None

    step = max(1, len(frames_bgr) // max_frames)
    sampled = [frames_bgr[i] for i in range(0, len(frames_bgr), step)]

    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in sampled]
    if not frames_rgb:
        return None

    buf = BytesIO()
    duration = target_seconds / len(frames_rgb)

    imageio.mimsave(
        buf,
        frames_rgb,
        format="GIF",
        duration=duration,
        loop=0,  # infinite
    )
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{b64}"


# ============================================================
# MODEL INTEGRATION (merged with backend_api.py)
# ============================================================

model: Optional[nn.Module] = None
gestures: Optional[List[str]] = None
label_map: Optional[Dict] = None
config: Optional[Dict] = None
device: Optional[torch.device] = None
processor: Optional[MediaPipeProcessor] = None


def load_model(model_path: str):
    """Load trained model from .pth checkpoint (backend_api version)."""
    global model, gestures, label_map, config, device, processor

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MediaPipeProcessor()

    print(f"[MSL] Loading model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device)

    required_keys = ["gestures", "label_map", "config", "model_state_dict"]
    missing_keys = [k for k in required_keys if k not in checkpoint]
    if missing_keys:
        raise ValueError(f"Model checkpoint missing keys: {missing_keys}")

    gestures_cp = checkpoint["gestures"]
    label_map_cp = checkpoint["label_map"]
    config_cp = checkpoint["config"]

    print("[MSL] Model config:")
    print(f"  - gestures: {len(gestures_cp)}")
    print(f"  - sequence_length: {config_cp.get('sequence_length', 'N/A')}")
    print(f"  - input_size: {config_cp.get('input_size', 'N/A')}")
    print(f"  - hidden_size: {config_cp.get('hidden_size', 'N/A')}")
    print(f"  - device: {device}")

    slr_model = SignLanguageLSTM(
        config_cp["input_size"],
        config_cp["hidden_size"],
        len(gestures_cp),
    ).to(device)

    slr_model.load_state_dict(checkpoint["model_state_dict"])
    slr_model.eval()

    gestures = gestures_cp
    label_map = label_map_cp
    config = config_cp
    model = slr_model

    print(f"✅ Model loaded. Gestures: {len(gestures)}")
    print(f"Gestures list: {gestures}")
    return model

def extract_keypoints(results) -> np.ndarray:
    """从 MediaPipe 结果中提取关键点"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                    for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    
    lh = np.array([[res.x, res.y, res.z] 
                  for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    
    rh = np.array([[res.x, res.y, res.z] 
                  for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, lh, rh])


def process_frames_to_sequence(frames: List[np.ndarray], max_frames: Optional[int] = None) -> Optional[np.ndarray]:
    """将帧序列转换为关键点序列"""
    # 如果没有指定max_frames，使用config中的sequence_length
    if max_frames is None:
        if config is None:
            raise RuntimeError("config未初始化，请先加载模型")
        max_frames = config.get('sequence_length', 120)
    
    keypoints_sequence = []
    
    for frame in frames:
        # 处理帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = processor.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        # 只保存检测到手部的帧
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)
    
    if len(keypoints_sequence) == 0:
        return None
    
    # 填充或截断到固定长度（120帧）
    if len(keypoints_sequence) < max_frames:
        last_frame = keypoints_sequence[-1]
        keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
    else:
        keypoints_sequence = keypoints_sequence[:max_frames]
    
    return np.array(keypoints_sequence)


def predict_sequence(keypoints_seq: np.ndarray) -> Tuple[Optional[str], float]:
    """预测关键点序列对应的手势"""
    if keypoints_seq is None:
        return None, 0.0
    
    # 转换为tensor
    keypoints_tensor = torch.FloatTensor(keypoints_seq).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(keypoints_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_gesture = gestures[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_gesture, confidence_score


def is_temporal_sign(gloss: str) -> bool:
    """Decide if a sign is temporal (requires motion)."""
    return gloss.lower() in CAMERA_TEMPORAL_GLOSSES


def predict_sign(video_path: str, model_obj=None) -> Dict:
    """
    Offline video processing – full gesture sequence.
    (This is the backend_api sliding-window logic.)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"tokens": [], "sentence": ""}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps < 1:
        fps = 30.0  # default FPS

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        return {"tokens": [], "sentence": ""}

    sequence_length = config["sequence_length"]
    step_size = max(1, sequence_length // 3)  # overlap 2/3

    tokens = []
    current_start = 0
    last_gloss = None
    last_end = -1

    # 如果视频帧数不足120帧，仍然尝试预测（使用所有可用帧）
    if len(all_frames) < sequence_length:
        window_frames = all_frames
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)
        
        if keypoints_seq is not None:
            gloss, confidence = predict_sequence(keypoints_seq)
            if gloss and confidence > 0.5:
                token = {
                    "gloss": gloss,
                    "translation": get_translation(gloss),
                    "confidence": float(confidence),
                    "temporal": is_temporal_sign(gloss),
                    "start_frame": 0,
                    "end_frame": len(all_frames) - 1,
                    "fps": float(fps)
                }
                return {
                    "tokens": [token],
                    "sentence": get_translation(gloss),
                    "warning": f"视频帧数不足{sequence_length}帧（实际{len(all_frames)}帧），预测结果可能不够准确"
                }
        
        # 无法提取有效手势
        return {
            "tokens": [],
            "sentence": "",
            "error": f"视频帧数不足{sequence_length}帧（实际{len(all_frames)}帧），且无法提取有效手势",
            "frame_count": len(all_frames),
            "required_frames": sequence_length
        }
    
    while current_start + sequence_length <= len(all_frames):
        # 提取当前窗口的帧
        window_frames = all_frames[current_start:current_start + sequence_length]
        
        # 处理帧序列（提取120帧的关键点）
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)
        
        if keypoints_seq is not None:
            # 预测手势
            gloss, confidence = predict_sequence(keypoints_seq)
            
            if gloss and confidence > 0.5:  # 置信度阈值
                end_frame = current_start + sequence_length - 1
                
                # 如果与上一个token相同，扩展范围
                if last_gloss == gloss and current_start <= last_end + step_size:
                    tokens[-1]['end_frame'] = int(end_frame)
                    tokens[-1]['confidence'] = max(tokens[-1]['confidence'], float(confidence))
                else:
                    # 新的手势token
                    token = {
                        "gloss": gloss,
                        "translation": get_translation(gloss),
                        "confidence": float(confidence),
                        "temporal": is_temporal_sign(gloss),
                        "start_frame": int(current_start),
                        "end_frame": int(end_frame),
                        "fps": float(fps)
                    }
                    tokens.append(token)
                    last_gloss = gloss
                    last_end = end_frame
        
        current_start += step_size
    
    # 构建句子
    sentence = " ".join([token['translation'] for token in tokens])
    
    return {
        "tokens": tokens,
        "sentence": sentence
    }


def run_msl_model_on_frames(
    frames_bgr: List[np.ndarray],
    model_obj=None,
) -> Optional[Dict]:
    """
    Streaming / real-time frames → one token or None.
    Backend_api logic, but used by /predict_camera.
    """
    if config is None:
        raise RuntimeError("config未初始化，请先加载模型")
    
    sequence_length = config['sequence_length']  # 120帧
    
    if len(frames_bgr) < sequence_length:
        return None  # 帧数不足，继续缓冲

    # Use the most recent sequence_length frames
    recent_frames = frames_bgr[-sequence_length:]

    keypoints_seq = process_frames_to_sequence(recent_frames, sequence_length)

    if keypoints_seq is None:
        return None  # no hands detected

    gloss, confidence = predict_sequence(keypoints_seq)
    if not gloss or confidence < 0.6:  # stricter threshold
        return None

    start_frame = len(frames_bgr) - sequence_length
    end_frame = len(frames_bgr) - 1

    token = {
        "gloss": gloss,
        "translation": get_translation(gloss),
        "confidence": float(confidence),
        "temporal": is_temporal_sign(gloss),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": 30.0,  # assumed FPS for streaming
    }
    return token


# Load model at import time (unless in demo mode)
if not DEV_MODE:
    load_model(MODEL_PATH)

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "num_gestures": len(gestures) if gestures else 0,
        "sequence_length": config.get('sequence_length', 'N/A') if config else 'N/A',
        "model_version": "v3"
    })

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Upload a video (field name: 'file' or 'video') and get:
      {
        "mode": "dev-demo" | "model",
        "sequence": [...],
        "sentence": "..."
      }
    """
    file = request.files.get("file") or request.files.get("video")
    if file is None:
        return jsonify({"error": "No file or video field found"}), 400
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # ---------------- DEMO MODE ----------------
        if DEV_MODE:
            frame_count, fps = get_video_metadata(tmp_path)
            if frame_count <= 0:
                return jsonify({"error": "Could not read video"}), 400

            n_tokens = 4
            segment_len = max(1, frame_count // n_tokens)
            fps = fps or 30.0

            glosses = [
                ("SALAAM", "Hello", False),
                ("SAYA", "I", False),
                ("PELAJAR", "am a student", True),
                ("UNIVERSITI_MALAYA", "at Universiti Malaya", True),
            ]

            tokens: List[Dict] = []
            for i, (gloss, translation, temporal) in enumerate(glosses):
                start = i * segment_len
                end = start + segment_len - 1
                start = max(0, min(start, frame_count - 1))
                end = max(start, min(end, frame_count - 1))

                tokens.append(
                    {
                        "id": i,
                        "gloss": gloss,
                        "translation": translation,
                        "confidence": 0.9 + 0.01 * i,
                        "temporal": temporal,
                        "start_frame": start,
                        "end_frame": end,
                        "fps": fps,
                    }
                )

            tokens = add_visuals_to_tokens(tmp_path, tokens)
            sentence = "Hello, I am a student at Universiti Malaya."

            return jsonify(
                {
                    "mode": "dev-demo",
                    "sequence": tokens,
                    "sentence": sentence,
                }
            )

        # ---------------- REAL MODEL MODE ----------------
        if model is None or config is None:
            return jsonify({"error": "Model not loaded"}), 500

        result = predict_sign(tmp_path, model)
        tokens = result.get("tokens", [])
        sentence = result.get("sentence", "")

        tokens = add_visuals_to_tokens(tmp_path, tokens)

        logging.info(f"{result}")

        # Frontend normalizePrediction() accepts sequence or tokens
        return jsonify(
            {
                "mode": "model",
                "sequence": tokens,
                "tokens": tokens,
                "sentence": sentence,
            }
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.route("/predict_camera", methods=["POST"])
def predict_camera_route():
    """
    Real-time camera endpoint.

    Frontend sends:
      - 'frame': image file (JPEG) – a single frame
      - 'session_id': string (same while camera is on)
    Returns:
      { "token": {...} } or { "token": null }
    """
    if "frame" not in request.files:
        return jsonify({"error": "No file field 'frame' in form-data"}), 400

    file = request.files["frame"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    data = file.read()
    if not data:
        return jsonify({"error": "Empty image data"}), 400

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode frame"}), 400

    session_id = (
        request.form.get("session_id")
        or request.args.get("session_id")
        or "default"
    )

    buf = SESSION_BUFFERS.setdefault(session_id, [])
    state = SESSION_STATE.setdefault(
        session_id,
        {
            "index": 0,
            "frames_since_token": 0,
        },
    )

    buf.append(frame)
    state["frames_since_token"] += 1

    if len(buf) > MAX_BUFFER_FRAMES:
        buf[:] = buf[-MAX_BUFFER_FRAMES:]

    # ---------------- DEMO MODE (fake sentence loop) ----------------
    if DEV_MODE:
        if state["frames_since_token"] < FRAMES_PER_TOKEN_MIN:
            return jsonify({"token": None})

        spec = CAMERA_DEMO_SEQUENCE[state["index"]]
        gloss = spec["gloss"]
        translation = spec["translation"]
        temporal = gloss in CAMERA_TEMPORAL_GLOSSES
        confidence = round(0.90 + random.random() * 0.05, 3)

        frames_for_token = list(buf)

        if temporal:
            thumb_url = gif_from_frames(frames_for_token)
            thumb_type = "gif"
        else:
            last = frames_for_token[-1]
            thumb_url = frame_to_data_url(last)
            thumb_type = "image"

        buf.clear()
        state["frames_since_token"] = 0
        state["index"] = (state["index"] + 1) % len(CAMERA_DEMO_SEQUENCE)

        token = {
            "id": f"{session_id}-{state['index']}",
            "gloss": gloss,
            "translation": translation,
            "confidence": confidence,
            "temporal": temporal,
            "start_frame": 0,
            "end_frame": len(frames_for_token) - 1,
            "fps": 30.0,
            "thumbnail_type": thumb_type,
            "thumbnail_url": thumb_url,
        }
        return jsonify({"token": token})

    # ---------------- REAL MODEL MODE ----------------
    if model is None or config is None:
        return jsonify({"error": "Model not loaded"}), 500

    token_from_model = run_msl_model_on_frames(buf, model)
    if token_from_model is None:
        return jsonify({"token": None})

    start_local = int(token_from_model["start_frame"])
    end_local = int(token_from_model["end_frame"])
    start_local = max(0, min(start_local, len(buf) - 1))
    end_local = max(start_local, min(end_local, len(buf) - 1))

    frames_for_token = buf[start_local : end_local + 1]

    temporal = bool(token_from_model.get("temporal", True))
    if temporal:
        thumb_url = gif_from_frames(frames_for_token)
        thumb_type = "gif"
    else:
        mid_idx = len(frames_for_token) // 2
        thumb_url = frame_to_data_url(frames_for_token[mid_idx])
        thumb_type = "image"

    # Drop consumed frames so next sign uses fresh motion
    del buf[: end_local + 1]

    token = {
        "id": f"{session_id}-{random.randint(0, 1_000_000)}",
        "gloss": token_from_model["gloss"],
        "translation": token_from_model.get(
            "translation",
            token_from_model["gloss"],
        ),
        "confidence": token_from_model.get("confidence", 0.9),
        "temporal": temporal,
        "start_frame": token_from_model.get("start_frame", 0),
        "end_frame": token_from_model.get(
            "end_frame",
            len(frames_for_token) - 1,
        ),
        "fps": token_from_model.get("fps", 30.0),
        "thumbnail_type": thumb_type,
        "thumbnail_url": thumb_url,
    }

    return jsonify({"token": token})


# --- ASGI wrapper for uvicorn (optional) ---
try:
    from asgiref.wsgi import WsgiToAsgi

    asgi_app = WsgiToAsgi(app)
except ImportError:
    asgi_app = None


if __name__ == "__main__":
    # If dev mode is off and model wasn't loaded yet, try again here
    if not DEV_MODE and model is None:
        load_model(MODEL_PATH)

    app.run(host="0.0.0.0", port=8000, debug=True)
