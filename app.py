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
# Make sure sign_language_recognition.py is in the same folder or on PYTHONPATH
from sign_language_recognition import SignLanguageLSTM, MediaPipeProcessor, CONFIG

# ============================================================
# CONFIG
# ============================================================

# Demo vs real model
DEV_MODE = False  # <-- set True for your old dummy demo, False to use real model

# Path to your .pth model (adjust if needed)
MODEL_PATH = os.environ.get("MSL_MODEL_PATH", "sign_language_model.pth")

app = Flask(__name__)
CORS(app)

# ============================================================
# REAL-TIME CAMERA CONSTANTS (used in DEV_MODE only)
# ============================================================

CAMERA_DEMO_SEQUENCE = [
    {"gloss": "hi", "translation": "hi"},
    {"gloss": "apa_khabar", "translation": "apa khabar"},
    {"gloss": "hari", "translation": "hari"},
    {"gloss": "hujan", "translation": "hujan"},
    {"gloss": "jangan", "translation": "jangan"},
    {"gloss": "curi", "translation": "curi (steal)"},
    {"gloss": "payung", "translation": "payung"},
]


CAMERA_TEMPORAL_GLOSSES = {
    "abang", "ambil", "apa_khabar", "ayah", "bapa", "bapa_saudara", "bawa", "buat", "curi", "hari", "hi", "hujan", "jangan", "kakak", "lelaki", "lemak", "lupa", "main", "marah", "minum", "panas_2", "payung", "pergi", "pukul", "ribut", "siapa", "tanya"
}

GLOSS_TRANSLATIONS = {
    "abang": "abang",
    "ambil": "ambil (take)",
    "apa_khabar": "apa khabar",
    "ayah": "ayah",
    "bapa": "bapa",
    "bapa_saudara": "bapa saudara",
    "bawa": "bawa (bring)",
    "buat": "buat (do/make)",
    "curi": "curi (steal)",
    "hari": "hari",
    "hi": "hi",
    "hujan": "hujan (rain)",
    "jangan": "jangan",
    "kakak": "kakak",
    "lelaki": "lelaki",
    "lemak": "lemak (fatty)",
    "lupa": "lupa (forget)",
    "main": "main (play)",
    "marah": "marah (angry)",
    "minum": "minum (drink)",
    "panas_2": "panas",
    "payung": "payung (umbrella)",
    "pergi": "pergi (go)",
    "pukul": "pukul (hit)",
    "ribut": "ribut (storm)",
    "siapa": "siapa",
    "tanya": "tanya (ask)",
}


# Per-session sliding buffers (for real-time camera)
SESSION_BUFFERS: dict[str, List[np.ndarray]] = {}
SESSION_STATE: dict[str, dict] = {}

MAX_BUFFER_FRAMES = 45        # keep last ~1–2 seconds worth (depending on fps)
FRAMES_PER_TOKEN_MIN = 10     # demo: need at least this many frames to “detect” a sign

# ============================================================
# GENERIC VIDEO/IMAGE HELPERS (for thumbnails & GIFs)
# ============================================================

def get_translation(gloss: str) -> str:
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


def frame_to_data_url(frame) -> Optional[str]:
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
    - target_seconds: approximate loop duration for the GIF
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

    frames_rgb = []
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
# MODEL INTEGRATION (from your ML engineer)
# ============================================================

model: Optional[nn.Module] = None
gestures: Optional[List[str]] = None
label_map: Optional[Dict] = None
config: Optional[Dict] = None
device: Optional[torch.device] = None
processor: Optional[MediaPipeProcessor] = None


def load_model(model_path: str):
    """Load trained model from .pth checkpoint."""
    global model, gestures, label_map, config, device, processor

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = MediaPipeProcessor()

    checkpoint = torch.load(model_path, map_location=device)
    gestures_cp = checkpoint["gestures"]
    label_map_cp = checkpoint["label_map"]
    config_cp = checkpoint["config"]

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

    print(f"[MSL] Model loaded. Gestures: {len(gestures)}")
    return model


def extract_keypoints(results) -> np.ndarray:
    """Extract pose + left/right hand keypoints from MediaPipe results."""
    pose = (
        np.array(
            [[res.x, res.y, res.z, res.visibility]
             for res in results.pose_landmarks.landmark]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    lh = (
        np.array(
            [[res.x, res.y, res.z]
             for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array(
            [[res.x, res.y, res.z]
             for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])


def process_frames_to_sequence(
    frames: List[np.ndarray],
    max_frames: int = 30,
) -> Optional[np.ndarray]:
    """Convert a sequence of frames into a fixed-length keypoint sequence."""
    global processor

    if processor is None:
        processor = MediaPipeProcessor()

    keypoints_sequence = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = processor.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            keypoints_sequence.append(keypoints)

    if len(keypoints_sequence) == 0:
        return None

    if len(keypoints_sequence) < max_frames:
        last_frame = keypoints_sequence[-1]
        keypoints_sequence.extend(
            [last_frame] * (max_frames - len(keypoints_sequence))
        )
    else:
        keypoints_sequence = keypoints_sequence[:max_frames]

    return np.array(keypoints_sequence)


def predict_sequence(keypoints_seq: np.ndarray) -> Tuple[Optional[str], float]:
    """Predict a gesture from a keypoint sequence."""
    if keypoints_seq is None:
        return None, 0.0

    global model, device, gestures
    if model is None or gestures is None:
        return None, 0.0

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
    temporal_keywords = ["abang", "ambil", "apa_khabar", "ayah", "bapa", "bapa_saudara", "bawa", "buat", "curi", "hari", "hi", "hujan", "jangan", "kakak", "lelaki", "lemak", "lupa", "main", "marah", "minum", "panas_2", "payung", "pergi", "pukul", "ribut", "siapa", "tanya"]
    return any(keyword in gloss.lower() for keyword in temporal_keywords)


def get_translation(gloss: str) -> str:
    """Map gloss to translation (for now: return gloss)."""
    return gloss


def predict_sign(video_path: str, model_obj=None) -> Dict:
    """
    Offline video processing – full gesture sequence.

    Returns:
      {
        "tokens": [...],
        "sentence": "..."
      }
    """
    global config

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"tokens": [], "sentence": ""}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps < 1:
        fps = 30.0

    all_frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        return {"tokens": [], "sentence": ""}

    sequence_length = config["sequence_length"]
    step_size = max(1, sequence_length // 3)

    tokens: List[Dict] = []
    current_start = 0
    last_gloss = None
    last_end = -1

    while current_start + sequence_length <= len(all_frames):
        window_frames = all_frames[current_start : current_start + sequence_length]
        keypoints_seq = process_frames_to_sequence(window_frames, sequence_length)

        if keypoints_seq is not None:
            gloss, confidence = predict_sequence(keypoints_seq)

            if gloss and confidence > 0.5:
                end_frame = current_start + sequence_length - 1

                if last_gloss == gloss and current_start <= last_end + step_size:
                    tokens[-1]["end_frame"] = int(end_frame)
                    tokens[-1]["confidence"] = max(
                        tokens[-1]["confidence"],
                        float(confidence),
                    )
                else:
                    token = {
                        "gloss": gloss,
                        "translation": get_translation(gloss),
                        "confidence": float(confidence),
                        "temporal": is_temporal_sign(gloss),
                        "start_frame": int(current_start),
                        "end_frame": int(end_frame),
                        "fps": float(fps),
                    }
                    tokens.append(token)
                    last_gloss = gloss
                    last_end = end_frame

        current_start += step_size

    sentence = " ".join([token["translation"] for token in tokens])
    return {
        "tokens": tokens,
        "sentence": sentence,
    }


def run_msl_model_on_frames(
    frames_bgr: List[np.ndarray],
    model_obj=None,
) -> Optional[Dict]:
    """
    Streaming / real-time frames → one token or None.

    Returns:
      None
      or token:
      {
        "gloss": "SAYA",
        "translation": "...",
        "confidence": 0.93,
        "temporal": True/False,
        "start_frame": <idx in this buffer>,
        "end_frame": <idx in this buffer>,
        "fps": 30.0
      }
    """
    global config
    if config is None:
        return None

    seq_len = config["sequence_length"]
    if len(frames_bgr) < seq_len:
        return None

    recent_frames = frames_bgr[-seq_len:]
    keypoints_seq = process_frames_to_sequence(recent_frames, seq_len)

    if keypoints_seq is None:
        return None

    gloss, confidence = predict_sequence(keypoints_seq)
    if not gloss or confidence < 0.6:
        return None

    start_frame = len(frames_bgr) - seq_len
    end_frame = len(frames_bgr) - 1

    token = {
        "gloss": gloss,
        "translation": get_translation(gloss),
        "confidence": float(confidence),
        "temporal": is_temporal_sign(gloss),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "fps": 30.0,
    }
    return token


# Load model at startup (unless in demo mode)
if not DEV_MODE:
    load_model(MODEL_PATH)

# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "dev_mode": DEV_MODE,
            "model_loaded": model is not None,
            "num_gestures": len(gestures) if gestures else 0,
        }
    ), 200


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
