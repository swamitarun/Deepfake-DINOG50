"""
DeepShield AI — Full-Stack FastAPI Backend
Serves the frontend UI + deepfake detection API from one HF Space.

Routes:
  GET  /          → Serves index.html (the web UI)
  GET  /health    → JSON health check
  POST /predict   → Video upload → REAL/FAKE prediction
"""

import os
import sys
import uuid
import shutil
import logging
import tempfile
from pathlib import Path
from functools import lru_cache

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageFile
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model Definition (self-contained)
# ─────────────────────────────────────────────

class DINOv2Extractor(nn.Module):
    def __init__(self, variant: str = "dinov2_vitb14"):
        super().__init__()
        logger.info(f"Loading {variant} from torch.hub...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", variant, pretrained=True
        )
        self.feature_dim = 768
        for p in self.backbone.parameters():
            p.requires_grad = False
        logger.info("DINOv2 backbone loaded (frozen).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepfakeDetector(nn.Module):
    def __init__(self, dual_input: bool = True):
        super().__init__()
        self.dual_input = dual_input
        self.extractor = DINOv2Extractor()
        feat_dim = 1536 if dual_input else 768
        self.classifier = MLPClassifier(input_dim=feat_dim)

    def forward(self, full_img: torch.Tensor, face_img: torch.Tensor = None) -> torch.Tensor:
        full_feat = self.extractor(full_img)
        if self.dual_input and face_img is not None:
            face_feat = self.extractor(face_img)
            feats = torch.cat([full_feat, face_feat], dim=1)
        else:
            feats = full_feat
        return self.classifier(feats)


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="DeepShield AI",
    description="DINO-G50 deepfake detector — full-stack web app",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("best_model.pth")
MAX_FRAMES = 20
MAX_FILE_MB = 30
MAX_DURATION_SEC = 60

# MTCNN face detector (initialized once, CPU is fine for detection)
try:
    MTCNN_DETECTOR = MTCNN(
        image_size=224,
        margin=40,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.9],
        keep_all=False,
        device='cpu'
    )
    logger.info("MTCNN face detector initialized.")
except Exception as e:
    MTCNN_DETECTOR = None
    logger.warning(f"MTCNN init failed (will use full frame fallback): {e}")

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def detect_face_crop(img: Image.Image) -> Image.Image:
    """Detect face with MTCNN and return cropped face, or None if not found."""
    if MTCNN_DETECTOR is None:
        return None
    try:
        # MTCNN returns the cropped tensor directly
        face_tensor = MTCNN_DETECTOR(img)
        if face_tensor is not None:
            # Convert tensor back to PIL Image
            face_np = face_tensor.permute(1, 2, 0).numpy()
            face_np = ((face_np * 128) + 127.5).clip(0, 255).astype(np.uint8)
            return Image.fromarray(face_np)
    except Exception:
        pass
    return None


@lru_cache(maxsize=1)
def load_model() -> DeepfakeDetector:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError("best_model.pth not found. Upload it to this HF Space.")

    logger.info(f"Loading checkpoint on {DEVICE}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)

    mlp_w = state.get("classifier.net.0.weight", None)
    dual = (mlp_w.shape[1] == 1536) if mlp_w is not None else True

    model = DeepfakeDetector(dual_input=dual).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Model ready. dual_input={dual}, device={DEVICE}")
    return model


def extract_frames(video_path: str, output_dir: str, num_frames: int = MAX_FRAMES) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / fps if fps > 0 else 0

    if duration > MAX_DURATION_SEC:
        cap.release()
        raise ValueError(f"Video too long ({duration:.0f}s). Max: {MAX_DURATION_SEC}s.")

    if total_frames <= 0:
        total_frames = int(fps * MAX_DURATION_SEC)

    step = max(1, total_frames // num_frames)
    target_indices = set(range(0, total_frames, step))
    saved_paths = []
    frame_idx = 0

    while len(saved_paths) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = os.path.join(output_dir, f"frame_{len(saved_paths):04d}.jpg")
            Image.fromarray(rgb).save(path, quality=90)
            saved_paths.append(path)
        frame_idx += 1

    cap.release()
    return saved_paths


def run_inference(model: DeepfakeDetector, frame_paths: list) -> dict:
    fake_probs = []
    with torch.no_grad():
        for fpath in frame_paths:
            try:
                img = Image.open(fpath).convert("RGB")
                t_img = TRANSFORM(img).unsqueeze(0).to(DEVICE)

                # Try MTCNN face detection first (same as test_real.py)
                t_face = t_img  # default fallback = full frame
                if model.dual_input:
                    face_crop = detect_face_crop(img)
                    if face_crop is not None:
                        t_face = TRANSFORM(face_crop).unsqueeze(0).to(DEVICE)
                    # else: fallback to full image (face not detected)

                logits = model(t_img, t_face if model.dual_input else None)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
                fake_probs.append(prob)
            except Exception as e:
                logger.warning(f"Skipping frame {fpath}: {e}")

    if not fake_probs:
        raise ValueError("No frames could be processed.")

    # 1. Advanced Aggregation (Top 50% Mean)
    # Deepfake artifacts might only appear in parts of the video.
    # Averaging all frames dilutes the score. We take the top 50% most suspicious frames.
    sorted_probs = sorted(fake_probs, reverse=True)
    top_k = max(1, len(sorted_probs) // 2)
    video_fake_prob = float(np.mean(sorted_probs[:top_k]))

    # 2. Ratio Check
    # If at least 30% of frames are distinctly flagged as Fake, mark the whole video as Fake.
    fake_frame_count = sum(1 for p in fake_probs if p > 0.5)
    fake_ratio = fake_frame_count / len(fake_probs)

    is_fake = (video_fake_prob > 0.5) or (fake_ratio >= 0.3)

    # Ensure UI consistency: If flagged as FAKE by ratio, but probability is low, boost it to 51%
    if is_fake and video_fake_prob <= 0.5:
        video_fake_prob = 0.51

    avg_real = 1.0 - video_fake_prob

    return {
        "verdict": "FAKE" if is_fake else "REAL",
        "fake_probability": round(video_fake_prob * 100, 1),
        "real_probability": round(avg_real * 100, 1),
        "frame_count": len(fake_probs),
        "confidence": round(max(video_fake_prob, avg_real) * 100, 1),
        "per_frame_scores": [round(p * 100, 1) for p in fake_probs],
    }


# ─────────────────────────────────────────────
# API Routes (must be defined BEFORE static mount)
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "DINO-G50 Deepfake Detector",
        "device": str(DEVICE),
        "model_loaded": CHECKPOINT_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""

    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported type '{ext}'. Use: {allowed_exts}")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_MB} MB.")

    job_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.gettempdir()) / f"deepshield_{job_id}"
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / f"input{ext}"

    try:
        with open(video_path, "wb") as f:
            f.write(content)
        del content

        model = load_model()
        logger.info(f"[{job_id}] Processing: {file.filename} ({size_mb:.1f} MB)")

        frame_paths = extract_frames(str(video_path), str(frames_dir))
        if not frame_paths:
            raise HTTPException(422, "No frames could be extracted from video.")

        result = run_inference(model, frame_paths)
        result["filename"] = file.filename
        result["file_size_mb"] = round(size_mb, 2)
        result["job_id"] = job_id

        logger.info(f"[{job_id}] Result: {result['verdict']} ({result['fake_probability']}% fake)")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"[{job_id}] Cleanup done.")


# ─────────────────────────────────────────────
# Static Frontend (mounted LAST — serves index.html at /)
# ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
