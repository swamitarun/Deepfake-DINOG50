"""
FastAPI backend for DINOv2 Deepfake Detection.
Deployed on Hugging Face Spaces (Docker).

Endpoint: POST /predict
  - Accepts: video file (mp4, mov, avi) max 30MB
  - Returns: JSON with fake_probability, real_probability, verdict, frame_count
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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Model Definition (self-contained, no src/ imports)
# ─────────────────────────────────────────────

class DINOv2Extractor(nn.Module):
    """Loads DINOv2 ViT-B/14 and extracts 768-dim CLS token."""

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
        return self.backbone(x)  # (B, 768)


class MLPClassifier(nn.Module):
    """MLP head for classification — matches the trained checkpoint."""

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
    """Full deepfake detector: DINOv2 backbone + MLP classifier."""

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
    title="Deepfake Detection API",
    description="DINOv2-based deepfake detector — dual-input video analysis",
    version="1.0.0",
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

# Image transforms (same as training val_transform)
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=1)
def load_model() -> DeepfakeDetector:
    """Load model once and cache it (lru_cache prevents multiple loads)."""
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "Please upload best_model.pth to the HF Space."
        )

    logger.info(f"Loading checkpoint from {CHECKPOINT_PATH} on {DEVICE}...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Check if dual_input based on shapes in checkpoint
    state = ckpt.get("model_state_dict", ckpt)
    # The MLP input is 'classifier.net.0.weight'
    mlp_in_dim = state.get("classifier.net.0.weight", None)
    if mlp_in_dim is not None:
        dual = (mlp_in_dim.shape[1] == 1536)
    else:
        dual = True  # Default to dual
    logger.info(f"Dual input mode: {dual}")

    model = DeepfakeDetector(dual_input=dual).to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    model.eval()
    logger.info("Model loaded and ready.")
    return model


def extract_frames(video_path: str, output_dir: str, num_frames: int = MAX_FRAMES) -> list:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = total_frames / fps if fps > 0 else 0

    if duration > MAX_DURATION_SEC:
        cap.release()
        raise ValueError(f"Video is too long ({duration:.0f}s). Max allowed: {MAX_DURATION_SEC}s.")

    # Select evenly-spaced frame indices
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
    """Run deepfake detection on all frames and aggregate results."""
    fake_probs = []

    with torch.no_grad():
        for fpath in frame_paths:
            try:
                img = Image.open(fpath).convert("RGB")
                t_img = TRANSFORM(img).unsqueeze(0).to(DEVICE)

                # For dual input, use full image as face fallback
                # (MTCNN not used to keep inference fast & dependency-free)
                t_face = t_img if model.dual_input else None

                logits = model(t_img, t_face)
                prob = torch.softmax(logits, dim=1)[0, 1].item()  # P(FAKE)
                fake_probs.append(prob)
            except Exception as e:
                logger.warning(f"Skipping frame {fpath}: {e}")

    if not fake_probs:
        raise ValueError("No valid frames could be processed.")

    avg_fake = float(np.mean(fake_probs))
    avg_real = 1.0 - avg_fake
    verdict = "FAKE" if avg_fake > 0.5 else "REAL"

    return {
        "verdict": verdict,
        "fake_probability": round(avg_fake * 100, 1),
        "real_probability": round(avg_real * 100, 1),
        "frame_count": len(fake_probs),
        "confidence": round(max(avg_fake, avg_real) * 100, 1),
        "per_frame_scores": [round(p * 100, 1) for p in fake_probs],
    }


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup so first request is fast."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Model load failed on startup: {e}")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": "DINOv2 ViT-B/14 Deepfake Detector",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
        "model_loaded": CHECKPOINT_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze a video for deepfakes.

    - Max size: 30 MB
    - Max duration: 60 seconds
    - Supported: mp4, mov, avi, mkv
    - Returns: REAL or FAKE with probability %
    """
    # Validate file type
    allowed_types = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"}
    allowed_exts = {".mp4", ".mov", ".avi", ".mkv"}
    ext = Path(file.filename).suffix.lower() if file.filename else ""

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_exts}"
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_MB} MB."
        )

    # Create a unique temp directory
    job_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.gettempdir()) / f"deepfake_{job_id}"
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / f"input{ext}"

    try:
        # Save uploaded video to disk
        with open(video_path, "wb") as f:
            f.write(content)
        del content  # Free RAM

        # Load model
        model = load_model()

        # Extract frames
        logger.info(f"[{job_id}] Extracting frames from {file.filename} ({size_mb:.1f} MB)")
        frame_paths = extract_frames(str(video_path), str(frames_dir))

        if not frame_paths:
            raise HTTPException(status_code=422, detail="Could not extract any frames from video.")

        logger.info(f"[{job_id}] Running inference on {len(frame_paths)} frames...")
        result = run_inference(model, frame_paths)

        result["filename"] = file.filename
        result["file_size_mb"] = round(size_mb, 2)
        result["job_id"] = job_id

        logger.info(
            f"[{job_id}] Done: {result['verdict']} "
            f"({result['fake_probability']}% fake, {len(frame_paths)} frames)"
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    finally:
        # Always cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"[{job_id}] Temp files cleaned up.")
