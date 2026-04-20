---
title: Deepfake DINOv2 API
emoji: 🎭
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Deepfake Detection API (DINOv2)

FastAPI backend for detecting deepfake videos using DINOv2 Vision Transformer.

## Endpoints

- `GET /` — Health check
- `POST /predict` — Upload a video file, receive fake/real prediction

## Usage

```bash
curl -X POST "https://mrtsp-deepfake-dinov2-api.hf.space/predict" \
  -F "file=@your_video.mp4"
```
