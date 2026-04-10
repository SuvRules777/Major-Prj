"""
Realtime Fish Biomass API

Phone captures image -> uploads to this API -> YOLO detection + biomass estimation -> JSON response.

Run:
    python src/realtime_biomass_api.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from run_biomass_estimation import DEFAULT_PARAMS, estimate_biomass, load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"

app = FastAPI(title="Realtime Fish Biomass API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

MODEL = None


def detect_fish_from_array(image_bgr: np.ndarray, model: Any, conf_threshold: float) -> list[dict[str, Any]]:
    """Run YOLO inference on an in-memory BGR image and return measurements per detection."""
    results = model(image_bgr, conf=conf_threshold, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        masks = result.masks
        mask_xy = masks.xy if masks is not None else None

        if boxes is None:
            continue

        for idx, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            det = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": class_name,
                "pixel_length": float(x2 - x1),
                "pixel_height": float(y2 - y1),
                "bbox_area": float((x2 - x1) * (y2 - y1)),
            }

            if mask_xy is not None and idx < len(mask_xy):
                coords = mask_xy[idx]
                if coords is not None and len(coords) > 0:
                    points = coords.astype(np.int32)
                    mask_area = cv2.contourArea(points)
                    det["mask_area"] = float(mask_area)

                    if len(points) >= 5:
                        ellipse = cv2.fitEllipse(points)
                        det["major_axis_length"] = float(max(ellipse[1]))
                    else:
                        det["major_axis_length"] = det["pixel_length"]
                else:
                    det["mask_area"] = det["bbox_area"]
                    det["major_axis_length"] = det["pixel_length"]
            else:
                det["mask_area"] = det["bbox_area"]
                det["major_axis_length"] = det["pixel_length"]

            detections.append(det)

    return detections


@app.on_event("startup")
def startup_event() -> None:
    """Load model once at API startup."""
    global MODEL
    MODEL = load_model(None)
    if MODEL is None:
        raise RuntimeError("Failed to load YOLO model. Install ultralytics and verify weights.")


@app.get("/", response_model=None)
def root():
    """Serve phone UI if available."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "API running. Open /docs for Swagger UI."}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded" if MODEL is not None else "not_loaded"}


@app.post("/predict-biomass")
async def predict_biomass(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    pixels_per_cm: float = Form(DEFAULT_PARAMS["pixels_per_cm"]),
) -> dict[str, Any]:
    """Infer fish detections and biomass from one uploaded image."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if conf < 0 or conf > 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1")
    if pixels_per_cm <= 0:
        raise HTTPException(status_code=400, detail="pixels_per_cm must be > 0")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    img_array = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")

    detections = detect_fish_from_array(image_bgr, MODEL, conf_threshold=conf)

    params = dict(DEFAULT_PARAMS)
    params["pixels_per_cm"] = float(pixels_per_cm)

    items: list[dict[str, Any]] = []
    total_biomass_g = 0.0

    for i, det in enumerate(detections, start=1):
        weight_g, length_cm = estimate_biomass(det["major_axis_length"], params=params)
        total_biomass_g += weight_g
        items.append(
            {
                "detection_id": i,
                "class_name": det["class_name"],
                "confidence": round(det["confidence"], 4),
                "pixel_length": round(det["pixel_length"], 2),
                "pixel_height": round(det["pixel_height"], 2),
                "bbox_area": round(det["bbox_area"], 2),
                "mask_area": round(det["mask_area"], 2),
                "estimated_length_cm": round(length_cm, 3),
                "estimated_weight_g": round(weight_g, 3),
            }
        )

    return {
        "filename": file.filename,
        "fish_count": len(items),
        "total_estimated_biomass_g": round(total_biomass_g, 3),
        "parameters": {
            "confidence_threshold": conf,
            "pixels_per_cm": pixels_per_cm,
            "allometric_a": params["a"],
            "allometric_b": params["b"],
        },
        "detections": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime fish biomass API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("realtime_biomass_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
