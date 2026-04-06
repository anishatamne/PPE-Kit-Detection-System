"""
PPE Compliance Monitor — FastAPI Backend
Install: pip install fastapi uvicorn python-multipart ultralytics mediapipe opencv-python numpy pillow
Run:     uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import cv2
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import Optional

# ── Lazy model loading ────────────────────────────────────────────────────────
_yolo = "E:\ppe detection github\PPE-Kit-Detection-System\model.pt"
_pose = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8n.pt")          # auto-downloads on first run
    return _yolo

def get_pose():
    global _pose
    if _pose is None:
        import mediapipe as mp
        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
    return _pose

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="PPE Compliance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
PPE_COLORS = {
    "helmet":  ([20,  90, 180], [35, 255, 255]),   # yellow/orange HSV
    "vest":    ([5,   150, 150], [20, 255, 255]),   # orange HSV
    "gloves":  ([100, 80,  80],  [130, 255, 255]),  # blue HSV
    "mask":    ([0,   0,   200], [180, 30, 255]),   # white-ish HSV
    "goggles": ([90, 100,  50],  [130, 255, 200]),  # teal HSV
    "boots":   ([0,   0,   20],  [180, 60, 80]),    # dark HSV
}

HEAD_KEYPOINTS  = [0, 1, 2, 3, 4]          # MediaPipe: nose + eyes + ears
TORSO_KEYPOINTS = [11, 12, 23, 24]         # shoulders + hips


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_humans(img: np.ndarray, conf_threshold: float) -> list[dict]:
    """Run YOLOv8 — returns list of person bboxes."""
    results = get_yolo()(img, classes=[0], conf=conf_threshold, verbose=False)
    persons = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(float(box.conf[0]), 3),
            })
    return persons


def estimate_pose(img: np.ndarray, bbox: list[int]) -> dict:
    """Run MediaPipe Pose on a cropped person ROI."""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return {"valid": False, "keypoints": []}

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = get_pose().process(rgb_crop)

    if not result.pose_landmarks:
        return {"valid": False, "keypoints": []}

    h, w = crop.shape[:2]
    kpts = []
    for lm in result.pose_landmarks.landmark:
        kpts.append({
            "x": round(x1 + lm.x * w, 1),
            "y": round(y1 + lm.y * h, 1),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 3),
        })
    return {"valid": True, "keypoints": kpts}


def detect_ppe_color(img: np.ndarray, bbox: list[int],
                     required_ppe: list[str]) -> dict[str, dict]:
    """
    Novelty feature: Pose-based PPE verification using color heuristics
    in anatomically-correct zones (head ROI for helmet, torso ROI for vest).
    Replace the color heuristic with a custom YOLO PPE model for production.
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def roi_has_color(roi_coords, color_range):
        rx1, ry1, rx2, ry2 = [max(0, v) for v in roi_coords]
        roi = hsv[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return 0.0
        lo, hi = np.array(color_range[0]), np.array(color_range[1])
        mask = cv2.inRange(roi, lo, hi)
        ratio = np.count_nonzero(mask) / mask.size
        return round(float(ratio), 4)

    # Anatomical zones
    head_roi   = [x1, y1,           x2, y1 + int(h * 0.28)]
    torso_roi  = [x1, y1 + int(h * 0.28), x2, y1 + int(h * 0.62)]
    hands_roi  = [x1, y1 + int(h * 0.55), x2, y2]
    feet_roi   = [x1, y2 - int(h * 0.2),  x2, y2]

    zone_map = {
        "helmet":  head_roi,
        "vest":    torso_roi,
        "gloves":  hands_roi,
        "mask":    head_roi,
        "goggles": head_roi,
        "boots":   feet_roi,
    }

    results = {}
    for item in required_ppe:
        lo, hi = PPE_COLORS.get(item, ([0, 0, 0], [180, 255, 30]))
        ratio = roi_has_color(zone_map.get(item, torso_roi), (lo, hi))
        # Confidence scaled from color coverage + a baseline
        confidence = min(0.99, round(0.25 + ratio * 5, 2))
        results[item] = {
            "present": confidence > 0.50,
            "confidence": confidence,
        }
    return results


def pose_based_ppe_verification(pose: dict, ppe: dict[str, dict]) -> dict:
    """
    Novelty: Cross-check PPE detections against pose keypoint visibility.
    If head keypoints are occluded, flag helmet detection as uncertain.
    """
    if not pose["valid"] or not pose["keypoints"]:
        return {k: {**v, "pose_verified": False} for k, v in ppe.items()}

    kpts = pose["keypoints"]

    def avg_vis(indices):
        vis = [kpts[i]["visibility"] for i in indices if i < len(kpts)]
        return sum(vis) / len(vis) if vis else 0.0

    head_vis  = avg_vis(HEAD_KEYPOINTS)
    torso_vis = avg_vis(TORSO_KEYPOINTS)

    verified = {}
    for item, det in ppe.items():
        if item in ("helmet", "mask", "goggles"):
            certainty = det["confidence"] * (0.4 + 0.6 * head_vis)
        elif item in ("vest",):
            certainty = det["confidence"] * (0.4 + 0.6 * torso_vis)
        else:
            certainty = det["confidence"]
        verified[item] = {
            **det,
            "confidence": round(certainty, 3),
            "present": certainty > 0.50,
            "pose_verified": True,
        }
    return verified


def compliance_decision(persons: list[dict]) -> list[dict]:
    alerts = []
    for p in persons:
        missing = [k for k, v in p["ppe"].items() if not v["present"]]
        if not missing:
            p["compliant"] = True
            alerts.append({
                "severity": "ok",
                "message": f"Person #{p['id']} fully compliant",
                "detail": "  ".join(f"{k} ✓" for k in p["ppe"]),
            })
        else:
            p["compliant"] = False
            sev = "danger" if len(missing) > 1 else "warn"
            alerts.append({
                "severity": sev,
                "message": f"Person #{p['id']} missing {', '.join(missing)}",
                "detail": f"{'Pose estimation failed — ' if not p['pose_valid'] else ''}"
                          f"Missing: {', '.join(missing)}",
            })
    return alerts


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "PPE Compliance API"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.50),
    required_ppe: str = Form('["helmet","vest"]'),
):
    if not file.content_type.startswith(("image/", "video/")):
        raise HTTPException(400, "Only image files are supported currently")

    required = json.loads(required_ppe)
    data = await file.read()
    img = load_image(data)

    # ── Stage 1: Human detection ──────────────────────────────────────────────
    raw_persons = detect_humans(img, confidence_threshold)
    if not raw_persons:
        return JSONResponse({
            "persons": [],
            "alerts": [{"severity": "warn",
                        "message": "No humans detected",
                        "detail": "Try lowering the confidence threshold"}],
            "summary": {"total_persons": 0, "compliant": 0,
                        "violations": 0, "compliance_rate": 0},
        })

    persons = []
    for idx, p in enumerate(raw_persons):
        pid = idx + 1

        # ── Stage 2: Pose estimation ──────────────────────────────────────────
        pose = estimate_pose(img, p["bbox"])

        # ── Stage 3: PPE detection ────────────────────────────────────────────
        ppe_raw = detect_ppe_color(img, p["bbox"], required)

        # ── Stage 4: Pose-based PPE verification (novelty) ────────────────────
        ppe_verified = pose_based_ppe_verification(pose, ppe_raw)

        persons.append({
            "id": pid,
            "bbox": p["bbox"],
            "confidence": p["confidence"],
            "pose_valid": pose["valid"],
            "ppe": ppe_verified,
            "compliant": False,  # filled in stage 5
        })

    # ── Stage 5: Compliance decision ─────────────────────────────────────────
    alerts = compliance_decision(persons)

    n_compliant  = sum(1 for p in persons if p["compliant"])
    n_violations = len(persons) - n_compliant

    return {
        "persons": persons,
        "alerts": alerts,
        "summary": {
            "total_persons": len(persons),
            "compliant": n_compliant,
            "violations": n_violations,
            "compliance_rate": round(n_compliant / len(persons) * 100),
        },
    }


@app.post("/analyze/batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Form(0.50),
    required_ppe: str = Form('["helmet","vest"]'),
):
    """Run /analyze on multiple frames (e.g. video keyframes)."""
    results = []
    for f in files:
        single = await analyze(f, confidence_threshold, required_ppe)
        results.append({"filename": f.filename, "result": single})
    return {"batch": results, "frame_count": len(results)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
