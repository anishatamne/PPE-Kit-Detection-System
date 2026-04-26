import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import json
import numpy as np
import io
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="PPE Compliance API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ppe-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_PATH = "model.pt"
POSE_PATH = "yolov8n-pose.pt"

obj_model = YOLO(YOLO_PATH)
pose_model = YOLO(POSE_PATH)

HEAD_KP = [0, 1, 2, 3, 4]
TORSO_KP = [11, 12, 23, 24]

PPE_ITEMS = ["helmet", "vest", "gloves", "boots", "goggles"]

CLASS_MAP = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_all(img, conf):
    res = obj_model(img, conf=conf, verbose=False)

    persons = []
    objects = []

    for r in res:
        for b in r.boxes:
            cls = int(b.cls[0])
            name = CLASS_MAP.get(cls, "unknown")

            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            confidence = round(float(b.conf[0]), 3)

            item = {
                "class": name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            }

            if cls == 6:
                persons.append(item)
            else:
                objects.append(item)

    return persons, objects


def estimate_pose(img, bbox=None, conf=0.25):
    results = pose_model(img, conf=conf, verbose=False)

    if not results or results[0].keypoints is None:
        return {"valid": False, "keypoints": []}

    kpts_xy = results[0].keypoints.xy
    kpts_conf = results[0].keypoints.conf

    if kpts_xy is None or len(kpts_xy) == 0:
        return {"valid": False, "keypoints": []}

    selected_idx = 0

    if bbox:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        min_dist = float("inf")

        for i, person_kpts in enumerate(kpts_xy):
            valid_points = [
                (x, y) for (x, y), c in zip(person_kpts, kpts_conf[i]) if c > 0.3
            ]

            if not valid_points:
                continue

            px = sum(p[0] for p in valid_points) / len(valid_points)
            py = sum(p[1] for p in valid_points) / len(valid_points)

            dist = (px - cx) ** 2 + (py - cy) ** 2

            if dist < min_dist:
                min_dist = dist
                selected_idx = i

    selected_xy = kpts_xy[selected_idx]
    selected_conf = kpts_conf[selected_idx]

    kpts = []
    valid_count = 0

    for (x, y), c in zip(selected_xy, selected_conf):
        visibility = float(c)
        if visibility > 0.3:
            valid_count += 1

        kpts.append({
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "visibility": round(visibility, 3)
        })

    return {
        "valid": valid_count >= 5,
        "keypoints": kpts
    }


def detect_ppe(img, person_bbox, detections, pose):
    def center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    px, py = center(person_bbox)

    head_region = None
    if pose["valid"]:
        kpts = pose["keypoints"]
        head_pts = [kpts[i] for i in HEAD_KP if i < len(kpts)]
        if head_pts:
            hx = [p["x"] for p in head_pts]
            hy = [p["y"] for p in head_pts]
            head_region = [min(hx), min(hy), max(hx), max(hy)]

    results = {}

    for item in PPE_ITEMS:
        present = False
        conf = 0.0

        for d in detections:
            if d["class"] != item:
                continue

            dx, dy = center(d["bbox"])

            if item == "helmet" and head_region:
                x1, y1, x2, y2 = head_region
                if not (x1 <= dx <= x2 and y1 <= dy <= y2):
                    continue
            else:
                dist = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5
                if dist > 200:
                    continue

            present = True
            conf = max(conf, d["confidence"])

        neg_class = f"no_{item}"
        for d in detections:
            if d["class"] == neg_class:
                dx, dy = center(d["bbox"])
                dist = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5

                if dist < 150:
                    present = False
                    conf = max(conf, d["confidence"])

        results[item] = {
            "present": present,
            "confidence": round(conf, 2),
        }

    return results


def verify_with_pose(pose, ppe):
    if not pose["valid"]:
        return {k: {**v, "pose_verified": False} for k, v in ppe.items()}

    kpts = pose["keypoints"]

    def avg_vis(indices):
        vals = [kpts[i]["visibility"] for i in indices if i < len(kpts)]
        return sum(vals) / len(vals) if vals else 0

    hv = avg_vis(HEAD_KP)
    tv = avg_vis(TORSO_KP)

    out = {}

    for item, det in ppe.items():
        present = det["present"]
        conf = det["confidence"]

        if item in ("helmet", "goggles") and hv < 0.2:
            present = False

        if item == "vest" and tv < 0.2:
            present = False

        out[item] = {
            "present": present,
            "confidence": conf,
            "pose_verified": True,
            "pose_score": round(hv if item != "vest" else tv, 3)
        }

    return out


def make_alerts(persons):
    alerts = []

    for p in persons:
        missing = [k for k, v in p["ppe"].items() if not v["present"]]
        p["compliant"] = not missing

        if not missing:
            alerts.append({
                "severity": "ok",
                "message": f"Person #{p['id']} fully compliant",
                "detail": "  ".join(f"{k} ✓" for k in p["ppe"]),
            })
        else:
            sev = "danger" if len(missing) > 1 else "warn"
            alerts.append({
                "severity": sev,
                "message": f"Person #{p['id']} missing {', '.join(missing)}",
                "detail": ("Pose occluded — " if not p["pose_valid"] else "") +
                          f"Missing: {', '.join(missing)}",
            })

    return alerts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.50),
):
    start_time = datetime.now()

    img_bytes = await file.read()
    img = load_image(img_bytes)

    persons_raw, detections = detect_all(img, confidence_threshold)

    if not persons_raw:
        return {
            "persons": [],
            "alerts": [{
                "severity": "warn",
                "message": "No humans detected",
                "detail": "Try lowering threshold",
            }],
            "summary": {
                "total_persons": 0,
                "compliant": 0,
                "violations": 0,
                "compliance_rate": 0,
            },
        }

    persons = []

    for i, p in enumerate(persons_raw):
        pose = estimate_pose(img, p["bbox"])
        ppe = detect_ppe(img, p["bbox"], detections, pose)
        ppe = verify_with_pose(pose, ppe)

        persons.append({
            "id": i + 1,
            "bbox": p["bbox"],
            "confidence": p["confidence"],
            "pose_valid": pose["valid"],
            "keypoints": pose["keypoints"],
            "ppe": ppe,
            "compliant": False,
        })

    alerts = make_alerts(persons)
    n_ok = sum(p["compliant"] for p in persons)

    return {
        "persons": persons,
        "alerts": alerts,
        "summary": {
            "total_persons": len(persons),
            "compliant": n_ok,
            "violations": len(persons) - n_ok,
            "compliance_rate": round(n_ok / len(persons) * 100),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)