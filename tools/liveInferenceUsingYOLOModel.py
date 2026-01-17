#!/usr/bin/env python3
"""
Live YOLO inference with OpenCV (webcam)
- Uses ultralytics.YOLO
- Skips frames for faster real-time (SKIP_FRAMES)
- Resizes frames to max height DISPLAY_H (720) and runs inference on resized frame
- Draws boxes, class name and confidence
"""

import cv2
import time
import os
import sys
from ultralytics import YOLO  # make sure ultralytics installed in same env
import numpy as np

# ---------- CONFIG ----------
CAM_INDEX = "vehicle-cctv-4.mp4"
SKIP_FRAMES = 2        # 2 => process every (SKIP_FRAMES + 1)th frame (every 3rd frame)
DISPLAY_H = 720        # max height for display/inference
MODEL_PATH = "models/vehicle/modelAssistant.pt"  # change to your model (or pretrained)
CONF_THRESHOLD = 0.4
DEVICE = "cuda" if (cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else False) else "cpu"
# Fallback model name if MODEL_PATH doesn't exist (you can change)
FALLBACK_MODEL = "yolov11s.pt"  # or "yolov8s.pt" if you prefer

OUT_SNAP_DIR = "snapshots"
os.makedirs(OUT_SNAP_DIR, exist_ok=True)

# ---------- Load model ----------
if os.path.exists(MODEL_PATH):
    model_file = MODEL_PATH
    print(f"[INFO] Using model: {model_file}")
else:
    print(f"[INFO] Model not found at '{MODEL_PATH}', trying fallback '{FALLBACK_MODEL}'")
    model_file = FALLBACK_MODEL

try:
    model = YOLO(model_file)
except Exception as e:
    print("[ERROR] Failed to load YOLO model:", e)
    sys.exit(1)

# ---------- helper: draw box ----------
def draw_box(img, x1, y1, x2, y2, label=None, conf=None, color=(0,255,0), thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label is not None:
        text = f"{label}"
        if conf is not None:
            text += f" {conf:.2f}"
        # text background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ---------- camera ----------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print(f"[ERROR] Cannot open camera index {CAM_INDEX}")
    sys.exit(1)

frame_count = 0
proc_count = 0
last_proc_time = time.time()
fps_display = 0.0
paused = False

print("[INFO] Starting live inference. Press 'q' or ESC to quit, 's' to save snapshot, 'p' to pause.")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed, retrying...")
            time.sleep(0.1)
            continue
        frame_count += 1
    else:
        # if paused, still show same frame
        time.sleep(0.05)
        # display paused state on the current frame
        if 'frame' in locals():
            disp = frame.copy()
            cv2.putText(disp, "PAUSED - press p to resume", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow("YOLO Live", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            paused = False
        elif key in [27, ord('q')]:
            break
        continue

    # resize for speed & display
    h, w = frame.shape[:2]
    scale = DISPLAY_H / h if h > DISPLAY_H else 1.0
    if scale != 1.0:
        inference_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        inference_frame = frame

    # do inference only every (SKIP_FRAMES+1)th frame
    if (frame_count % (SKIP_FRAMES + 1)) == 0:
        t0 = time.time()
        try:
            # predict returns list of Results; pass inference_frame directly (BGR numpy)
            results = model.predict(inference_frame, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
        except Exception as e:
            print("[ERROR] Inference failed:", e)
            results = []

        proc_count += 1
        t1 = time.time()
        last_proc_time = t1 - t0
        # compute FPS over processed frames (approx)
        if last_proc_time > 0:
            fps_display = 1.0 / last_proc_time

        # prepare overlay
        disp = inference_frame.copy()

        # iterate results and draw
        # results may be list-like, usually single-element per image
        try:
            for r in results:
                # r.boxes is an object; iterate boxes
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        # robust extraction of xyxy, cls, conf
                        try:
                            # earlier pattern works: box.xyxy[0,0]...
                            x1 = int(box.xyxy[0,0].item())
                            y1 = int(box.xyxy[0,1].item())
                            x2 = int(box.xyxy[0,2].item())
                            y2 = int(box.xyxy[0,3].item())
                        except Exception:
                            # fallback: convert to numpy
                            try:
                                arr = box.xyxy.cpu().numpy()[0]
                                x1,y1,x2,y2 = arr.astype(int).tolist()
                            except Exception:
                                continue
                        # class and conf
                        try:
                            cls_idx = int(box.cls[0].item())
                        except Exception:
                            cls_idx = None
                        try:
                            conf = float(box.conf[0].item())
                        except Exception:
                            conf = None

                        # label name if model.names exists
                        label = None
                        try:
                            names = model.names if hasattr(model, "names") else None
                            if names is not None and cls_idx is not None and cls_idx in names:
                                label = names[int(cls_idx)]
                            elif cls_idx is not None:
                                label = str(cls_idx)
                        except Exception:
                            label = str(cls_idx) if cls_idx is not None else None

                        # draw
                        color = (0, 255, 0)
                        draw_box(disp, x1, y1, x2, y2, label=label, conf=conf, color=color)
        except Exception as e:
            print("[WARN] Exception drawing boxes:", e)

    else:
        # reuse previous 'disp' (if exists) or just use inference_frame
        if 'disp' in locals():
            disp = disp
        else:
            disp = inference_frame.copy()

    # overlay info
    cv2.putText(disp, f"FPS(proc): {fps_display:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(disp, f"Frame: {frame_count}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.imshow("YOLO Live", disp)

    key = cv2.waitKey(30) & 0xFF
    if key in [27, ord('q')]:
        break
    elif key == ord('s'):
        # save snapshot of current displayed (resized) frame
        snap_name = os.path.join(OUT_SNAP_DIR, f"snap_{int(time.time())}.jpg")
        cv2.imwrite(snap_name, disp)
        print(f"[INFO] Snapshot saved: {snap_name}")
    elif key == ord('p'):
        paused = True

# cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Exited.")
