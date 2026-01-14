import os
import cv2
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================
image_folder = "No-Glasses"
output_folder = "objectCrop/No-GlassesOutput"
model_path = "yolov8n-face.pt"

TARGET_CLASS_ID = 0   # Body
CONF_THRES = 0.2

os.makedirs(output_folder, exist_ok=True)

# =====================================================
# LOAD YOLO MODEL
# =====================================================
model = YOLO(model_path)

# =====================================================
# IMAGE LIST
# =====================================================
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"[INFO] Total images: {len(image_files)}")

# =====================================================
# MAIN LOOP
# =====================================================
for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[SKIP] Failed to read {img_name}")
        continue

    results = model(img, conf=CONF_THRES, verbose=False)[0]

    crop_idx = 0

    if results.boxes is None:
        print(f"[INFO] No detection: {img_name}")
        continue

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id != TARGET_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]

        out_name = f"{os.path.splitext(img_name)[0]}_body_{crop_idx}.jpg"
        cv2.imwrite(os.path.join(output_folder, out_name), crop)
        crop_idx += 1

    print(f"[OK] {img_name} -> {crop_idx} crop(s)")

print("[DONE] All images processed")
