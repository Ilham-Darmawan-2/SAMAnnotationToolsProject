"""
Training and inference functions
"""
import os
import shutil
import numpy as np
import torch
import gc
import cv2

from .config import model_path, CLASSLIST, state, input_folder

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work.", e)

def inference_current(images, current_index, conf=0.5):
    """Run inference on current image"""
    if not os.path.exists(model_path):
        print("[INFO] Model assistant does not exist.")
        return

    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)

    model = YOLO(model_path)

    with torch.no_grad():
        results = model.predict(orig_img, conf=conf, iou=0.4)

    # Process boxes
    pred_boxes = []
    for r in results:
        if hasattr(r, "boxes"):
            for box in r.boxes:
                x1 = int(box.xyxy[0, 0].item())
                y1 = int(box.xyxy[0, 1].item())
                x2 = int(box.xyxy[0, 2].item())
                y2 = int(box.xyxy[0, 3].item())
                cls_idx = int(box.cls[0].item())
                cls_name = CLASSLIST[cls_idx] if cls_idx < len(CLASSLIST) else str(cls_idx)

                pred_boxes.append([
                    int(round(x1 * state.display_scale)),
                    int(round(y1 * state.display_scale)),
                    int(round(x2 * state.display_scale)),
                    int(round(y2 * state.display_scale)),
                    cls_name
                ])

    state.bboxes = pred_boxes
    print("[INFO] Inference saved and bboxes updated.")

    # Cleanup
    del results
    torch.cuda.empty_cache()
    gc.collect()