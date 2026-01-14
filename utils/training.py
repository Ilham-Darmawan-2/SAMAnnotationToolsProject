"""
Training and inference functions
"""
import os
import shutil
import numpy as np
import torch
import gc
import cv2

from .config import (inference_root, inference_images, inference_labels, 
                     model_folder, model_path, CLASSLIST, state, input_folder)

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work.", e)

def split_train_val(root, ratio=0.7):
    """Split dataset into train and validation sets"""
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")

    imgs = [f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(imgs) < 2:
        print("[WARN] Not enough images to split train/val.")
        return None, None

    np.random.shuffle(imgs)

    train_count = int(len(imgs) * ratio)
    train_imgs = imgs[:train_count]
    val_imgs = imgs[train_count:]

    # Create directories
    train_img_dir = os.path.join(root, "train/images")
    train_lbl_dir = os.path.join(root, "train/labels")
    val_img_dir = os.path.join(root, "val/images")
    val_lbl_dir = os.path.join(root, "val/labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy files
    for img in train_imgs:
        base = os.path.splitext(img)[0]
        txt = base + ".txt"
        shutil.copy2(os.path.join(images_dir, img), os.path.join(train_img_dir, img))
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(train_lbl_dir, txt))

    for img in val_imgs:
        base = os.path.splitext(img)[0]
        txt = base + ".txt"
        shutil.copy2(os.path.join(images_dir, img), os.path.join(val_img_dir, img))
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(val_lbl_dir, txt))

    return train_img_dir, val_img_dir

def train_model():
    """Train YOLO model in background"""
    if state.training_running:
        print("[INFO] Training already running.")
        return
    state.training_running = True

    # Check image count
    images_infer = [f for f in os.listdir(inference_images)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(images_infer) < 10:
        print("[INFO] Not enough images to train (min 10 required).")
        state.training_running = False
        return

    # Split dataset
    print("[INFO] Splitting dataset...")
    train_dir, val_dir = split_train_val(inference_root, ratio=0.5)

    if train_dir is None or val_dir is None:
        print("[ERROR] Dataset split failed.")
        state.training_running = False
        return

    # Absolute paths
    train_abs = os.path.abspath(train_dir)
    val_abs = os.path.abspath(val_dir)

    # Write YAML
    yaml_path = os.path.join(inference_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_abs}\n")
        f.write(f"val: {val_abs}\n")
        f.write(f"nc: {len(CLASSLIST)}\n")
        f.write("names: [" + ", ".join([f"'{n}'" for n in CLASSLIST]) + "]\n")

    print("[INFO] YAML created:", yaml_path)

    # Initialize model
    init_model = model_path if os.path.exists(model_path) else "yolo11s.pt"

    try:
        model = YOLO(init_model)
        print("[INFO] Starting training...")

        model.train(
            data=yaml_path,
            epochs=5,
            imgsz=640,
            batch=4,
            optimizer="SGD",
            lr0=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1,
            project=model_folder,
            name="train_run",
            exist_ok=True
        )

        model.save(model_path)
        print(f"[INFO] Training finished. Model saved to {model_path}")

    except Exception as e:
        print("[ERROR] Training failed:", e)

    # Cleanup
    try:
        print("[INFO] Cleaning temporary train/val folders...")
        shutil.rmtree(os.path.join(inference_root, "train"))
        shutil.rmtree(os.path.join(inference_root, "val"))
        print("[INFO] train/ & val/ removed successfully.")
    except Exception as e:
        print("[WARN] Failed to delete train/val folders:", e)

    state.training_running = False
    del model
    torch.cuda.empty_cache()
    gc.collect()

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