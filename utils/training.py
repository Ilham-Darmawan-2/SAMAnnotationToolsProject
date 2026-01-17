"""
Training and inference functions with argparse
"""
import os
import shutil
import numpy as np
import torch
import gc
import argparse

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work.", e)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Script")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root folder of dataset (contains images/ and labels/)"
    )

    parser.add_argument(
        "--images_folder",
        type=str,
        required=True,
        help="Folder containing images for training"
    )

    parser.add_argument(
        "--classlist",
        nargs="+",
        required=True,
        help="List of class names, example: --classlist person car motor"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="best.pt",
        help="Path to save or load YOLO model"
    )

    parser.add_argument(
        "--model_folder",
        type=str,
        default="runs",
        help="Folder to store training results"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size"
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Train/val split ratio"
    )

    return parser.parse_args()


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

    train_img_dir = os.path.join(root, "train/images")
    train_lbl_dir = os.path.join(root, "train/labels")
    val_img_dir = os.path.join(root, "val/images")
    val_lbl_dir = os.path.join(root, "val/labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

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


def train_model(args):
    """Train YOLO model using argparse inputs"""

    inference_images = args.images_folder
    inference_root = args.dataset_root
    CLASSLIST = args.classlist
    model_path = args.model_path
    model_folder = args.model_folder

    images_infer = [f for f in os.listdir(inference_images)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(images_infer) < 10:
        print("[INFO] Not enough images to train (min 10 required).")
        return

    print("[INFO] Splitting dataset...")
    train_dir, val_dir = split_train_val(inference_root, ratio=args.ratio)

    if train_dir is None or val_dir is None:
        print("[ERROR] Dataset split failed.")
        return

    train_abs = os.path.abspath(train_dir)
    val_abs = os.path.abspath(val_dir)

    yaml_path = os.path.join(inference_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_abs}\n")
        f.write(f"val: {val_abs}\n")
        f.write(f"nc: {len(CLASSLIST)}\n")
        f.write("names: [" + ", ".join([f"'{n}'" for n in CLASSLIST]) + "]\n")

    print("[INFO] YAML created:", yaml_path)

    # Determine which model will be used
    if os.path.exists(model_path):
        init_model = model_path

        print("\n" + "="*70)
        print("[INFO] USING EXISTING CUSTOM MODEL FOR TRAINING")
        print("Model path :", model_path)
        print("This training will CONTINUE from your custom model weights.")
        print("="*70 + "\n")

    else:
        init_model = "yolo11s.pt"

        print("\n" + "="*70)
        print("[INFO] CUSTOM MODEL NOT FOUND")
        print("Path checked :", model_path)
        print("Falling back to default pretrained YOLO model: yolo11s.pt")
        print("="*70 + "\n")

    try:
        model = YOLO(init_model)
        print("[INFO] Starting training...")

        model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=640,
            batch=args.batch,
            optimizer="SGD",
            lr0=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1,
            project=model_folder,
            name="train_run",
            exist_ok=True,
            device=0,
            amp=True,
            workers=12
        )

        model.save(model_path)
        print(f"[INFO] Training finished. Model saved to {model_path}")

    except Exception as e:
        print("[ERROR] Training failed:", e)

    try:
        print("[INFO] Cleaning temporary train/val folders...")
        shutil.rmtree(os.path.join(inference_root, "train"))
        shutil.rmtree(os.path.join(inference_root, "val"))
        print("[INFO] train/ & val/ removed successfully.")
    except Exception as e:
        print("[WARN] Failed to delete train/val folders:", e)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
