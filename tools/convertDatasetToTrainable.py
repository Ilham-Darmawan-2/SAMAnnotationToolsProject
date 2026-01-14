import os
import shutil

# ================================
# CONFIG
# ================================
DATASET_ROOT = "datasetsOutput/vehicle"

YOLOX_ROOT = "VOCdevkit/VOC2012"
IMAGESETS_DIR = os.path.join(YOLOX_ROOT, "ImageSets", "Main")
JPEG_DIR = os.path.join(YOLOX_ROOT, "JPEGImages")
ANNOT_DIR = os.path.join(YOLOX_ROOT, "Annotations")

SPLITS = ["train", "valid"]
IMAGE_EXTS = (".jpg", ".png", ".jpeg")

# ================================
# Prepare folders
# ================================
os.makedirs(IMAGESETS_DIR, exist_ok=True)
os.makedirs(JPEG_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

# ================================
# Function: buat train.txt / valid.txt
# ================================
def create_split_txt(split_name):
    ann_folder = os.path.join(DATASET_ROOT, split_name)
    txt_path = os.path.join(IMAGESETS_DIR, f"{split_name}.txt")

    file_names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(ann_folder)
        if f.endswith(".xml")
    )

    with open(txt_path, "w") as f:
        for name in file_names:
            f.write(name + "\n")

    print(f"[OK] {split_name}.txt dibuat ({len(file_names)} file)")

# ================================
# Function: pindahkan image & xml
# ================================
def move_files(split_name):
    src_folder = os.path.join(DATASET_ROOT, split_name)

    for file_name in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file_name)

        if file_name.lower().endswith(IMAGE_EXTS):
            shutil.copy(src_path, os.path.join(JPEG_DIR, file_name))

        elif file_name.lower().endswith(".xml"):
            shutil.copy(src_path, os.path.join(ANNOT_DIR, file_name))

    print(f"[OK] File {split_name} dipindahkan")

# ================================
# EKSEKUSI
# ================================
for split in SPLITS:
    create_split_txt(split)
    move_files(split)

# ================================
# Hapus folder dataset root
# ================================
if os.path.exists(DATASET_ROOT):
    shutil.rmtree(DATASET_ROOT)
    print(f"[CLEAN] Folder {DATASET_ROOT} berhasil dihapus")

print("\n🔥 Dataset VOC siap dipakai YOLOX")
