import os
import shutil
import xml.etree.ElementTree as ET

# ====== CONFIG ======
IMAGE_DIR = "datasetsInput/marker-1"
ANNOTATION_DIR = "output/marker"
OUTPUT_DIR = "folder_C"   # tujuan copy
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
# ====================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_annotation_empty(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = root.findall("object")
    return len(objects) == 0

copied = 0

for xml_file in os.listdir(ANNOTATION_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)

    if is_annotation_empty(xml_path):
        base_name = os.path.splitext(xml_file)[0]

        # cari gambar dengan nama sama
        for ext in IMAGE_EXTENSIONS:
            img_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(img_path):
                shutil.copy(img_path, OUTPUT_DIR)
                copied += 1
                print(f"[COPIED] {img_path}")
                break
        else:
            print(f"[WARNING] Gambar tidak ditemukan untuk {base_name}")

print(f"\nSelesai. Total gambar null dicopy: {copied}")
