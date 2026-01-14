import os
import cv2
import xml.etree.ElementTree as ET

# ==============================
# CONFIG
# ==============================
IMG_DIR = "datasetsInput/ppeKujangv2-1"                      # folder gambar
XML_DIR = "output/ppeKujangv2"                      # folder XML Pascal VOC
OUTPUT_DIR = "objectCrop"          # folder output crop
target_classes = ["Face"]  # isi target class bebas

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# FUNGSI UTAMA
# ==============================
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append((name, xmin, ymin, xmax, ymax))
    return objects

# ==============================
# LOOP SEMUA XML
# ==============================
for xml_file in os.listdir(XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(XML_DIR, xml_file)
    base_name = os.path.splitext(xml_file)[0]

    # gambar harus punya nama yang sama dengan XML
    img_path = os.path.join(IMG_DIR, base_name + ".jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, base_name + ".png")
    if not os.path.exists(img_path):
        print(f"[SKIP] Gambar tidak ditemukan untuk {xml_file}")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    objects = parse_voc_xml(xml_path)

    for idx, (cls, xmin, ymin, xmax, ymax) in enumerate(objects):

        # Skip kalau bukan target class
        if cls not in target_classes:
            continue

        # Bikin folder class
        save_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(save_dir, exist_ok=True)

        # Clamp biar aman
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        crop = img[ymin:ymax, xmin:xmax]

        save_path = os.path.join(save_dir, f"{base_name}_{idx}.jpg")
        cv2.imwrite(save_path, crop)

        print(f"[SAVE] {save_path}")
