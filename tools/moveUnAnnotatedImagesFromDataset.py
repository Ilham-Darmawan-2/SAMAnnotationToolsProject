import os
import shutil

# =====================
# CONFIG
# =====================
FOLDER_A = "datasetsInput/markerv2-7"     # misal: datasets/images
FOLDER_B = "output/markerv2"        # misal: datasets/annotations
FOLDER_C = "temp/markerv2"     # misal: datasets/no_xml_images

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Pastikan folder C ada
os.makedirs(FOLDER_C, exist_ok=True)

# =====================
# PROCESS
# =====================
for img_name in os.listdir(FOLDER_A):
    if img_name.lower().endswith(IMAGE_EXTS):
        base_name = os.path.splitext(img_name)[0]
        xml_name = base_name + ".xml"
        xml_path = os.path.join(FOLDER_B, xml_name)

        # Jika XML tidak ada, pindahkan gambar
        if not os.path.exists(xml_path):
            src_img = os.path.join(FOLDER_A, img_name)
            dst_img = os.path.join(FOLDER_C, img_name)
            shutil.move(src_img, dst_img)
            print(f"Pindah: {img_name} (XML tidak ditemukan)")
