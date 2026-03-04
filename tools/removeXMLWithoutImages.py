import os
import xml.etree.ElementTree as ET

workspace_name = "ppeKujangv2"

xml_dir = os.path.join("output", workspace_name)
images_root = "datasetsInput"

# 1. Kumpulin semua image yang ada di SEMUA folder image
all_images = set()

for folder in os.listdir(images_root):
    if folder.startswith(f"{workspace_name}-"):
        folder_path = os.path.join(images_root, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                all_images.add(f)

print(f"Total image ditemukan: {len(all_images)}")

# 2. Cek satu-satu xml
deleted = 0

for xml_file in os.listdir(xml_dir):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_dir, xml_file)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_tag = root.find("filename")

        if filename_tag is not None and filename_tag.text:
            image_name = filename_tag.text
        else:
            # fallback: nama xml tapi beda ekstensi
            image_name = os.path.splitext(xml_file)[0]

    except Exception as e:
        print(f"[ERROR PARSE] {xml_file}: {e}")
        continue

    # 3. Kalau image gak ada di folder manapun → hapus xml
    if image_name not in all_images:
        os.remove(xml_path)
        deleted += 1
        print(f"[DELETED] {xml_file} (image {image_name} tidak ditemukan)")

print(f"Selesai. Total XML dihapus: {deleted}")
