import os
import xml.etree.ElementTree as ET

# ================= CONFIG =================
INPUT_XML_DIR = "output/ppeKujangv2"
OUTPUT_XML_DIR = "output/ppeKujangv2v2"

TARGET_CLASS = "Safety-Helmet"
CROP_RATIO = 0.4   # 25% dari BAWAH bbox

os.makedirs(OUTPUT_XML_DIR, exist_ok=True)

# ================= MAIN =================
for xml_file in os.listdir(INPUT_XML_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(INPUT_XML_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    modified = False

    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls != TARGET_CLASS:
            continue

        bnd = obj.find("bndbox")

        ymin = int(bnd.find("ymin").text)
        ymax = int(bnd.find("ymax").text)

        bbox_h = ymax - ymin
        crop_pixels = int(bbox_h * CROP_RATIO)

        new_ymax = ymax - crop_pixels

        # safety check
        if new_ymax <= ymin:
            print(f"[SKIP] bbox terlalu kecil di {xml_file}")
            continue

        bnd.find("ymax").text = str(new_ymax)
        modified = True

    if modified:
        out_path = os.path.join(OUTPUT_XML_DIR, xml_file)
        tree.write(out_path)
        print(f"✅ Updated XML: {xml_file}")
    else:
        print(f"⚠️ No target class in: {xml_file}")

print("🔥 DONE — XML bbox di-crop dari bawah ke atas.")
