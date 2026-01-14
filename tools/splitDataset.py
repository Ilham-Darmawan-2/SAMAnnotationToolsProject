import os
import shutil
import random
import xml.etree.ElementTree as ET

# ================================
# CONFIG
# ================================
workspaceName = "vehicle"
input_root = "datasetsInput"
input_xml_root = f"output/{workspaceName}"
output_root = "datasetsOutput"

train_ratio = 0.7
seed = 42
random.seed(seed)

# ================================
# Helper
# ================================
def sanitize_name(name):
    return name.replace(" ", "_")

def fix_xml(xml_path, new_img_name):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    changed = False

    # update <filename>
    filename = root.find("filename")
    if filename is None:
        filename = ET.SubElement(root, "filename")
    if filename.text != new_img_name:
        filename.text = new_img_name
        changed = True

    # fix objects
    for obj in root.findall("object"):
        tags = {c.tag: c for c in obj}

        if "pose" not in tags:
            pose = ET.Element("pose")
            pose.text = "Unspecified"
            obj.insert(1, pose)
            changed = True

        if "truncated" not in tags:
            truncated = ET.Element("truncated")
            truncated.text = "0"
            obj.insert(2, truncated)
            changed = True

        if "difficult" not in tags:
            difficult = ET.Element("difficult")
            difficult.text = "0"
            obj.insert(3, difficult)
            changed = True

    if changed:
        tree.write(xml_path)

# ================================
# Ambil folder gambar
# ================================
folders = [
    os.path.join(input_root, f)
    for f in os.listdir(input_root)
    if f.startswith(workspaceName + "-")
    and os.path.isdir(os.path.join(input_root, f))
]

folders.sort()
print(f"Folder ditemukan: {len(folders)}")

# ================================
# Pair & FIX
# ================================
all_data = []

for folder in folders:
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            old_img_path = os.path.join(folder, file)
            new_img_name = sanitize_name(file)
            new_img_path = os.path.join(folder, new_img_name)

            # rename image if needed
            if file != new_img_name:
                os.rename(old_img_path, new_img_path)

            xml_old_name = os.path.splitext(file)[0] + ".xml"
            xml_new_name = os.path.splitext(new_img_name)[0] + ".xml"

            xml_old_path = os.path.join(input_xml_root, xml_old_name)
            xml_new_path = os.path.join(input_xml_root, xml_new_name)

            if not os.path.exists(xml_old_path):
                print(f"[WARNING] XML tidak ditemukan: {file}")
                continue

            if xml_old_name != xml_new_name:
                os.rename(xml_old_path, xml_new_path)

            # fix xml content
            fix_xml(xml_new_path, new_img_name)

            all_data.append((new_img_path, xml_new_path))

print(f"Total pasangan valid: {len(all_data)}")

# ================================
# Shuffle + split
# ================================
random.shuffle(all_data)
train_count = int(len(all_data) * train_ratio)

train_data = all_data[:train_count]
val_data = all_data[train_count:]

print(f"Train: {len(train_data)}")
print(f"Val  : {len(val_data)}")

# ================================
# Prepare output
# ================================
train_dir = os.path.join(output_root, workspaceName, "train")
val_dir   = os.path.join(output_root, workspaceName, "valid")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def copy_dataset(data, out_dir):
    for img_src, xml_src in data:
        shutil.copy(img_src, os.path.join(out_dir, os.path.basename(img_src)))
        shutil.copy(xml_src, os.path.join(out_dir, os.path.basename(xml_src)))

copy_dataset(train_data, train_dir)
copy_dataset(val_data, val_dir)

print("✅ DONE: XML rapi, nama aman, dataset siap YOLOX")
