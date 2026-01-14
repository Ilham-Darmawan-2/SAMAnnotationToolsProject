import os
import xml.etree.ElementTree as ET

def rename_class_in_xml(folder, old_class, new_class):
    xml_files = [f for f in os.listdir(folder) if f.endswith(".xml")]

    if not xml_files:
        print("Tidak ada file XML ditemukan.")
        return

    for xml_name in xml_files:
        xml_path = os.path.join(folder, xml_name)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            changed = False

            # Loop semua <object>
            for obj in root.findall("object"):
                name_tag = obj.find("name")
                if name_tag is not None and name_tag.text == old_class:
                    name_tag.text = new_class
                    changed = True

            if changed:
                tree.write(xml_path)
                print(f"[UPDATED] {xml_name}")
            else:
                print(f"[SKIPPED] {xml_name} (tidak ada class '{old_class}')")

        except Exception as e:
            print(f"[ERROR] {xml_name}: {e}")


# ==============================
# Cara pakai
# ==============================
folder_xml = "output/vehicle"   # folder tempat xml
old_class = "motorcycle"            # class lama
new_class = "motorbike"       # class baru

rename_class_in_xml(folder_xml, old_class, new_class)
