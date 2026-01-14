import os
import xml.etree.ElementTree as ET

def filter_pascal_voc_annotations(xml_folder, classes_to_keep):
    """
    Hapus object di XML Pascal VOC yang class-nya tidak ada di classes_to_keep.

    Parameters:
        xml_folder (str)       : Folder tempat XML annotation.
        classes_to_keep (list) : List class yang mau dipertahankan.
    """
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        objects = root.findall('object')
        removed_count = 0
        for obj in objects:
            class_name = obj.find('name').text
            if class_name in classes_to_keep:
                root.remove(obj)
                removed_count += 1

        if removed_count > 0:
            tree.write(xml_path)
            print(f"{xml_file}: {removed_count} object dihapus")

# ===== Contoh pemakaian =====
xml_folder = "output/PPETambahan"
classes_to_keep = ["Gloves"]
filter_pascal_voc_annotations(xml_folder, classes_to_keep)
