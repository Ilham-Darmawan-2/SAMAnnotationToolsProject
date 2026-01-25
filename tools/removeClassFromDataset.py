import os
import xml.etree.ElementTree as ET

def filter_pascal_voc_annotations(xml_folder, classes_to_remove):
    """
    Hapus object di XML Pascal VOC yang class-nya ada di classes_to_remove.

    Parameters:
        xml_folder (str)       : Folder tempat XML annotation.
        classes_to_remove (list) : List class yang mau dihapus.
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
            if class_name in classes_to_remove:
                root.remove(obj)
                removed_count += 1

        if removed_count > 0:
            tree.write(xml_path)
            print(f"{xml_file}: {removed_count} object dihapus")

# ===== Contoh pemakaian =====
xml_folder = "output/markerv2"
classes_to_remove = ["Object"]
filter_pascal_voc_annotations(xml_folder, classes_to_remove)
