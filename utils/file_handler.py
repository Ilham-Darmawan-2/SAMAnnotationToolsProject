"""
File handling functions for VOC and YOLO formats
"""
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from .config import output_folder, inference_labels, inference_images, input_folder, CLASSLIST, state

def prettify_xml(elem):
    """Convert XML to pretty-printed string"""
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="   ")

def save_pascal_voc(img_name, img_shape):

    """Save annotations in Pascal VOC format"""
    xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + ".xml")
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "ppeKujangv2"
    ET.SubElement(ann, "filename").text = img_name
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2] if len(img_shape) > 2 else 3)
    
    for bbox in state.bboxes:
        x1 = int(round(bbox[0] / state.display_scale))
        y1 = int(round(bbox[1] / state.display_scale))
        x2 = int(round(bbox[2] / state.display_scale))
        y2 = int(round(bbox[3] / state.display_scale))
        cls = bbox[4]
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(max(0, x1))
        ET.SubElement(bnd, "ymin").text = str(max(0, y1))
        ET.SubElement(bnd, "xmax").text = str(max(0, x2))
        ET.SubElement(bnd, "ymax").text = str(max(0, y2))
    
    with open(xml_path, "w") as f:
        f.write(prettify_xml(ann))
    print(f"[INFO] Saved VOC: {xml_path}")

def save_yolo_label_and_image(img_name, orig_img, classList):
    """Save YOLO format labels and copy image"""
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(inference_labels, base + ".txt")
    dest_img = os.path.join(inference_images, img_name)
    h, w = orig_img.shape[:2]
    lines = []
    
    for bbox in state.bboxes:
        x1 = int(round(bbox[0] / state.display_scale))
        y1 = int(round(bbox[1] / state.display_scale))
        x2 = int(round(bbox[2] / state.display_scale))
        y2 = int(round(bbox[3] / state.display_scale))
        cls = bbox[4]
        if cls not in classList:
            continue
        idx = classList.index(cls)
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    shutil.copy2(os.path.join(input_folder, img_name), dest_img)
    print(f"[INFO] Saved YOLO label: {label_path}")

def load_annotation_local(img_name_local):
    """Load annotations from VOC XML file"""
    xml_path = os.path.join(output_folder, os.path.splitext(img_name_local)[0] + ".xml")
    if not os.path.exists(xml_path):
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    
    for obj in root.findall("object"):
        cls = obj.find("name").text
        bb = obj.find("bndbox")
        x1 = int(bb.find("xmin").text)
        y1 = int(bb.find("ymin").text)
        x2 = int(bb.find("xmax").text)
        y2 = int(bb.find("ymax").text)
        boxes.append([
            int(round(x1 * state.display_scale)),
            int(round(y1 * state.display_scale)),
            int(round(x2 * state.display_scale)),
            int(round(y2 * state.display_scale)),
            cls
        ])
    
    return boxes