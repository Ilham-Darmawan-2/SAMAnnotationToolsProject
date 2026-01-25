import os
import cv2
import xml.etree.ElementTree as ET

# ================== CONFIG ==================
IMAGE_DIR = "datasetsInput/markerv2-7"
VOC_DIR   = "output/markerv2"
YOLO_DIR  = "inference/markerv2/labels"
WINDOW_NAME = "Annotation Viewer"

CLASS_NAMES = ['patokPersegi', 'patokPersegiIR', 'patokPersegiPanjang', 'patokPersegiPanjangIR']

ADJUST_CLASSES = ["patokPersegiPanjangIR", "patokPersegiIR"]
SHRINK_RATIO = 0.03   # 5% (bisa ganti 0.03, 0.1, dll)

DISP_W, DISP_H = 1280, 720
# ============================================

def load_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")

        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        boxes.append((label, xmin, ymin, xmax, ymax))

    return boxes


def load_yolo(txt_path, img_w, img_h):
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])

            bw = w * img_w
            bh = h * img_h
            x1 = int((xc * img_w) - bw / 2)
            y1 = int((yc * img_h) - bh / 2)
            x2 = int((xc * img_w) + bw / 2)
            y2 = int((yc * img_h) + bh / 2)

            if CLASS_NAMES and cls_id < len(CLASS_NAMES):
                label = CLASS_NAMES[cls_id]
            else:
                label = str(cls_id)

            boxes.append((label, x1, y1, x2, y2))

    return boxes


def draw_boxes(image, boxes, color, adjust_classes=None, shrink_ratio=0.0):
    H, W = image.shape[:2]

    for label, xmin, ymin, xmax, ymax in boxes:
        if adjust_classes and label in adjust_classes:
            h = ymax - ymin
            offset = int(h * shrink_ratio)
            ymin = ymin + offset
            ymax = ymax - offset

            ymin = max(0, ymin)
            ymax = min(H - 1, ymax)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            label,
            (xmin, max(0, ymin - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
    return image


def main():
    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not images:
        print("❌ Tidak ada gambar")
        return

    idx = 0

    while True:
        idx = max(0, min(idx, len(images) - 1))

        img_name = images[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        xml_path = os.path.join(VOC_DIR, base + ".xml")
        txt_path = os.path.join(YOLO_DIR, base + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Gagal load image: {img_name}")
            images.pop(idx)
            continue

        H, W = image.shape[:2]

        if os.path.exists(xml_path):
            boxes_voc = load_voc(xml_path)
            print(f"[VOC] {img_name}")
            image = draw_boxes(
                image, boxes_voc, (0, 255, 0),
                adjust_classes=ADJUST_CLASSES,
                shrink_ratio=SHRINK_RATIO
            )

        if os.path.exists(txt_path):
            boxes_yolo = load_yolo(txt_path, W, H)
            print(f"[YOLO] {img_name}")
            image = draw_boxes(
                image, boxes_yolo, (0, 0, 255),
                adjust_classes=ADJUST_CLASSES,
                shrink_ratio=SHRINK_RATIO
            )

        image_disp = cv2.resize(image, (DISP_W, DISP_H))
        cv2.imshow(WINDOW_NAME, image_disp)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("d"):
            idx = (idx + 1) % len(images)

        elif key == ord("a"):
            idx = (idx - 1) % len(images)

        elif key == 8:  # BACKSPACE
            print(f"🗑️ Menghapus: {img_name}")

            try:
                os.remove(img_path)
                print("  ✔ image deleted")
            except Exception as e:
                print(f"  ❌ gagal hapus image: {e}")

            if os.path.exists(xml_path):
                os.remove(xml_path)
                print("  ✔ xml deleted")

            if os.path.exists(txt_path):
                os.remove(txt_path)
                print("  ✔ txt deleted")

            images.pop(idx)
            if not images:
                print("✅ Semua gambar sudah dihapus")
                break

            idx = min(idx, len(images) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
