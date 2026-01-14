import os
import cv2
import xml.etree.ElementTree as ET

# ================== CONFIG ==================
IMAGE_DIR = "datasetsInput/example-1"
ANNOT_DIR = "output/example"
WINDOW_NAME = "Pascal VOC Viewer"
# ============================================


def load_annotation(xml_path):
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


def draw_boxes(image, boxes):
    for label, xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
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
        if idx >= len(images):
            idx = max(len(images) - 1, 0)

        img_name = images[idx]
        img_path = os.path.join(IMAGE_DIR, img_name)
        xml_path = os.path.join(
            ANNOT_DIR,
            os.path.splitext(img_name)[0] + ".xml"
        )

        image = cv2.imread(img_path)
        if image is None:
            print(f"Gagal load image: {img_name}")
            images.pop(idx)
            continue

        if os.path.exists(xml_path):
            boxes = load_annotation(xml_path)
            image = draw_boxes(image, boxes)

        image = cv2.resize(image, (1280,720))
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(0) & 0xFF

        # ========= KEY CONTROL =========
        if key == ord("q"):
            break

        elif key == ord("d"):  # next
            idx = (idx + 1) % len(images)

        elif key == ord("a"):  # prev
            idx = (idx - 1) % len(images)

        elif key == 8:  # BACKSPACE
            print(f"🗑️ Menghapus: {img_name}")

            try:
                os.remove(img_path)
                print(f"  ✔ image deleted")
            except Exception as e:
                print(f"  ❌ gagal hapus image: {e}")

            if os.path.exists(xml_path):
                try:
                    os.remove(xml_path)
                    print(f"  ✔ annotation deleted")
                except Exception as e:
                    print(f"  ❌ gagal hapus xml: {e}")

            images.pop(idx)

            if not images:
                print("✅ Semua gambar sudah dihapus")
                break

            idx = min(idx, len(images) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
