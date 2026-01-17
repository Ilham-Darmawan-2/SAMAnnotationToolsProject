import os
import cv2
import xml.etree.ElementTree as ET

IMG_DIR = "datasetsInput/vehicle-1"
ANN_DIR = "inference/vehicle/labels"
MARGIN = 24

preview_shown = False

def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))

for ann_file in os.listdir(ANN_DIR):
    ext = os.path.splitext(ann_file)[1].lower()

    ann_path = os.path.join(ANN_DIR, ann_file)

    # cari image pair
    base = os.path.splitext(ann_file)[0]
    img_path = None
    for e in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(IMG_DIR, base + e)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        continue

    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    preview = img.copy()
    to_draw = []

    # ================= XML (Pascal VOC) =================
    if ext == ".xml":
        tree = ET.parse(ann_path)
        root = tree.getroot()
        updated = False

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            new_xmin = clamp(xmin, 0, W - 1 - MARGIN)
            new_ymin = clamp(ymin, 0, H - 1 - MARGIN)
            new_xmax = clamp(xmax, MARGIN, W - 1)
            new_ymax = clamp(ymax, MARGIN, H - 1)

            if new_xmax <= new_xmin:
                new_xmax = min(W - 1, new_xmin + MARGIN)
            if new_ymax <= new_ymin:
                new_ymax = min(H - 1, new_ymin + MARGIN)

            if (xmin, ymin, xmax, ymax) != (new_xmin, new_ymin, new_xmax, new_ymax):
                if not preview_shown:
                    cv2.rectangle(preview, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    cv2.rectangle(preview, (new_xmin, new_ymin), (new_xmax, new_ymax), (0,255,0), 2)

                bndbox.find("xmin").text = str(new_xmin)
                bndbox.find("ymin").text = str(new_ymin)
                bndbox.find("xmax").text = str(new_xmax)
                bndbox.find("ymax").text = str(new_ymax)
                updated = True

        if updated:
            if not preview_shown:
                cv2.imshow("VOC Clamp Preview", preview)
                print("Preview VOC. Tekan ENTER...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                preview_shown = True

            tree.write(ann_path)
            print(f"Clamped VOC: {ann_file}")

    # ================= YOLO TXT =================
    elif ext == ".txt":
        with open(ann_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        updated = False

        for line in lines:
            parts = line.strip().split()
            cls = parts[0]
            xc, yc, w, h = map(float, parts[1:])

            bw = w * W
            bh = h * H
            x1 = (xc * W) - bw / 2
            y1 = (yc * H) - bh / 2
            x2 = (xc * W) + bw / 2
            y2 = (yc * H) + bh / 2

            new_x1 = clamp(x1, 0, W - 1 - MARGIN)
            new_y1 = clamp(y1, 0, H - 1 - MARGIN)
            new_x2 = clamp(x2, MARGIN, W - 1)
            new_y2 = clamp(y2, MARGIN, H - 1)

            if abs(x1-new_x1)>1 or abs(y1-new_y1)>1 or abs(x2-new_x2)>1 or abs(y2-new_y2)>1:
                updated = True
                cv2.rectangle(preview, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
                cv2.rectangle(preview, (int(new_x1),int(new_y1)), (int(new_x2),int(new_y2)), (0,255,0), 2)

            new_bw = new_x2 - new_x1
            new_bh = new_y2 - new_y1
            new_xc = (new_x1 + new_x2) / 2 / W
            new_yc = (new_y1 + new_y2) / 2 / H
            new_w = new_bw / W
            new_h = new_bh / H

            new_lines.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}\n")

        if updated:
            if not preview_shown:
                cv2.imshow("YOLO Clamp Preview", preview)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                preview_shown = True

            with open(ann_path, "w") as f:
                f.writelines(new_lines)

            print(f"Clamped YOLO: {ann_file}")

        else:
            print(f"OK (no clamp needed): {ann_file}")


print("Semua bbox (VOC & YOLO) sudah di-clamp.")
