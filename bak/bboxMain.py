#!/usr/bin/env python3
"""
Annotator + YOLO label saver + background quick-train (1 epoch)
Requirements:
  - python3, opencv-python, numpy
  - ultralytics installed in same env (pip install ultralytics)
"""
import cv2, os, xml.etree.ElementTree as ET, shutil, numpy as np
from xml.dom import minidom
import threading
import copy
import torch, gc

colorsPalette = [
    (255,   0,   0),    # Neon Red
    (0,   255,   0),    # Neon Green
    (0,     0, 255),    # Neon Blue
    (255, 255,   0),    # Neon Yellow
    (255,   0, 255),    # Neon Magenta
    (0,   255, 255),    # Neon Cyan
    (255, 128,   0),    # Bright Orange
    (128,   0, 255),    # Vivid Purple
    (0,   128, 255),    # Neon Sky Blue
    (255,   0, 128),    # Hot Pink
    (0,   255, 128),    # Mint Green
    (255,  50,   0),    # Strong Red-Orange
    (255, 255, 255),    # White

    # Tambahan warna baru
    (128, 255,   0),    # Lime Yellow
    (0,   255, 180),    # Aqua Green
    (255,   0, 180),    # Electric Pink
    (180,   0, 255),    # Electric Purple
    (0,   180, 255),    # Bright Sky Blue
    (255, 180,   0),    # Vivid Amber
    (255, 60,  60),     # Bright Soft Red
    (60, 255,  60),     # Soft Neon Green
    (60,  60, 255),     # Soft Neon Blue
    (255, 100, 200),    # Neon Peach Pink
    (100, 255, 200),    # Light Mint Aqua
    (200, 100, 255),    # Soft Purple Neon
    (255, 200, 100),    # Warm Neon Orange
    (200, 255, 100),    # Neon Lime Soft
    (100, 200, 255),    # Neon Sky Soft
    (255, 100, 100),    # Light Coral Neon
    (255, 100, 50),    # Light Coral Neon
    (100, 255, 100),    # Pastel Neon Green
]


# ======== TRAINING LIB ========
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work until installed.", e)

# ======== CONFIG ========
workspaceName = "ppeKujangv2"
input_folder = "datasetsInput/ppeKujangv2-13"
output_folder = f"output/{workspaceName}"                # Pascal VOC
inference_root = f"inference/{workspaceName}"            # YOLO data (images + labels)
inference_images = os.path.join(inference_root, "images")
inference_labels = os.path.join(inference_root, "labels")
model_folder = f"models/{workspaceName}"
model_path = os.path.join(model_folder, "modelAssistant.pt")

for d in [output_folder, inference_images, inference_labels, model_folder]:
    os.makedirs(d, exist_ok=True)

# CLASSLIST = ["Boots", "Front", "Gloves", "Mask", "No-Boots", "No-Gloves", "Safety-Helmet", "Safety-Vest", "Safety-Wearpack", "Side", "Worker", "Shoes", "Glasses"]
CLASSLIST = [
    "Face",
    "Shoes",
    "Safety-Helmet",
    "Body",
    "Worker"
]
# CLASSLIST = [
#     "bus",
#     "car",
#     "motorbike",
#     "truck"
# ]
# CLASSLIST = ["person"]
# CLASSLIST = ["patokPersegi","patokPersegiIR", "patokPersegiPanjang", "patokPersegiPanjangIR"]
current_class = CLASSLIST[0]
VISIBLE_CLASS = {cls: True for cls in CLASSLIST}

# ======== GLOBALS ========
images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
images.sort()
if not images:
    raise SystemExit(f"No images found in {input_folder}")

current_index = 0
bboxes = []
prev_bboxes = []
selected_bbox = None
drawing = False
moving = False
resizing = False
force_new_bbox = False
ix, iy = -1, -1
display_scale = 1.0
frame = None
orig_shape = None
scroll_offset = 0
CLASS_HEIGHT = 35
CLASS_WINDOW_W = 220
CLASS_WINDOW_H = 360

training_running = False

automatedAnnotation = False

# ======== UTIL ========
def prettify_xml(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="   ")

def save_pascal_voc(img_name, img_shape):
    xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0]+".xml")
    ann = ET.Element("annotation")
    ET.SubElement(ann,"folder").text = workspaceName
    ET.SubElement(ann,"filename").text = img_name
    size = ET.SubElement(ann,"size")
    ET.SubElement(size,"width").text = str(img_shape[1])
    ET.SubElement(size,"height").text = str(img_shape[0])
    ET.SubElement(size,"depth").text = str(img_shape[2] if len(img_shape)>2 else 3)
    for bbox in bboxes:
        # Perbaikan: Gunakan round() untuk meminimalkan error presisi saat konversi dari tampilan ke asli
        x1 = int(round(bbox[0]/display_scale))
        y1 = int(round(bbox[1]/display_scale))
        x2 = int(round(bbox[2]/display_scale))
        y2 = int(round(bbox[3]/display_scale))
        cls = bbox[4]
        obj = ET.SubElement(ann,"object")
        ET.SubElement(obj,"name").text = cls
        bnd = ET.SubElement(obj,"bndbox")
        ET.SubElement(bnd,"xmin").text = str(max(0,x1))
        ET.SubElement(bnd,"ymin").text = str(max(0,y1))
        ET.SubElement(bnd,"xmax").text = str(max(0,x2))
        ET.SubElement(bnd,"ymax").text = str(max(0,y2))
    with open(xml_path,"w") as f: f.write(prettify_xml(ann))
    print(f"[INFO] Saved VOC: {xml_path}")

def save_yolo_label_and_image(img_name, orig_img):
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(inference_labels, base+".txt")
    dest_img = os.path.join(inference_images,img_name)
    h,w = orig_img.shape[:2]
    lines=[]
    for bbox in bboxes:
        # Menggunakan round() untuk konversi yang lebih presisi
        x1=int(round(bbox[0]/display_scale)); y1=int(round(bbox[1]/display_scale))
        x2=int(round(bbox[2]/display_scale)); y2=int(round(bbox[3]/display_scale))
        cls = bbox[4]
        if cls not in CLASSLIST: continue
        idx=CLASSLIST.index(cls)
        bw=(x2-x1)/w; bh=(y2-y1)/h
        cx=(x1+x2)/2/w; cy=(y1+y2)/2/h
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path,"w") as f: f.write("\n".join(lines))
    shutil.copy2(os.path.join(input_folder,img_name),dest_img)
    print(f"[INFO] Saved YOLO label: {label_path}")

def draw_all(frame_draw):
    for i,(x1,y1,x2,y2,cls) in enumerate(bboxes):
        if not VISIBLE_CLASS.get(cls, True):
            continue  # skip class yang disembunyikan
        classIndex = CLASSLIST.index(cls)
        color= colorsPalette[classIndex] if i!=selected_bbox else (60,60,200)
        cv2.rectangle(frame_draw,(x1,y1),(x2,y2),color,1)
        cv2.putText(frame_draw,cls,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    # indikator mode new box
    if force_new_bbox:
        cv2.rectangle(frame_draw,(5,5),(frame.shape[1]-5,frame.shape[0]-5),(200,60,60),2)

def load_annotation_local(img_name_local):
    xml_path=os.path.join(output_folder,os.path.splitext(img_name_local)[0]+".xml")
    if not os.path.exists(xml_path): return []
    tree=ET.parse(xml_path)
    root=tree.getroot()
    boxes=[]
    for obj in root.findall("object"):
        cls=obj.find("name").text
        bb=obj.find("bndbox")
        x1=int(bb.find("xmin").text)
        y1=int(bb.find("ymin").text)
        x2=int(bb.find("xmax").text)
        y2=int(bb.find("ymax").text)
        # Perbaikan: Gunakan round() untuk meminimalkan error presisi saat konversi dari asli ke tampilan
        boxes.append([int(round(x1*display_scale)),int(round(y1*display_scale)),int(round(x2*display_scale)),int(round(y2*display_scale)),cls])
    return boxes

# ======== MOUSE ========
def mouse_event(event, x, y, flags, param):
    global ix, iy, drawing, selected_bbox, moving, resizing, frame, force_new_bbox, bboxes, resize_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        selected_bbox = None
        resizing = False
        resize_mode = None

        # mode buat bbox baru
        if force_new_bbox:
            drawing = True
            return

        # cari semua bbox yang diklik → pilih yang paling kecil
        clicked = []
        for i, (x1, y1, x2, y2, cls) in enumerate(bboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                clicked.append((area, i))
        if clicked:
            _, selected_bbox = min(clicked, key=lambda a: a[0])
            x1, y1, x2, y2, _ = bboxes[selected_bbox]

            # === cek klik di area corner handle ===
            handle_size = 10
            if abs(x - x1) < handle_size and abs(y - y1) < handle_size:
                resizing = True; resize_mode = 'tl'  # top-left
            elif abs(x - x2) < handle_size and abs(y - y1) < handle_size:
                resizing = True; resize_mode = 'tr'  # top-right
            elif abs(x - x1) < handle_size and abs(y - y2) < handle_size:
                resizing = True; resize_mode = 'bl'  # bottom-left
            elif abs(x - x2) < handle_size and abs(y - y2) < handle_size:
                resizing = True; resize_mode = 'br'  # bottom-right
            else:
                moving = True  # klik di tengah → geser bbox

            return

        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = frame.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (255, 0, 0), 2)
            draw_all(temp)
            cv2.imshow("Annotator", temp)

        elif moving and selected_bbox is not None:
            dx, dy = x - ix, y - iy
            bboxes[selected_bbox][0] += dx
            bboxes[selected_bbox][1] += dy
            bboxes[selected_bbox][2] += dx
            bboxes[selected_bbox][3] += dy
            ix, iy = x, y

        elif resizing and selected_bbox is not None:
            x1, y1, x2, y2, cls = bboxes[selected_bbox]
            if resize_mode == 'tl':
                bboxes[selected_bbox][0] = min(x, x2 - 5)
                bboxes[selected_bbox][1] = min(y, y2 - 5)
            elif resize_mode == 'tr':
                bboxes[selected_bbox][2] = max(x, x1 + 5)
                bboxes[selected_bbox][1] = min(y, y2 - 5)
            elif resize_mode == 'bl':
                bboxes[selected_bbox][0] = min(x, x2 - 5)
                bboxes[selected_bbox][3] = max(y, y1 + 5)
            elif resize_mode == 'br':
                bboxes[selected_bbox][2] = max(x, x1 + 5)
                bboxes[selected_bbox][3] = max(y, y1 + 5)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            x1, y1, x2, y2 = sorted([ix, x])[0], sorted([iy, y])[0], sorted([ix, x])[1], sorted([iy, y])[1]
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w >= 8 and h >= 8:
                bboxes.append([x1, y1, x2, y2, current_class])
            else:
                print("[INFO] Skipped tiny bbox (<8px).")
        drawing = moving = resizing = False
        force_new_bbox = False


# ======== CLASS SELECTOR ========
def class_mouse_event(event, x, y, flags, param):
    global current_class, scroll_offset

    MAX_PER_COL = 10
    col_width = 220

    # jumlah kolom yg ditampung di UI
    total_cols = (len(CLASSLIST) + MAX_PER_COL - 1) // MAX_PER_COL

    if event == cv2.EVENT_LBUTTONDOWN:
        # jika klik di area bawah info, ignore
        if y >= CLASS_WINDOW_H:
            return

        # column index (clamp agar tidak keluar window)
        col = x // col_width
        if col < 0: col = 0
        if col >= total_cols:
            return  # klik di area kosong di kanan, ignore

        # baris relatif di jendela dan baris awal (scroll)
        start_row = scroll_offset // CLASS_HEIGHT
        row_local = y // CLASS_HEIGHT  # 0..(MAX_PER_COL-1) normally

        # validasi row_local
        if row_local < 0 or row_local >= MAX_PER_COL:
            return

        absolute_row = start_row + row_local

        # absolute_row harus dalam 0..MAX_PER_COL-1
        if absolute_row < 0 or absolute_row >= MAX_PER_COL:
            return

        idx = col * MAX_PER_COL + absolute_row

        if 0 <= idx < len(CLASSLIST):

            # posisi x relatif terhadap kolom untuk deteksi icon mata
            local_x = x - col * col_width

            # klik icon mata (local_x < 30)
            if local_x < 30:
                cls = CLASSLIST[idx]
                VISIBLE_CLASS[cls] = not VISIBLE_CLASS[cls]
                print(f"[INFO] Toggle visibility {cls}: {VISIBLE_CLASS[cls]}")
                return

            # klik nama class → pilih class
            current_class = CLASSLIST[idx]
            print(f"[INFO] Selected class: {current_class}")

    elif event == cv2.EVENT_MOUSEWHEEL:
        try:
            steps = int(flags / 120)
        except:
            steps = 0

        # scroll hanya berpatokan pada tinggi satu kolom (MAX_PER_COL)
        max_off = max(0, MAX_PER_COL * CLASS_HEIGHT - CLASS_WINDOW_H)
        scroll_offset -= steps * CLASS_HEIGHT
        scroll_offset = max(0, min(scroll_offset, max_off))


def draw_class_window():
    global scroll_offset

    # ============ CONFIG ============
    MAX_PER_COL = 10                       # <= 10 item per column
    col_width = 220
    rows_per_col = MAX_PER_COL

    total_classes = len(CLASSLIST)
    total_cols = (total_classes + rows_per_col - 1) // rows_per_col

    canvas_w = total_cols * col_width
    canvas_h = CLASS_WINDOW_H + 60

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    class_counts = {cls: 0 for cls in CLASSLIST}
    for _, _, _, _, cls in bboxes:
        if cls in class_counts:
            class_counts[cls] += 1

    # compute visible row range
    start_row = scroll_offset // CLASS_HEIGHT
    end_row = min(start_row + CLASS_WINDOW_H // CLASS_HEIGHT + 1, rows_per_col)

    # Draw each column
    for col in range(total_cols):
        x0 = col * col_width

        for row in range(start_row, end_row):
            idx = col * rows_per_col + row
            if idx >= total_classes:
                continue

            cls = CLASSLIST[idx]
            is_selected = (cls == current_class)
            is_visible = VISIBLE_CLASS.get(cls, True)

            y_pos = (row - start_row) * CLASS_HEIGHT

            # background
            bg_color = (0, 0, 255) if is_selected else (60, 60, 60)
            cv2.rectangle(canvas, (x0, y_pos), (x0 + col_width, y_pos + CLASS_HEIGHT - 2), bg_color, -1)

            # ===== eye icon =====
            cx = x0 + 18
            cy = y_pos + 18
            eye_w, eye_h = 16, 8
            eye_thickness = 2

            if is_visible:
                cv2.circle(canvas, (cx, cy), 3, (200, 80, 80), -1)
                cv2.ellipse(canvas, (cx, cy), (eye_w // 2, eye_h), 0, 0, 360, (60, 200, 60), eye_thickness)
            else:
                cv2.ellipse(canvas, (cx, cy), (eye_w // 2, eye_h), 0, 0, 360, (60, 60, 200), eye_thickness)
                cv2.line(canvas, (cx - eye_w // 2, cy - eye_h // 2),
                                  (cx + eye_w // 2, cy + eye_h // 2),
                                  (255, 255, 255), 2)
                cv2.circle(canvas, (cx, cy), 3, (80, 80, 200), -1)

            # text
            text = f"{cls} ({class_counts[cls]})"
            cv2.putText(canvas, text, (x0 + 45, y_pos + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ===== bottom info =====
    total_images = len(images)
    info_text = f"Image {current_index + 1}/{total_images}"

    cv2.rectangle(canvas, (0, CLASS_WINDOW_H), (canvas_w, CLASS_WINDOW_H + 60), (40, 40, 40), -1)
    cv2.putText(canvas, info_text, (10, CLASS_WINDOW_H + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("ClassSelector", canvas)

# ======== TRAIN ========
def split_train_val(root, ratio=0.7):
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")

    imgs = [f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(imgs) < 2:
        print("[WARN] Not enough images to split train/val.")
        return None, None

    np.random.shuffle(imgs)

    train_count = int(len(imgs) * ratio)
    train_imgs = imgs[:train_count]
    val_imgs   = imgs[train_count:]

    # Create dirs
    train_img_dir = os.path.join(root, "train/images")
    train_lbl_dir = os.path.join(root, "train/labels")
    val_img_dir   = os.path.join(root, "val/images")
    val_lbl_dir   = os.path.join(root, "val/labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy files
    for img in train_imgs:
        base = os.path.splitext(img)[0]
        txt = base + ".txt"

        shutil.copy2(os.path.join(images_dir, img), os.path.join(train_img_dir, img))
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(train_lbl_dir, txt))

    for img in val_imgs:
        base = os.path.splitext(img)[0]
        txt = base + ".txt"

        shutil.copy2(os.path.join(images_dir, img), os.path.join(val_img_dir, img))
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(val_lbl_dir, txt))

    return train_img_dir, val_img_dir

def train_model():
    global training_running

    if training_running:
        print("[INFO] Training already running.")
        return
    training_running = True

    # Check image count
    images_infer = [f for f in os.listdir(inference_images)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(images_infer) < 10:
        print("[INFO] Not enough images to train (min 10 required).")
        training_running = False
        return

    # ============= SPLIT 70/30 =============
    print("[INFO] Splitting dataset....")
    train_dir, val_dir = split_train_val(inference_root, ratio=0.5)

    if train_dir is None or val_dir is None:
        print("[ERROR] Dataset split failed.")
        training_running = False
        return

    # Absolute paths (penting!)
    train_abs = os.path.abspath(train_dir)
    val_abs   = os.path.abspath(val_dir)

    # ============= WRITE YAML =============
    yaml_path = os.path.join(inference_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_abs}\n")
        f.write(f"val: {val_abs}\n")
        f.write(f"nc: {len(CLASSLIST)}\n")
        f.write("names: [" + ", ".join([f"'{n}'" for n in CLASSLIST]) + "]\n")

    print("[INFO] YAML created:", yaml_path)
    print(open(yaml_path).read())

    # ============= INITIALIZE MODEL =============
    init_model = model_path if os.path.exists(model_path) else "yolo11s.pt"

    if os.path.exists(model_path):
        print(f"[INFO] Found existing model {model_path}")
    else:
        print("[INFO] Using pretrained yolo11s.pt")

    try:
        model = YOLO(init_model)
        print("[INFO] Starting training (epochs)...")

        model.train(
            data=yaml_path,
            epochs=5,
            imgsz=640,
            batch=4,

            optimizer="SGD",
            lr0=0.001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=1,

            project=model_folder,
            name="train_run",
            exist_ok=True
        )

        model.save(model_path)
        print(f"[INFO] Training finished. Model saved to {model_path}")

    except Exception as e:
        print("[ERROR] Training failed:", e)

    # ============= CLEANUP: DELETE train/val folders =============
    try:
        print("[INFO] Cleaning temporary train/val folders...")

        shutil.rmtree(os.path.join(inference_root, "train"))
        shutil.rmtree(os.path.join(inference_root, "val"))

        print("[INFO] train/ & val/ removed successfully.")

    except Exception as e:
        print("[WARN] Failed to delete train/val folders:", e)

    training_running = False
    del model
    torch.cuda.empty_cache()
    gc.collect()

# ======== REPEAT ========
def repeat_last_annotations():
    global bboxes, prev_bboxes, images, current_index
    if not prev_bboxes:
        print("[INFO] No previous annotations to repeat.")
        return
    print("[INFO] Reapplying previous annotations...")
    bboxes = [b.copy() for b in prev_bboxes]
    save_pascal_voc(images[current_index], frame.shape)
    save_yolo_label_and_image(images[current_index], frame)

# ======== INFERENCE ========
def inference_current(conf=0.5):
    global bboxes, frame, display_scale, images, current_index, input_folder
    if not os.path.exists(model_path):
        print("[INFO] Model assistant does not exist.")
        return
    
    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)
    
    model = YOLO(model_path)

    with torch.no_grad():                      # <-- penting!
        results = model.predict(orig_img, conf=conf, iou=0.4)

    # --- processing boxes ---
    pred_boxes = []
    for r in results:
        if hasattr(r, "boxes"):
            for box in r.boxes:
                x1 = int(box.xyxy[0,0].item())
                y1 = int(box.xyxy[0,1].item())
                x2 = int(box.xyxy[0,2].item())
                y2 = int(box.xyxy[0,3].item())
                cls_idx = int(box.cls[0].item())
                cls_name = CLASSLIST[cls_idx] if cls_idx < len(CLASSLIST) else str(cls_idx)

                # Saat membuat bbox baru dari inferensi, skalakan ke koordinat tampilan
                pred_boxes.append([
                    int(round(x1*display_scale)),
                    int(round(y1*display_scale)),
                    int(round(x2*display_scale)),
                    int(round(y2*display_scale)),
                    cls_name
                ])

    bboxes = pred_boxes
    print("[INFO] Inference saved and bboxes updated for display.")
    
    # --- CUDA cleanup ---
    del results
    torch.cuda.empty_cache()
    gc.collect()

# ======== MAIN LOOP ========
cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotator", 1420, 800)
cv2.setWindowProperty("Annotator", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("ClassSelector",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator",mouse_event)
cv2.setMouseCallback("ClassSelector",class_mouse_event)

autoAnnotation = False
while True:
    img_name=images[current_index]
    orig=cv2.imread(os.path.join(input_folder,img_name))
    if orig is None:
        print(f"[INFO] Skip {img_name}")
        current_index=(current_index+1)%len(images)
        continue
    orig_shape=orig.shape
    h,w=orig_shape[:2]
    scale=720/h if h>720 else 1.0
    display_scale=scale
    frame=cv2.resize(orig,(int(w*scale),int(h*scale)))
    h_disp, w_disp = frame.shape[:2]
    cv2.resizeWindow("Annotator", w_disp, h_disp)
    bboxes=load_annotation_local(img_name)

    if autoAnnotation:
        save_pascal_voc(img_name,orig_shape)
        save_yolo_label_and_image(img_name,orig)
        prev_bboxes = copy.deepcopy(bboxes)
        inference_current(conf=0.2)
        print("[INFO] Auto annotation active.")
        autoAnnotation = False

    while True:
        disp=frame.copy()
        if automatedAnnotation:
            cv2.putText(disp,"AI Assistant Enabled",(frame.shape[1],frame.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(60,200,60),1)
        draw_all(disp)
        draw_class_window()
        cv2.imshow("Annotator",disp)
        key=cv2.waitKey(30)&0xFF

        if key==ord('d'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            prev_bboxes = copy.deepcopy(bboxes)
            current_index=(current_index+1)%len(images)
            if automatedAnnotation:
                autoAnnotation = True
            break
        elif key==ord('a'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            prev_bboxes = copy.deepcopy(bboxes)
            current_index=(current_index-1)%len(images)
            if automatedAnnotation:
                autoAnnotation = True
            break
        elif key==ord('r') and selected_bbox is not None:
            del bboxes[selected_bbox]; selected_bbox=None
        elif key in [ord('s')] and selected_bbox is not None:
            idx=CLASSLIST.index(bboxes[selected_bbox][4])
            idx=(idx+1)%len(CLASSLIST)
            bboxes[selected_bbox][4]=CLASSLIST[idx]
        elif key==ord('t'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            if not training_running:
                print("[INFO] Starting background training thread...")
                threading.Thread(target=train_model,daemon=True).start()
            else:
                print("[INFO] Training already running. Please wait...")
        elif key==ord('g'):
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            prev_bboxes = copy.deepcopy(bboxes)
            inference_current(conf=0.3)
        elif key==ord('e'):
            repeat_last_annotations()
        elif key==ord('b'):
            force_new_bbox=True
            print("[INFO] Force new bbox mode ON (next click = new box).")
        elif key==ord('p'):
            if automatedAnnotation == False:
                print("[INFO] Enable auto annotation.")
                automatedAnnotation = True
            else:
                print("[INFO] Disable auto annotation.")
                automatedAnnotation = False
        elif key in [27, ord('q')]:
            save_pascal_voc(img_name,orig_shape)
            save_yolo_label_and_image(img_name,orig)
            print("[INFO] Exiting and saved current annotations.")
            cv2.destroyAllWindows()
            exit(0)
        elif key in range(49,49+len(CLASSLIST)):
            idx=key-49
            current_class=CLASSLIST[idx]
            print(f"[INFO] Selected class via key: {current_class}")
        elif key == 8:  # Backspace = hapus gambar + annotasi
            if not images:
                print("[WARN] No images left.")
                continue

            img_name = images[current_index]
            base_name = os.path.splitext(img_name)[0]

            # Path semua file terkait
            input_path = os.path.join(input_folder, img_name)
            xml_path = os.path.join(output_folder, base_name + ".xml")
            infer_img_path = os.path.join(inference_images, img_name)
            infer_label_path = os.path.join(inference_labels, base_name + ".txt")

            # Hapus file-file jika ada
            for f in [input_path, xml_path, infer_img_path, infer_label_path]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[INFO] Deleted: {f}")

            # Hapus dari list images
            del images[current_index]

            # Bersihkan bbox
            bboxes.clear()

            # Tentukan index baru
            if current_index >= len(images):
                current_index = max(0, len(images) - 1)

            # Kalau udah gak ada gambar, tampilin notif
            if not images:
                print("[INFO] All images deleted.")
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "No images left", (400, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Annotator", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit(0)

            # Load gambar berikutnya
            img_name = images[current_index]
            orig = cv2.imread(os.path.join(input_folder, img_name))
            if orig is None:
                print(f"[WARN] Failed to load next image: {img_name}")
                continue

            h, w = orig.shape[:2]
            scale = 720 / h if h > 720 else 1.0
            display_scale = scale
            frame = cv2.resize(orig, (int(w * scale), int(h * scale)))
            bboxes = load_annotation_local(img_name)

            print(f"[INFO] Moved to next image: {img_name}")