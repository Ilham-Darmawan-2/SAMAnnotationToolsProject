#!/usr/bin/env python3
"""
POLYGON Annotator (Roboflow XML Format Support)
Capabilities:
  - Load Logic: Prioritas XML (Roboflow Format: x1, y1 tags), Fallback YOLO.
  - Save: XML persis format Roboflow & YOLO TXT.
  - Draw: Left Click add point, Click Start Point to Close.
"""
import cv2, os, xml.etree.ElementTree as ET, shutil, numpy as np
from xml.dom import minidom
import threading
import copy
import torch, gc
import math

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
    (128, 255,   0),    # Lime Yellow
    (0,   255, 180),    # Aqua Green
    (255,   0, 180),    # Electric Pink
    (180,   0, 255),    # Electric Purple
    (0,   180, 255),    # Bright Sky Blue
    (255, 180,   0),    # Vivid Amber
]

# ======== TRAINING LIB ========
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("[INFO] ultralytics not installed. Training won't work.", e)

# ======== CONFIG ========
workspaceName = "licensePlateSegmentation"
input_folder = "datasetsInput/licensePlateSegmentation-6"
output_folder = f"output/{workspaceName}"                # XML Output
inference_root = f"inference/{workspaceName}"            # YOLO data
inference_images = os.path.join(inference_root, "images")
inference_labels = os.path.join(inference_root, "labels")
model_folder = f"models/{workspaceName}"
model_path = os.path.join(model_folder, "modelAssistantSeg.pt")

for d in [output_folder, inference_images, inference_labels, model_folder]:
    os.makedirs(d, exist_ok=True)

CLASSLIST = ["licensePlate"]
current_class = CLASSLIST[0]
VISIBLE_CLASS = {cls: True for cls in CLASSLIST}

# ======== GLOBALS ========
images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
images.sort()
if not images:
    raise SystemExit(f"No images found in {input_folder}")

current_index = 0
polygons = []         
prev_polygons = []
selected_polygon = None

drawing = False
moving = False
force_new_poly = False
ix, iy = -1, -1
display_scale = 1.0
frame = None
orig_shape = None
scroll_offset = 0
CLASS_HEIGHT = 35
CLASS_WINDOW_W = 220
CLASS_WINDOW_H = 360

current_points = [] 
training_running = False

# ======== UTIL ========
def prettify_xml(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent="    ")

def save_pascal_voc(img_name, img_shape):
    """
    Menyimpan XML dengan struktur PERSIS seperti Roboflow
    (Menggunakan tag <x1>, <y1>, dst untuk polygon)
    """
    xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0]+".xml")
    ann = ET.Element("annotation")
    
    # Metadata standar Roboflow
    ET.SubElement(ann, "folder").text = "" # Biasanya kosong di Roboflow export
    ET.SubElement(ann, "filename").text = img_name
    ET.SubElement(ann, "path").text = img_name
    
    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = "roboflow.com"
    
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(img_shape[1])
    ET.SubElement(size, "height").text = str(img_shape[0])
    ET.SubElement(size, "depth").text = str(img_shape[2] if len(img_shape)>2 else 3)
    
    ET.SubElement(ann, "segmented").text = "0"

    for poly in polygons:
        cls = poly[-1]
        pts = poly[:-1] 
        
        # Konversi ke koordinat asli (Real Image Coordinates)
        real_pts = [[p[0]/display_scale, p[1]/display_scale] for p in pts]
        
        # Hitung Bounding Box (untuk compatibilitas tag <bndbox>)
        xs = [p[0] for p in real_pts]
        ys = [p[1] for p in real_pts]
        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = cls
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "occluded").text = "0"
        
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(max(0, xmin))
        ET.SubElement(bnd, "xmax").text = str(max(0, xmax))
        ET.SubElement(bnd, "ymin").text = str(max(0, ymin))
        ET.SubElement(bnd, "ymax").text = str(max(0, ymax))
        
        # === FORMAT POLYGON ROBOFLOW ===
        # <polygon><x1>val</x1><y1>val</y1>...</polygon>
        poly_tag = ET.SubElement(obj, "polygon")
        
        for i, (px, py) in enumerate(real_pts, start=1):
            # Roboflow support float values
            ET.SubElement(poly_tag, f"x{i}").text = f"{px:.3f}".rstrip('0').rstrip('.')
            ET.SubElement(poly_tag, f"y{i}").text = f"{py:.3f}".rstrip('0').rstrip('.')

    # Tulis file
    with open(xml_path, "w") as f: 
        # Hack sedikit supaya tidak terlalu banyak whitespace aneh
        xml_str = prettify_xml(ann)
        f.write(xml_str)
        
    print(f"[INFO] Saved Roboflow-style XML: {xml_path}")

def save_yolo_label_and_image(img_name, orig_img):
    base = os.path.splitext(img_name)[0]
    label_path = os.path.join(inference_labels, base+".txt")
    dest_img = os.path.join(inference_images,img_name)
    h,w = orig_img.shape[:2]
    lines=[]
    
    for poly in polygons:
        cls = poly[-1]
        pts = poly[:-1]
        if cls not in CLASSLIST: continue
        idx = CLASSLIST.index(cls)
        
        norm_pts = []
        for (px, py) in pts:
            rx = px / display_scale
            ry = py / display_scale
            nx = rx / w
            ny = ry / h
            norm_pts.append(f"{nx:.6f} {ny:.6f}")
            
        line_str = f"{idx} " + " ".join(norm_pts)
        lines.append(line_str)
        
    with open(label_path,"w") as f: f.write("\n".join(lines))
    shutil.copy2(os.path.join(input_folder,img_name),dest_img)
    print(f"[INFO] Saved YOLO Segmentation: {label_path}")

def draw_all(frame_draw):
    for i, poly in enumerate(polygons):
        cls = poly[-1]
        pts = poly[:-1]
        if not VISIBLE_CLASS.get(cls, True): continue

        classIndex = CLASSLIST.index(cls)
        base_color = colorsPalette[classIndex]
        color = (60, 60, 200) if i == selected_polygon else base_color
        
        np_pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_draw, [np_pts], True, color, 2)
        if len(pts) > 0:
            cv2.putText(frame_draw, cls, (pts[0][0], pts[0][1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if drawing and len(current_points) > 0:
        np_curr = np.array(current_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame_draw, [np_curr], False, (255, 255, 255), 1)
        
        sx, sy = current_points[0]
        cv2.circle(frame_draw, (sx, sy), 6, (0, 255, 0), -1) 
        cv2.circle(frame_draw, (sx, sy), 8, (255, 255, 255), 1)

        for p in current_points[1:]:
            cv2.circle(frame_draw, (p[0], p[1]), 3, (0, 255, 255), -1)
            
        if ix != -1 and iy != -1:
            cv2.line(frame_draw, tuple(current_points[-1]), (ix, iy), (200, 200, 200), 1)

    if force_new_poly:
        cv2.rectangle(frame_draw,(5,5),(frame.shape[1]-5,frame.shape[0]-5),(200,60,60),2)

def load_annotation_local(img_name_local):
    """
    PRIORITAS: XML (Roboflow Format: x1, y1...) -> YOLO TXT (Fallback)
    """
    base = os.path.splitext(img_name_local)[0]
    
    # 1. CEK XML DI FOLDER OUTPUT (Format Roboflow)
    xml_path = os.path.join(output_folder, base + ".xml")
    
    # Jika tidak ada di output folder, cek apakah ada xml bawaan di input folder (opsional)
    # xml_input_path = os.path.join(input_folder, base + ".xml")
    # if not os.path.exists(xml_path) and os.path.exists(xml_input_path):
    #     xml_path = xml_input_path

    if os.path.exists(xml_path):
        print(f"[INFO] Found XML: {xml_path}")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            loaded_polys = []
            
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in CLASSLIST:
                    print(f"[WARN] Unknown class '{cls_name}' in XML. Skipping.")
                    continue

                poly_tag = obj.find("polygon")
                
                if poly_tag is not None:
                    # === PARSE ROBOFLOW POLYGON (x1, y1, x2, y2...) ===
                    poly_pts = []
                    i = 1
                    while True:
                        # Cari tag x1, y1, x2, y2 dst...
                        x_node = poly_tag.find(f"x{i}")
                        y_node = poly_tag.find(f"y{i}")
                        
                        if x_node is None or y_node is None:
                            break # Berhenti jika tidak ada titik selanjutnya
                        
                        px = float(x_node.text)
                        py = float(y_node.text)
                        
                        # Scale ke Display
                        dx = int(px * display_scale)
                        dy = int(py * display_scale)
                        poly_pts.append([dx, dy])
                        
                        i += 1
                    
                    if len(poly_pts) > 2:
                        poly_pts.append(cls_name)
                        loaded_polys.append(poly_pts)
                
                else:
                    # Parse Bounding Box Legacy
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)
                        
                        pts_real = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                        poly_pts = []
                        for pr in pts_real:
                            dx = int(pr[0] * display_scale)
                            dy = int(pr[1] * display_scale)
                            poly_pts.append([dx, dy])
                        
                        poly_pts.append(cls_name)
                        loaded_polys.append(poly_pts)
            
            return loaded_polys
            
        except Exception as e:
            print(f"[ERROR] Failed loading Roboflow XML: {e}")
            return []

    # 2. FALLBACK KE YOLO TXT
    txt_path = os.path.join(inference_labels, base+".txt")
    if os.path.exists(txt_path):
        loaded_polys = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                cls_idx = int(parts[0])
                if cls_idx >= len(CLASSLIST): continue
                cls_name = CLASSLIST[cls_idx]
                coords = parts[1:]
                poly_pts = []
                for i in range(0, len(coords), 2):
                    nx = float(coords[i])
                    ny = float(coords[i+1])
                    x = int(nx * orig_shape[1] * display_scale)
                    y = int(ny * orig_shape[0] * display_scale)
                    poly_pts.append([x, y])
                poly_pts.append(cls_name) 
                loaded_polys.append(poly_pts)
        return loaded_polys

    return []

# ======== MOUSE ========
def mouse_event(event, x, y, flags, param):
    global ix, iy, drawing, selected_polygon, moving, frame, force_new_poly, polygons, current_points

    ix, iy = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            if len(current_points) > 0:
                sx, sy = current_points[0]
                dist = math.sqrt((x - sx)**2 + (y - sy)**2)
                
                if dist < 10: 
                    if len(current_points) >= 3:
                        new_poly = copy.deepcopy(current_points)
                        new_poly.append(current_class)
                        polygons.append(new_poly)
                        print(f"[INFO] Polygon finished ({len(current_points)} pts).")
                        drawing = False
                        current_points = []
                        force_new_poly = False
                        return
                    else:
                        print(f"[WARN] Need 3+ points.")
                        return 
            
            current_points.append([x, y])
            return

        selected_polygon = None
        moving = False
        
        if force_new_poly:
            drawing = True
            current_points = [[x, y]]
            print("[INFO] Started new polygon.")
            return

        clicked_candidates = []
        for i, poly in enumerate(polygons):
            pts = np.array(poly[:-1], np.int32)
            dist = cv2.pointPolygonTest(pts, (x, y), False) 
            if dist >= 0: 
                area = cv2.contourArea(pts)
                clicked_candidates.append((area, i))
        
        if clicked_candidates:
            _, selected_polygon = min(clicked_candidates, key=lambda x: x[0])
            moving = True
            ix, iy = x, y
            return

        drawing = True
        current_points = [[x, y]]
        print("[INFO] Started new polygon.")

    elif event == cv2.EVENT_MOUSEMOVE:
        if moving and selected_polygon is not None:
            dx, dy = x - ix, y - iy
            for i in range(len(polygons[selected_polygon]) - 1):
                polygons[selected_polygon][i][0] += dx
                polygons[selected_polygon][i][1] += dy
            ix, iy = x, y

    elif event == cv2.EVENT_RBUTTONDOWN:
        if drawing:
            print("[INFO] Drawing cancelled.")
            drawing = False
            current_points = []
            force_new_poly = False
        else:
             selected_polygon = None

# ======== CLASS SELECTOR ========
def class_mouse_event(event, x, y, flags, param):
    global current_class, scroll_offset
    MAX_PER_COL = 10
    col_width = 220
    total_cols = (len(CLASSLIST) + MAX_PER_COL - 1) // MAX_PER_COL

    if event == cv2.EVENT_LBUTTONDOWN:
        if y >= CLASS_WINDOW_H: return
        col = x // col_width
        if col < 0: col = 0
        if col >= total_cols: return
        start_row = scroll_offset // CLASS_HEIGHT
        row_local = y // CLASS_HEIGHT 
        if row_local < 0 or row_local >= MAX_PER_COL: return
        absolute_row = start_row + row_local
        if absolute_row < 0 or absolute_row >= MAX_PER_COL: return
        idx = col * MAX_PER_COL + absolute_row

        if 0 <= idx < len(CLASSLIST):
            local_x = x - col * col_width
            if local_x < 30:
                cls = CLASSLIST[idx]
                VISIBLE_CLASS[cls] = not VISIBLE_CLASS[cls]
                return
            current_class = CLASSLIST[idx]
            print(f"[INFO] Selected class: {current_class}")

    elif event == cv2.EVENT_MOUSEWHEEL:
        try: steps = int(flags / 120)
        except: steps = 0
        max_off = max(0, MAX_PER_COL * CLASS_HEIGHT - CLASS_WINDOW_H)
        scroll_offset -= steps * CLASS_HEIGHT
        scroll_offset = max(0, min(scroll_offset, max_off))

def draw_class_window():
    global scroll_offset
    MAX_PER_COL = 10
    col_width = 220
    rows_per_col = MAX_PER_COL
    total_classes = len(CLASSLIST)
    total_cols = (total_classes + rows_per_col - 1) // rows_per_col
    canvas_w = total_cols * col_width
    canvas_h = CLASS_WINDOW_H + 60
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    class_counts = {cls: 0 for cls in CLASSLIST}
    for poly in polygons:
        cls = poly[-1]
        if cls in class_counts: class_counts[cls] += 1

    start_row = scroll_offset // CLASS_HEIGHT
    end_row = min(start_row + CLASS_WINDOW_H // CLASS_HEIGHT + 1, rows_per_col)

    for col in range(total_cols):
        x0 = col * col_width
        for row in range(start_row, end_row):
            idx = col * rows_per_col + row
            if idx >= total_classes: continue
            cls = CLASSLIST[idx]
            is_selected = (cls == current_class)
            is_visible = VISIBLE_CLASS.get(cls, True)
            y_pos = (row - start_row) * CLASS_HEIGHT
            bg_color = (0, 0, 255) if is_selected else (60, 60, 60)
            cv2.rectangle(canvas, (x0, y_pos), (x0 + col_width, y_pos + CLASS_HEIGHT - 2), bg_color, -1)
            cx = x0 + 18; cy = y_pos + 18
            if is_visible:
                cv2.circle(canvas, (cx, cy), 3, (200, 80, 80), -1)
                cv2.ellipse(canvas, (cx, cy), (8, 4), 0, 0, 360, (60, 200, 60), 2)
            else:
                cv2.ellipse(canvas, (cx, cy), (8, 4), 0, 0, 360, (60, 60, 200), 2)
                cv2.line(canvas, (cx - 8, cy - 4), (cx + 8, cy + 4), (255, 255, 255), 2)
            text = f"{cls} ({class_counts[cls]})"
            cv2.putText(canvas, text, (x0 + 45, y_pos + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
    info_text = f"Image {current_index + 1}/{len(images)}"
    cv2.rectangle(canvas, (0, CLASS_WINDOW_H), (canvas_w, CLASS_WINDOW_H + 60), (40, 40, 40), -1)
    cv2.putText(canvas, info_text, (10, CLASS_WINDOW_H + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.imshow("ClassSelector", canvas)

# ======== TRAIN ========
def split_train_val(root, ratio=0.7):
    images_dir = os.path.join(root, "images")
    labels_dir = os.path.join(root, "labels")
    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(imgs) < 2: return None, None
    np.random.shuffle(imgs)
    train_count = int(len(imgs) * ratio)
    train_imgs = imgs[:train_count]; val_imgs = imgs[train_count:]
    
    train_img_dir = os.path.join(root, "train/images"); train_lbl_dir = os.path.join(root, "train/labels")
    val_img_dir = os.path.join(root, "val/images"); val_lbl_dir = os.path.join(root, "val/labels")
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]: os.makedirs(d, exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(images_dir, img), os.path.join(train_img_dir, img))
        txt = os.path.splitext(img)[0]+".txt"
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(train_lbl_dir, txt))
            
    for img in val_imgs:
        shutil.copy2(os.path.join(images_dir, img), os.path.join(val_img_dir, img))
        txt = os.path.splitext(img)[0]+".txt"
        if os.path.exists(os.path.join(labels_dir, txt)):
            shutil.copy2(os.path.join(labels_dir, txt), os.path.join(val_lbl_dir, txt))
    return train_img_dir, val_img_dir

def train_model():
    global training_running
    if training_running: return
    training_running = True

    images_infer = [f for f in os.listdir(inference_images) if f.lower().endswith(('.jpg', '.png'))]
    if len(images_infer) < 5:
        print("[INFO] Not enough images for training."); training_running = False; return

    print("[INFO] Splitting dataset for SEGMENTATION training....")
    train_dir, val_dir = split_train_val(inference_root, ratio=0.8)
    if not train_dir: training_running = False; return

    train_abs = os.path.abspath(train_dir)
    val_abs   = os.path.abspath(val_dir)

    yaml_path = os.path.join(inference_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_abs}\nval: {val_abs}\nnc: {len(CLASSLIST)}\nnames: {CLASSLIST}\n")

    init_model = model_path if os.path.exists(model_path) else "yolov8n-seg.pt" 
    print(f"[INFO] Using model: {init_model}")

    try:
        model = YOLO(init_model)
        model.train(data=yaml_path, epochs=40, imgsz=640, batch=16, project=model_folder, name="train_seg", exist_ok=True)
        model.save(model_path)
        print(f"[INFO] Segmentation Model Saved: {model_path}")
    except Exception as e:
        print("[ERROR]", e)

    try:
        shutil.rmtree(os.path.join(inference_root, "train"))
        shutil.rmtree(os.path.join(inference_root, "val"))
    except: pass

    training_running = False
    del model; torch.cuda.empty_cache(); gc.collect()

def repeat_last_annotations():
    global polygons, prev_polygons, images, current_index
    if not prev_polygons: return
    polygons = copy.deepcopy(prev_polygons)
    save_pascal_voc(images[current_index], frame.shape)
    save_yolo_label_and_image(images[current_index], frame)

# ======== INFERENCE ========
def inference_current(conf=0.5):
    global polygons, frame, display_scale, images, current_index
    if not os.path.exists(model_path): print("[INFO] No model found."); return
    img_path = os.path.join(input_folder, images[current_index])
    orig_img = cv2.imread(img_path)
    h, w = orig_img.shape[:2]
    model = YOLO(model_path)
    with torch.no_grad():
        results = model.predict(orig_img, conf=conf, iou=0.4, retina_masks=True)

    new_polys = []
    for r in results:
        if r.masks is None: continue
        masks_xyn = r.masks.xyn 
        clss = r.boxes.cls.cpu().numpy()
        for i, poly_norm in enumerate(masks_xyn):
            cls_idx = int(clss[i])
            cls_name = CLASSLIST[cls_idx] if cls_idx < len(CLASSLIST) else str(cls_idx)
            poly_display = []
            for p in poly_norm:
                dx = int(p[0] * w * display_scale)
                dy = int(p[1] * h * display_scale)
                poly_display.append([dx, dy])
            if len(poly_display) > 2:
                poly_display.append(cls_name)
                new_polys.append(poly_display)
    polygons = new_polys
    print("[INFO] Assistant updated polygons.")
    del results; torch.cuda.empty_cache(); gc.collect()

# ======== MAIN ========
cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotator", 1420, 800)
cv2.setMouseCallback("Annotator", mouse_event)
cv2.namedWindow("ClassSelector", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ClassSelector", class_mouse_event)

while True:
    img_name = images[current_index]
    orig = cv2.imread(os.path.join(input_folder, img_name))
    if orig is None: current_index = (current_index + 1) % len(images); continue
    orig_shape = orig.shape
    h, w = orig_shape[:2]
    scale = 720/h if h > 720 else 1.0
    display_scale = scale
    frame = cv2.resize(orig, (int(w*scale), int(h*scale)))
    polygons = load_annotation_local(img_name)

    while True:
        disp = frame.copy()
        draw_all(disp)
        draw_class_window()
        cv2.imshow("Annotator", disp)
        
        k = cv2.waitKey(30) & 0xFF
        if k == ord('d'): 
            save_pascal_voc(img_name, orig_shape)
            save_yolo_label_and_image(img_name, orig)
            prev_polygons = copy.deepcopy(polygons)
            current_index = (current_index + 1) % len(images)
            break
        elif k == ord('a'): 
            save_pascal_voc(img_name, orig_shape)
            save_yolo_label_and_image(img_name, orig)
            prev_polygons = copy.deepcopy(polygons)
            current_index = (current_index - 1) % len(images)
            break
        elif k == ord('b'): 
            force_new_poly = True
            selected_polygon = None
            print("[INFO] Draw mode active.")
        elif k == ord('r'): 
            if selected_polygon is not None:
                del polygons[selected_polygon]
                selected_polygon = None
        elif k == ord('s') and selected_polygon is not None: 
            cls = polygons[selected_polygon][-1]
            idx = CLASSLIST.index(cls)
            new_idx = (idx + 1) % len(CLASSLIST)
            polygons[selected_polygon][-1] = CLASSLIST[new_idx]
        elif k == ord('g'): 
            save_pascal_voc(img_name, orig_shape)
            save_yolo_label_and_image(img_name, orig)
            prev_polygons = copy.deepcopy(polygons)
            inference_current(conf=0.3)
        elif k == ord('t'): 
            save_pascal_voc(img_name, orig_shape)
            save_yolo_label_and_image(img_name, orig)
            if not training_running:
                threading.Thread(target=train_model, daemon=True).start()
        elif k == ord('e'): 
            repeat_last_annotations()
        elif k == 8: 
             if not images: continue
             base = os.path.splitext(img_name)[0]
             for p in [os.path.join(input_folder, img_name), 
                       os.path.join(output_folder, base+".xml"),
                       os.path.join(inference_images, img_name),
                       os.path.join(inference_labels, base+".txt")]:
                 if os.path.exists(p): os.remove(p)
             del images[current_index]
             polygons = []
             if not images: exit(0)
             current_index = max(0, current_index - 1)
             break
        elif k == 27 or k == ord('q'):
            save_pascal_voc(img_name, orig_shape)
            save_yolo_label_and_image(img_name, orig)
            cv2.destroyAllWindows()
            exit(0)
        elif k in range(49, 49+len(CLASSLIST)): 
             current_class = CLASSLIST[k-49]
             print(f"[INFO] Class: {current_class}")