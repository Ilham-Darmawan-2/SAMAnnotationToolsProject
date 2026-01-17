import cv2
import os
import numpy as np

INPUT_DIR = "datasetsInput/sampingRSPrimaya"
OUTPUT_DIR = "datasetsInput/caman2"
DISPLAY_W, DISPLAY_H = 1280, 720
SNAP_RADIUS = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg','.png','.jpeg'))])
assert len(image_files) > 0, "No images found"

first_img = cv2.imread(os.path.join(INPUT_DIR, image_files[0]))
h0, w0 = first_img.shape[:2]

scale_x = w0 / DISPLAY_W
scale_y = h0 / DISPLAY_H

display = cv2.resize(first_img, (DISPLAY_W, DISPLAY_H))

current_poly = []
all_polygons = []

def draw():
    img = display.copy()

    # Draw finished polygons
    for poly in all_polygons:
        pts = np.array([(int(x/scale_x), int(y/scale_y)) for x,y in poly], np.int32)
        cv2.polylines(img, [pts], True, (0,0,0), 2)

    # Draw current polygon
    if len(current_poly) > 0:
        pts = np.array([(int(x/scale_x), int(y/scale_y)) for x,y in current_poly], np.int32)
        cv2.polylines(img, [pts], False, (0,0,255), 2)
        for p in pts:
            cv2.circle(img, tuple(p), 5, (0,0,255), -1)
        cv2.circle(img, tuple(pts[0]), 10, (255,0,0), 2)

    cv2.imshow("Polygon Mask Tool", img)

def mouse_cb(event, x, y, flags, param):
    global current_poly, all_polygons

    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        if len(current_poly) >= 3:
            x0, y0 = current_poly[0]
            if abs(orig_x - x0) < SNAP_RADIUS*scale_x and abs(orig_y - y0) < SNAP_RADIUS*scale_y:
                all_polygons.append(current_poly.copy())
                print(f"[+] Polygon closed with {len(current_poly)} points")
                current_poly.clear()
                draw()
                return

        current_poly.append((orig_x, orig_y))
        draw()

cv2.namedWindow("Polygon Mask Tool")
cv2.setMouseCallback("Polygon Mask Tool", mouse_cb)

draw()
print("""
Controls:
Left Click : Add point
Snap to first point to close polygon
C           : Reset current polygon
ENTER       : Apply to all images & save
ESC         : Exit
""")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord('c'):
        current_poly.clear()
        print("[!] Current polygon cleared")
        draw()

    elif key == 13:  # Enter
        print("[*] Applying masks to all images...")
        for name in image_files:
            img = cv2.imread(os.path.join(INPUT_DIR, name))
            for poly in all_polygons:
                pts = np.array(poly, np.int32)
                cv2.fillPoly(img, [pts], (0,0,0))
            cv2.imwrite(os.path.join(OUTPUT_DIR, name), img)
        print(f"[OK] Saved masked images to: {OUTPUT_DIR}")
        break

cv2.destroyAllWindows()
