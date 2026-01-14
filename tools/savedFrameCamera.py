import cv2
import os
import time

# --- CONFIG ---
output_folder = "datasetsInput/ppeKujangv2-5"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_interval = 1  # detik
skip_frame = 3      # skip setiap 3 frame
rtsp_url = 0

# --- Fungsi bantu untuk buka kamera ---
def open_camera():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Kamera tidak ditemukan!")
        return None
    return cap

# --- Ambil frame pertama untuk menggambar bbox ---
print("Menghubungkan ke kamera...")
cap = open_camera()
if cap is None:
    exit()

# Tunggu sampai dapat frame pertama
for _ in range(30):
    ret, first_frame = cap.read()
    if ret:
        break
    time.sleep(0.2)

if not ret:
    print("Gagal membaca frame pertama!")
    exit()

orig_h, orig_w = first_frame.shape[:2]

# Hitung rasio resize ke ~1MP
target_pixels = 1_000_000
scale_factor = (target_pixels / (orig_w * orig_h)) ** 0.5
new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)

resized_frame = cv2.resize(first_frame, (new_w, new_h))
temp_frame = resized_frame.copy()

drawing = False
ix, iy = -1, -1
bboxes_resized = []

# --- Fungsi callback mouse ---
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, temp_frame

    frame_copy = resized_frame.copy()
    for (x1, y1, x2, y2) in bboxes_resized:
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
        bboxes_resized.append((x1, y1, x2, y2))
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    temp_frame = frame_copy.copy()

# --- Window untuk menggambar ---
cv2.namedWindow("Draw BBox (1MP Preview)")
cv2.setMouseCallback("Draw BBox (1MP Preview)", draw_bbox)

print("🟩 Gambar kotak hitam di area yang ingin disembunyikan (pada versi resize 1 MP).")
print("👉 Tekan 's' jika sudah selesai menggambar, atau 'r' untuk reset.")

while True:
    cv2.imshow("Draw BBox (1MP Preview)", temp_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # selesai
        break
    elif key == ord('r'):  # reset
        bboxes_resized = []
        temp_frame = resized_frame.copy()
    elif key == ord('q'):
        print("Dibatalkan oleh user.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw BBox (1MP Preview)")

# --- Rescale bbox ke ukuran asli ---
bboxes = []
for (x1, y1, x2, y2) in bboxes_resized:
    X1 = int(x1 / scale_factor)
    Y1 = int(y1 / scale_factor)
    X2 = int(x2 / scale_factor)
    Y2 = int(y2 / scale_factor)
    bboxes.append((X1, Y1, X2, Y2))

print(f"Total kotak yang digambar: {len(bboxes)}")
for i, b in enumerate(bboxes):
    print(f"  BBox {i+1}: {b}")

# --- Tutup dulu kamera lama dan buka ulang (biar koneksi segar) ---
cap.release()
time.sleep(1)
cap = open_camera()
if cap is None:
    exit()

print("Mulai mengambil frame... tekan 'q' untuk berhenti.")
frame_count = 0
last_save_time = 0
fail_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame!")
        fail_count += 1
        time.sleep(0.2)

        # --- jika gagal 10x, coba reconnect kamera ---
        if fail_count >= 10:
            print("🔁 Reconnecting ke kamera...")
            cap.release()
            time.sleep(1)
            cap = open_camera()
            if cap is None:
                print("❌ Gagal reconnect ke kamera.")
                break
            fail_count = 0
        continue
    else:
        fail_count = 0

    frame_count += 1
    if frame_count % skip_frame != 0:
        continue

    # timpa area bbox dengan warna hitam
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.imshow("Camera", frame)

    # Simpan frame setiap interval
    current_time = time.time()
    if current_time - last_save_time >= frame_interval:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(current_time))
        file_name = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(file_name, frame)
        print(f"Frame disimpan di {file_name}")
        last_save_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
