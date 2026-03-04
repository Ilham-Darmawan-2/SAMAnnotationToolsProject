"""
Duplicate Image Finder
======================
Mendeteksi gambar identik dalam folder terlepas dari ukurannya.
Menggunakan perceptual hashing (pHash) agar tahan terhadap perbedaan resolusi/format.

Dependensi:
    pip install Pillow imagehash opencv-python numpy
"""

import os
import sys
from pathlib import Path
from itertools import combinations

try:
    from PIL import Image
    import imagehash
    import cv2
    import numpy as np
except ImportError:
    print("❌ Library belum terinstall. Jalankan dulu:")
    print("   pip install Pillow imagehash opencv-python numpy")
    sys.exit(1)

# ─────────────────────────────────────────────
# !! KONFIGURASI — UBAH SESUAI KEBUTUHAN !!
# ─────────────────────────────────────────────

# Folder yang akan di-scan (gunakan path absolut atau relatif)
FOLDER = "datasetsInput/fireSmoke-1"

# Toleransi perbedaan hash:
#   0 = identik persis (recommended untuk duplikat sejati)
#   5 = hampir mirip (tangkap gambar yang sedikit beda crop/kompresi)
THRESHOLD = 0

# ─────────────────────────────────────────────
# Konfigurasi tampilan (sesuaikan layar)
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
MAX_DISPLAY_WIDTH  = 1800   # px total lebar dua gambar berdampingan (sesuai layar 21")
MAX_DISPLAY_HEIGHT = 850    # px tinggi maksimum saat ditampilkan


def collect_images(folder: Path) -> list[Path]:
    """Kumpulkan semua file gambar secara rekursif."""
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(Path(root) / f)
    return sorted(images)


def compute_hashes(image_paths: list[Path]) -> dict[Path, imagehash.ImageHash]:
    """Hitung perceptual hash untuk setiap gambar."""
    hashes = {}
    total = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        try:
            with Image.open(path) as img:
                hashes[path] = imagehash.phash(img)
            print(f"\r  Menghitung hash... {i}/{total}", end="", flush=True)
        except Exception as e:
            print(f"\n  ⚠️  Skip {path.name}: {e}")
    print()
    return hashes


def find_duplicate_pairs(hashes: dict, threshold: int) -> list[tuple[Path, Path]]:
    """Temukan pasangan gambar yang hash-nya berjarak <= threshold."""
    pairs = []
    items = list(hashes.items())
    for (path_a, hash_a), (path_b, hash_b) in combinations(items, 2):
        distance = hash_a - hash_b
        if distance <= threshold:
            pairs.append((path_a, path_b, distance))
    # Urutkan: yang paling mirip dulu (distance terkecil)
    pairs.sort(key=lambda x: x[2])
    return pairs


def make_side_by_side_cv(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Buat gambar berdampingan (numpy array BGR) yang muat di layar."""
    gap = 20

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    scale = min(
        (MAX_DISPLAY_WIDTH - gap) / 2 / max(w_a, w_b),
        MAX_DISPLAY_HEIGHT / max(h_a, h_b),
        1.0
    )

    def resize(img, w, h):
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    ra = resize(img_a, w_a, h_a)
    rb = resize(img_b, w_b, h_b)

    total_h = max(ra.shape[0], rb.shape[0])
    total_w = ra.shape[1] + gap + rb.shape[1]

    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    # Tempel gambar kiri (tengah vertikal)
    y_a = (total_h - ra.shape[0]) // 2
    canvas[y_a:y_a + ra.shape[0], :ra.shape[1]] = ra

    # Tempel gambar kanan
    x_b = ra.shape[1] + gap
    y_b = (total_h - rb.shape[0]) // 2
    canvas[y_b:y_b + rb.shape[0], x_b:x_b + rb.shape[1]] = rb

    return canvas


def draw_label(canvas: np.ndarray, text_lines: list[str]) -> np.ndarray:
    """Tambahkan bar teks di bawah gambar."""
    bar_h = 28 * len(text_lines) + 10
    bar = np.full((bar_h, canvas.shape[1], 3), 20, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(bar, line, (10, 24 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def show_and_confirm(path_a: Path, path_b: Path, distance: int, pair_index: int, total: int) -> bool:
    """
    Tampilkan dua gambar berdampingan pakai OpenCV window.
    Return True  → hapus satu gambar (Enter)
    Return False → lewati (Backspace)
    """
    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))

    if img_a is None or img_b is None:
        print(f"  ❌ Gagal membuka gambar.")
        return False

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    print(f"\n{'─'*60}")
    print(f"  Pasangan {pair_index}/{total}  |  Jarak hash: {distance} {'(identik persis)' if distance == 0 else ''}")
    print(f"  [KIRI]  {path_a.name}  ({w_a}×{h_a}, {path_a.stat().st_size // 1024} KB)")
    print(f"  [KANAN] {path_b.name}  ({w_b}×{h_b}, {path_b.stat().st_size // 1024} KB)")
    print(f"{'─'*60}")
    print("  Di window OpenCV:")
    print("    ENTER     → hapus (yang ukuran filenya lebih kecil)")
    print("    BACKSPACE → lewati")
    print("    Q / ESC   → keluar program")

    canvas = make_side_by_side_cv(img_a, img_b)
    canvas = draw_label(canvas, [
        f"[{pair_index}/{total}]  KIRI: {path_a.name} ({w_a}x{h_a}, {path_a.stat().st_size//1024}KB)   "
        f"KANAN: {path_b.name} ({w_b}x{h_b}, {path_b.stat().st_size//1024}KB)   hash-dist: {distance}",
        "ENTER = hapus  |  BACKSPACE = lewati  |  Q / ESC = keluar"
    ])

    win = "Duplicate Finder"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, canvas)

    while True:
        key = cv2.waitKey(0)

        if key in (13, 10):        # Enter
            cv2.destroyWindow(win)
            return True
        elif key in (8, 127):      # Backspace / Delete
            cv2.destroyWindow(win)
            return False
        elif key in (ord('q'), ord('Q'), 27):  # Q atau ESC
            cv2.destroyAllWindows()
            print("\n  Keluar dari program.")
            sys.exit(0)


def delete_smaller(path_a: Path, path_b: Path) -> Path:
    """Hapus file yang ukurannya lebih kecil, kembalikan path yang dihapus."""
    size_a = path_a.stat().st_size
    size_b = path_b.stat().st_size

    # Jika ukuran sama, hapus yang kedua (b)
    to_delete = path_b if size_a >= size_b else path_a
    to_delete.unlink()
    return to_delete


def main():
    folder = Path(FOLDER).resolve()
    if not folder.is_dir():
        print(f"❌ Folder tidak ditemukan: {folder}")
        sys.exit(1)

    print(f"\n🔍 Scanning folder: {folder}")
    print(f"   Threshold      : {THRESHOLD} {'(identik persis)' if THRESHOLD == 0 else ''}\n")

    # 1. Kumpulkan gambar
    images = collect_images(folder)
    if not images:
        print("  Tidak ada file gambar yang ditemukan.")
        sys.exit(0)
    print(f"  ✅ Ditemukan {len(images)} gambar.\n")

    # 2. Hash semua gambar
    hashes = compute_hashes(images)

    # 3. Cari pasangan duplikat
    pairs = find_duplicate_pairs(hashes, THRESHOLD)
    if not pairs:
        print("\n  ✅ Tidak ada duplikat yang ditemukan. Semua gambar unik!")
        sys.exit(0)

    print(f"\n  ⚠️  Ditemukan {len(pairs)} pasangan duplikat potensial.\n")

    # 4. Proses satu per satu dengan konfirmasi
    deleted_count  = 0
    skipped_count  = 0
    deleted_paths  = set()

    for idx, (path_a, path_b, distance) in enumerate(pairs, 1):
        # Skip jika salah satu sudah dihapus di iterasi sebelumnya
        if path_a in deleted_paths or path_b in deleted_paths:
            continue

        # Cek file masih ada (mungkin dihapus manual)
        if not path_a.exists() or not path_b.exists():
            continue

        should_delete = show_and_confirm(path_a, path_b, distance, idx, len(pairs))

        if should_delete:
            removed = delete_smaller(path_a, path_b)
            deleted_paths.add(removed)
            deleted_count += 1
            print(f"\n  🗑️  Dihapus: {removed.name}")
        else:
            skipped_count += 1
            print(f"\n  ⏭️  Dilewati.")

    # 5. Ringkasan
    print(f"\n{'═'*60}")
    print(f"  SELESAI")
    print(f"  Gambar dihapus : {deleted_count}")
    print(f"  Pasangan dilewati : {skipped_count}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()