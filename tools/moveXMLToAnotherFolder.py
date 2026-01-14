import os
import shutil

src = "temp/vehiclePublic"   # ganti dengan folder sumber
dst = "output/vehicle"   # ganti dengan folder tujuan

# pastikan folder tujuan ada
os.makedirs(dst, exist_ok=True)

# loop semua file di folder sumber
for fname in os.listdir(src):
    if fname.lower().endswith(".xml"):
        src_path = os.path.join(src, fname)
        dst_path = os.path.join(dst, fname)
        shutil.move(src_path, dst_path)
        print(f"Moved: {fname}")

print("Selesai memindahkan semua XML.")
