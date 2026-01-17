# 🎨 GUI Annotation Tool

✅ **Tested on Python3.10** - I recommend that you run it on Python 3.10 like I do.

## ✨ Features

### Modern Full GUI Interface
- **Left Panel**: Class selector dengan color coding
- **Center**: Canvas untuk annotasi dengan zoom/pan support
- **Right Panel**: Info & keyboard shortcuts reference
- **Top Toolbar**: Quick access buttons untuk semua fungsi

### Key Features
✅ **Drag & Drop BBox** - Gambar bbox langsung di canvas  
✅ **Resize & Move** - Drag corner untuk resize, drag tengah untuk move  
✅ **Class Selection** - Klik class di sidebar atau tekan 1-9  
✅ **Visibility Toggle** - Hide/show class tertentu dengan checkbox  
✅ **Auto Annotation** - AI inference otomatis saat navigasi  
✅ **Training** - Train model langsung dari GUI (background thread)  
✅ **Smart Scaling** - Bbox conversion tetap presisi antara display & original size

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install Tkinter
sudo apt install python3.10-tk

# Run Tools
python ObjectDetectionAnnotator.py
```

### Folder Structure
```
project/
├── ObjectDetectionAnnotator.py
├── utils/
│   ├── config.py
│   ├── file_handler.py
│   ├── ui_handler.py
│   ├── class_selector.py
│   ├── training.py
│   └── image_manager.py
├── datasetsInput/
│   └── ppeKujangv2-13/   # Your images here
├── output/               # Pascal VOC XML files
├── inference/            # YOLO format files
└── models/               # Trained models
```

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `A` / `←` | Previous Image |
| `D` / `→` | Next Image |
| `R` | Delete Selected BBox |
| `S` | Change Class (cycle) |
| `T` | Train Model |
| `G` | Run Inference |
| `E` | Repeat Last Annotations |
| `B` | Force New BBox Mode |
| `P` | Toggle Auto Annotation |
| `Del` / `Backspace` | Delete Current Image |
| `1-9` | Select Class by Number |

## 🖱️ Mouse Controls

- **Click & Drag**: Draw new bbox
- **Click on BBox**: Select bbox
- **Drag BBox**: Move selected bbox
- **Drag Corner**: Resize selected bbox
- **Right Panel**: Scroll for shortcuts info

## 🎯 Workflow

1. **Load Images** - Auto load dari `datasetsInput/`
2. **Draw BBoxes** - Click & drag di canvas
3. **Select Class** - Klik di sidebar atau tekan angka 1-9
4. **Navigate** - Tekan `D` (next) atau `A` (prev)
5. **Train** - Tekan `T` untuk training (min 10 images)
6. **Auto Annotate** - Tekan `P` untuk enable auto inference saat navigasi

## 💡 Tips

1. **Precision Conversion**: Sistem menggunakan `round()` untuk konversi koordinat display ↔ original, meminimalkan error presisi
2. **Force New BBox**: Aktifkan mode ini (`B`) jika klik di dalam bbox existing tapi mau buat bbox baru
3. **Auto Annotation**: Perfect untuk dataset besar - model akan annotate otomatis saat navigasi
4. **Visibility Toggle**: Hide class yang sudah selesai untuk fokus ke class lain
5. **Training Progress**: Check console/terminal untuk training progress

## 🐛 Troubleshooting

**BBox tidak presisi saat save?**
- Sudah fixed dengan `round()` conversion di semua module

**Canvas kosong?**
- Resize window atau restart app
- Check image path di config

**Training gagal?**
- Minimal 10 images dengan annotasi
- Check `ultralytics` sudah terinstall

**Auto annotation tidak jalan?**
- Train model dulu minimal 1x (`T`)
- Enable auto mode (`P`)

## 📝 Notes

- **Sistem konversi bbox** tetap sama seperti CLI - menggunakan `display_scale` untuk presisi
- **Thread safety** untuk training - tidak akan freeze UI
- **Auto save** saat navigasi - annotations tersimpan otomatis

---

## Export Dataset
- Saat ini dataset yang disupport hanya format pascalVOC
```bash
# Export Dataset
python exportToVOCDatasetFormat.py
```

# Tools
- file python pada folder tools merupakan tools saya selama annotasi dataset, saya belum berfikir untuk membuatnya menjadi GUI atau semacamnya, anda bebas menghapusnya atau tidak menggunakannya.

🎉 **Enjoy annotating with style!** 🎉