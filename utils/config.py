"""
Configuration file for annotation tool - UPDATED VERSION
Now uses ClassManager for dynamic class handling
"""
import os
from utils.class_manager import ClassManager
import tkinter as tk
from tkinter import filedialog

# ======== SELECT INPUT FOLDER AT START ========
_root = tk.Tk()
_root.withdraw()  # Hide main tkinter window

input_folder = filedialog.askdirectory(
    title="Select Input Image Folder"
)

_root.destroy()

if not input_folder:
    raise SystemExit("No input folder selected. Exiting.")

# ======== WORKSPACE CONFIG ========
folder_name = os.path.basename(os.path.normpath(input_folder))

# Default
workspaceName = folder_name

# Buang suffix "-{index}" kalau ada
if "-" in folder_name:
    base, suffix = folder_name.rsplit("-", 1)
    if suffix.isdigit():
        workspaceName = base

output_folder = f"output/{workspaceName}"
inference_root = f"inference/{workspaceName}"
inference_images = os.path.join(inference_root, "images")
inference_labels = os.path.join(inference_root, "labels")
model_folder = f"models/{workspaceName}"
model_path = os.path.join(model_folder, "modelAssistant.pt")

# Create directories
for d in [output_folder, inference_images, inference_labels, model_folder]:
    os.makedirs(d, exist_ok=True)

# ======== DYNAMIC CLASS CONFIGURATION ========
# Initialize ClassManager
class_manager = ClassManager(workspaceName)

# Get classes and colors from ClassManager
CLASSLIST = class_manager.get_classes()
colorsPalette = class_manager.get_colors()

# Jika tidak ada class, set default minimal 1 class
if not CLASSLIST:
    print("[Config] No classes found, using default class 'Object'")
    CLASSLIST = ["Object"]
    class_manager.add_class("Object")
    colorsPalette = class_manager.get_colors()

# ======== UI SETTINGS ========
CLASS_HEIGHT = 35
CLASS_WINDOW_W = 220
CLASS_WINDOW_H = 360

# ======== STATE VARIABLES ========
class State:
    def __init__(self):
        self.current_index = 0
        self.bboxes = []
        self.prev_bboxes = []
        self.selected_bbox = None
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.resize_mode = None
        self.force_new_bbox = False
        self.ix, self.iy = -1, -1
        self.display_scale = 1.0
        self.frame = None
        self.orig_shape = None
        self.scroll_offset = 0
        self.training_running = False
        self.automated_annotation = False
        self.auto_annotation = False
        self.current_class = CLASSLIST[0] if CLASSLIST else "Object"
        self.visible_class = {cls: True for cls in CLASSLIST}
        self.show_bbox_text = True
        self.training_running = False
        self.training_process = None
        


# Global state instance
state = State()