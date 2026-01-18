import os
import shutil
import signal
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import xml.etree.ElementTree as ET
from PIL import Image
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

WORKSPACENAME = None
PROCESSING_FLAG = False
CURRENT_WORKSPACE = None

class DatasetGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Generator")
        self.root.geometry("800x1024")
        self.root.resizable(True, True)
        
        # Dark theme colors
        self.bg_dark = "#1e1e1e"
        self.bg_secondary = "#2d2d2d"
        self.bg_tertiary = "#3d3d3d"
        self.fg_light = "#e0e0e0"
        self.fg_secondary = "#b0b0b0"
        self.accent_blue = "#0d7377"
        self.accent_green = "#14a76c"
        self.accent_orange = "#ff6b35"
        
        self.root.configure(bg=self.bg_dark)
        
        # Variables
        self.workspaces = []
        self.selected_workspace = tk.StringVar()
        self.train_ratio = tk.DoubleVar(value=70.0)
        self.valid_ratio = tk.DoubleVar(value=30.0)
        self.dataset_format = tk.StringVar(value="Pascal VOC")
        self.is_cancelled = False
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        # Setup GUI
        self.setup_gui()
        
        # Load workspaces
        self.load_workspaces()
    
    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self.handle_cancellation()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def handle_cancellation(self):
        """Handle cancellation and cleanup"""
        global PROCESSING_FLAG, CURRENT_WORKSPACE
        
        if PROCESSING_FLAG and CURRENT_WORKSPACE:
            self.log("\n" + "=" * 60)
            self.log("⚠️ PEMBATALAN TERDETEKSI!")
            self.log("🧹 Membersihkan proses dan menghapus file sementara...")
            self.log("=" * 60)
            
            self.is_cancelled = True
            
            # Cleanup output folder
            self.cleanup_workspace(CURRENT_WORKSPACE)
            
            self.log("✅ Pembersihan selesai. Aplikasi akan ditutup.")
            self.log("=" * 60)
            
            PROCESSING_FLAG = False
            self.root.after(2000, self.root.destroy)
        else:
            self.root.destroy()
    
    def cleanup_workspace(self, workspace_name):
        """Clean up workspace output folder"""
        output_dir = f"output/{workspace_name}"
        
        if os.path.exists(output_dir):
            try:
                # Remove all image files
                image_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                removed_count = 0
                
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.splitext(file)[1] in image_exts:
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except Exception as e:
                            self.log(f"⚠️ Gagal menghapus {file}: {e}")
                
                self.log(f"🗑️ {removed_count} file gambar dihapus dari {output_dir}")
                
                # Try to remove empty directory
                if not os.listdir(output_dir):
                    os.rmdir(output_dir)
                    self.log(f"📁 Folder {output_dir} dihapus")
                    
            except Exception as e:
                self.log(f"⚠️ Error saat cleanup: {e}")
    
    def setup_gui(self):
        # Title
        title_frame = tk.Frame(self.root, bg=self.accent_blue, height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="📦 Dataset Generator",
            font=("Segoe UI", 22, "bold"),
            bg=self.accent_blue,
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_dark, padx=25, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Workspace Selection
        workspace_frame = tk.LabelFrame(
            main_frame,
            text=" 📁 Pilih Workspace ",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_light,
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=2
        )
        workspace_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.workspace_listbox = tk.Listbox(
            workspace_frame,
            height=5,
            font=("Consolas", 10),
            bg=self.bg_tertiary,
            fg=self.fg_light,
            selectbackground=self.accent_blue,
            selectforeground="white",
            selectmode=tk.SINGLE,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.bg_dark,
            highlightcolor=self.accent_blue
        )
        self.workspace_listbox.pack(fill=tk.BOTH, expand=True)
        self.workspace_listbox.bind('<<ListboxSelect>>', self.on_workspace_select)
        
        # Selected workspace label
        self.selected_label = tk.Label(
            workspace_frame,
            text="❌ Belum ada workspace dipilih",
            font=("Segoe UI", 9),
            bg=self.bg_secondary,
            fg=self.fg_secondary
        )
        self.selected_label.pack(pady=(10, 0))
        
        # Dataset Format Selection
        format_frame = tk.LabelFrame(
            main_frame,
            text=" 🎯 Dataset Format ",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_light,
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=2
        )
        format_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.format_btn = tk.Button(
            format_frame,
            text="Pascal VOC",
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green,
            fg="white",
            activebackground=self.accent_green,
            activeforeground="white",
            cursor="hand2",
            height=2,
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED
        )
        self.format_btn.pack(fill=tk.X)
        
        format_info = tk.Label(
            format_frame,
            text="Format default untuk YOLOX training",
            font=("Segoe UI", 8),
            bg=self.bg_secondary,
            fg=self.fg_secondary
        )
        format_info.pack(pady=(5, 0))
        
        # Split Ratio Configuration
        split_frame = tk.LabelFrame(
            main_frame,
            text=" ⚙️ Konfigurasi Split Ratio ",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_light,
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=2
        )
        split_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Train ratio
        train_frame = tk.Frame(split_frame, bg=self.bg_secondary)
        train_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            train_frame, 
            text="Train:", 
            font=("Segoe UI", 10, "bold"), 
            width=10, 
            anchor="w",
            bg=self.bg_secondary,
            fg=self.fg_light
        ).pack(side=tk.LEFT)
        
        self.train_scale = tk.Scale(
            train_frame,
            from_=10,
            to=90,
            orient=tk.HORIZONTAL,
            variable=self.train_ratio,
            command=self.update_ratios,
            length=350,
            bg=self.bg_tertiary,
            fg=self.fg_light,
            troughcolor=self.bg_dark,
            activebackground=self.accent_blue,
            highlightthickness=0,
            relief=tk.FLAT
        )
        self.train_scale.pack(side=tk.LEFT, padx=10)
        
        self.train_label = tk.Label(
            train_frame, 
            text="70%", 
            font=("Segoe UI", 11, "bold"), 
            width=8,
            bg=self.bg_secondary,
            fg=self.accent_green
        )
        self.train_label.pack(side=tk.LEFT)
        
        # Valid ratio
        valid_frame = tk.Frame(split_frame, bg=self.bg_secondary)
        valid_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(
            valid_frame, 
            text="Valid:", 
            font=("Segoe UI", 10, "bold"), 
            width=10, 
            anchor="w",
            bg=self.bg_secondary,
            fg=self.fg_light
        ).pack(side=tk.LEFT)
        
        self.valid_scale = tk.Scale(
            valid_frame,
            from_=10,
            to=90,
            orient=tk.HORIZONTAL,
            variable=self.valid_ratio,
            command=self.update_ratios_from_valid,
            length=350,
            bg=self.bg_tertiary,
            fg=self.fg_light,
            troughcolor=self.bg_dark,
            activebackground=self.accent_blue,
            highlightthickness=0,
            relief=tk.FLAT
        )
        self.valid_scale.pack(side=tk.LEFT, padx=10)
        
        self.valid_label = tk.Label(
            valid_frame, 
            text="30%", 
            font=("Segoe UI", 11, "bold"), 
            width=8,
            bg=self.bg_secondary,
            fg=self.accent_orange
        )
        self.valid_label.pack(side=tk.LEFT)
        
        # Log Console
        log_frame = tk.LabelFrame(
            main_frame,
            text=" 📋 Log Console ",
            font=("Segoe UI", 11, "bold"),
            bg=self.bg_secondary,
            fg=self.fg_light,
            padx=10,
            pady=10,
            relief=tk.FLAT,
            borderwidth=2
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            font=("Consolas", 9),
            bg=self.bg_dark,
            fg=self.fg_light,
            insertbackground=self.fg_light,
            selectbackground=self.accent_blue,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.bg_dark,
            highlightcolor=self.accent_blue,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Add initial welcome message
        self.log("=" * 70)
        self.log("🎯 Dataset Generator - Ready!")
        self.log("⚠️  Tekan Ctrl+C untuk membatalkan proses dengan aman")
        self.log("=" * 70)
        
        # Generate Button
        button_container = tk.Frame(main_frame, bg=self.bg_dark, pady=10)
        button_container.pack(fill=tk.X)
        
        self.generate_btn = tk.Button(
            button_container,
            text="🚀 GENERATE DATASET",
            font=("Segoe UI", 14, "bold"),
            bg=self.accent_green,
            fg="white",
            activebackground="#12925f",
            activeforeground="white",
            cursor="hand2",
            height=2,
            relief=tk.RAISED,
            borderwidth=3,
            command=self.generate_dataset
        )
        self.generate_btn.pack(fill=tk.X, ipady=5)
        
        # Hover effect for generate button
        def on_enter(e):
            self.generate_btn.config(bg="#12925f", relief=tk.SUNKEN)
        
        def on_leave(e):
            if self.generate_btn['state'] == 'normal':
                self.generate_btn.config(bg=self.accent_green, relief=tk.RAISED)
        
        self.generate_btn.bind("<Enter>", on_enter)
        self.generate_btn.bind("<Leave>", on_leave)
    
    def log(self, message):
        """Add message to log console with auto-scroll"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()
    
    def load_workspaces(self):
        self.log("🔍 Memindai folder datasetsInput...")
        
        datasets_input = "datasetsInput"
        if not os.path.exists(datasets_input):
            os.makedirs(datasets_input)
            self.log("⚠️  Folder datasetsInput tidak ditemukan. Folder baru dibuat.")
            return
        
        workspace_dict = {}
        
        for folder_name in os.listdir(datasets_input):
            folder_path = os.path.join(datasets_input, folder_name)
            if os.path.isdir(folder_path):
                if '-' in folder_name:
                    workspace_name = folder_name.rsplit('-', 1)[0]
                    workspace_dict[workspace_name] = workspace_dict.get(workspace_name, 0) + 1
        
        self.workspaces = sorted(workspace_dict.keys())
        
        self.workspace_listbox.delete(0, tk.END)
        for ws in self.workspaces:
            count = workspace_dict[ws]
            self.workspace_listbox.insert(tk.END, f"{ws} ({count} folder)")
        
        self.log(f"✅ Ditemukan {len(self.workspaces)} workspace")
    
    def on_workspace_select(self, event):
        global WORKSPACENAME
        selection = self.workspace_listbox.curselection()
        if selection:
            idx = selection[0]
            workspace_text = self.workspace_listbox.get(idx)
            workspace_name = workspace_text.split(' (')[0]
            WORKSPACENAME = workspace_name
            self.selected_workspace.set(workspace_name)
            self.selected_label.config(
                text=f"✅ Workspace dipilih: {workspace_name}",
                fg=self.accent_green
            )
    
    def update_ratios(self, value):
        train = float(value)
        valid = 100 - train
        self.valid_ratio.set(valid)
        self.train_label.config(text=f"{int(train)}%")
        self.valid_label.config(text=f"{int(valid)}%")
    
    def update_ratios_from_valid(self, value):
        valid = float(value)
        train = 100 - valid
        self.train_ratio.set(train)
        self.train_label.config(text=f"{int(train)}%")
        self.valid_label.config(text=f"{int(valid)}%")
    
    def fix_bbox_clamp(self, workspace_name):
        """Optimized bbox fixing with batch processing"""
        if self.is_cancelled:
            return False
            
        self.log(f"\n🔧 Memperbaiki bbox yang keluar dari gambar...")
        
        output_dir = f"output/{workspace_name}"
        if not os.path.exists(output_dir):
            self.log(f"⚠️  Folder {output_dir} tidak ditemukan!")
            return False
        
        xml_files = [f for f in os.listdir(output_dir) if f.endswith('.xml')]
        fixed_count = 0
        margin = 3
        
        for xml_file in xml_files:
            if self.is_cancelled:
                return False
                
            xml_path = os.path.join(output_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                size = root.find('size')
                if size is None:
                    continue
                
                width = int(size.find('width').text)
                height = int(size.find('height').text)
                changed = False
                
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        continue
                    
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    
                    new_xmin = max(margin, min(xmin, width - margin))
                    new_ymin = max(margin, min(ymin, height - margin))
                    new_xmax = max(margin, min(xmax, width - margin))
                    new_ymax = max(margin, min(ymax, height - margin))
                    
                    if (new_xmin != xmin or new_ymin != ymin or 
                        new_xmax != xmax or new_ymax != ymax):
                        bbox.find('xmin').text = str(new_xmin)
                        bbox.find('ymin').text = str(new_ymin)
                        bbox.find('xmax').text = str(new_xmax)
                        bbox.find('ymax').text = str(new_ymax)
                        changed = True
                
                if changed:
                    tree.write(xml_path)
                    fixed_count += 1
                    
            except Exception as e:
                self.log(f"⚠️ Error pada {xml_file}: {e}")
        
        self.log(f"✅ Bbox diperbaiki: {fixed_count} file")
        return True
    
    def fix_xml_structure(self, workspace_name):
        """Optimized XML structure fixing"""
        if self.is_cancelled:
            return False
            
        self.log(f"🔧 Memperbaiki struktur XML...")
        
        ann_dir = f"output/{workspace_name}"
        xml_files = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
        fixed = 0
        
        for file in xml_files:
            if self.is_cancelled:
                return False
                
            path = os.path.join(ann_dir, file)
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                changed = False
                
                for obj in root.findall("object"):
                    tags = {child.tag for child in obj}
                    
                    if "pose" not in tags:
                        pose = ET.Element("pose")
                        pose.text = "Unspecified"
                        obj.insert(1, pose)
                        changed = True
                    
                    if "truncated" not in tags:
                        truncated = ET.Element("truncated")
                        truncated.text = "0"
                        obj.insert(2, truncated)
                        changed = True
                    
                    if "difficult" not in tags:
                        difficult = ET.Element("difficult")
                        difficult.text = "0"
                        obj.insert(3, difficult)
                        changed = True
                
                if changed:
                    tree.write(path)
                    fixed += 1
                    
            except Exception as e:
                self.log(f"⚠️ Error pada {file}: {e}")
        
        self.log(f"✅ XML diperbaiki: {fixed} file")
        return True
    
    def copy_all_images_to_output(self, workspace_name):
        """Optimized image copying with batch operations"""
        if self.is_cancelled:
            return False
            
        self.log("\n📥 Menyalin semua gambar ke folder output...")

        datasets_input_root = "datasetsInput"
        output_dir = f"output/{workspace_name}"
        os.makedirs(output_dir, exist_ok=True)

        image_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        copied = 0

        folders = [f for f in os.listdir(datasets_input_root) 
                   if f.startswith(workspace_name + "-")]

        for folder in folders:
            if self.is_cancelled:
                return False
                
            full_folder = os.path.join(datasets_input_root, folder)
            
            if os.path.isdir(full_folder):
                files = [f for f in os.listdir(full_folder) 
                        if os.path.splitext(f)[1] in image_exts]
                
                for f in files:
                    if self.is_cancelled:
                        return False
                        
                    src = os.path.join(full_folder, f)
                    dst = os.path.join(output_dir, f)
                    
                    if not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                            copied += 1
                        except Exception as e:
                            self.log(f"⚠️ Error menyalin {f}: {e}")

        self.log(f"✅ {copied} gambar berhasil disalin ke output")
        return True
    
    def split_and_generate(self, workspace_name):
        """Optimized dataset splitting"""
        if self.is_cancelled:
            return False
            
        self.log(f"\n📊 Melakukan split dataset...")
        
        output_dir = f"output/{workspace_name}"
        dataset_root = f"datasetsOutput/{workspace_name}"
        
        xml_files = [f for f in os.listdir(output_dir) if f.endswith('.xml')]
        random.shuffle(xml_files)
        
        total_files = len(xml_files)
        train_count = int(total_files * self.train_ratio.get() / 100)
        
        train_files = xml_files[:train_count]
        valid_files = xml_files[train_count:]
        
        self.log(f"   📦 Train: {len(train_files)} file")
        self.log(f"   📦 Valid: {len(valid_files)} file")
        
        train_dir = os.path.join(dataset_root, "train")
        valid_dir = os.path.join(dataset_root, "valid")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        
        image_exts = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
        
        # Process train files
        for xml_file in train_files:
            if self.is_cancelled:
                return False
                
            base_name = os.path.splitext(xml_file)[0]
            
            try:
                shutil.copy2(
                    os.path.join(output_dir, xml_file),
                    os.path.join(train_dir, xml_file)
                )
                
                for ext in image_exts:
                    img_path = os.path.join(output_dir, base_name + ext)
                    if os.path.exists(img_path):
                        shutil.move(img_path, os.path.join(train_dir, base_name + ext))
                        break
            except Exception as e:
                self.log(f"⚠️ Error memproses {xml_file}: {e}")
        
        # Process valid files
        for xml_file in valid_files:
            if self.is_cancelled:
                return False
                
            base_name = os.path.splitext(xml_file)[0]
            
            try:
                shutil.copy2(
                    os.path.join(output_dir, xml_file),
                    os.path.join(valid_dir, xml_file)
                )
                
                for ext in image_exts:
                    img_path = os.path.join(output_dir, base_name + ext)
                    if os.path.exists(img_path):
                        shutil.move(img_path, os.path.join(valid_dir, base_name + ext))
                        break
            except Exception as e:
                self.log(f"⚠️ Error memproses {xml_file}: {e}")
        
        self.log(f"✅ Split dataset selesai")
        
        return self.generate_voc_format(dataset_root)
    
    def generate_voc_format(self, dataset_root):
        """Optimized VOC format generation"""
        if self.is_cancelled:
            return False
            
        self.log(f"\n🗂️  Membuat format VOC...")
        
        voc_output = f"VOCDatasetOutput/{WORKSPACENAME}"
        os.makedirs(voc_output, exist_ok=True)
        
        yolox_root = os.path.join(voc_output, "VOCdevkit/VOC2012")
        imagesets_dir = os.path.join(yolox_root, "ImageSets", "Main")
        jpeg_dir = os.path.join(yolox_root, "JPEGImages")
        annot_dir = os.path.join(yolox_root, "Annotations")
        
        os.makedirs(imagesets_dir, exist_ok=True)
        os.makedirs(jpeg_dir, exist_ok=True)
        os.makedirs(annot_dir, exist_ok=True)
        
        splits = ["train", "valid"]
        image_exts = {".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"}
        
        for split_name in splits:
            if self.is_cancelled:
                return False
                
            ann_folder = os.path.join(dataset_root, split_name)
            txt_path = os.path.join(imagesets_dir, f"{split_name}.txt")
            
            file_names = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(ann_folder)
                if f.endswith(".xml")
            )
            
            with open(txt_path, "w") as f:
                f.write("\n".join(file_names) + "\n")
            
            self.log(f"   ✅ {split_name}.txt dibuat ({len(file_names)} file)")
            
            # Move files efficiently
            for file_name in os.listdir(ann_folder):
                if self.is_cancelled:
                    return False
                    
                src_path = os.path.join(ann_folder, file_name)
                
                try:
                    if os.path.splitext(file_name)[1] in image_exts:
                        shutil.copy2(src_path, os.path.join(jpeg_dir, file_name))
                    elif file_name.endswith(".xml"):
                        shutil.copy2(src_path, os.path.join(annot_dir, file_name))
                except Exception as e:
                    self.log(f"⚠️ Error memindahkan {file_name}: {e}")
            
            self.log(f"   ✅ File {split_name} dipindahkan")
        
        # Cleanup
        if os.path.exists(dataset_root):
            try:
                shutil.rmtree(dataset_root)
                self.log(f"🧹 Folder {dataset_root} dihapus")
            except Exception as e:
                self.log(f"⚠️ Error menghapus folder: {e}")
        
        self.log(f"\n🎉 Dataset VOC siap dipakai YOLOX!")
        self.log(f"📁 Lokasi: {voc_output}/VOCdevkit/VOC2012/")
        return True
    
    def generate_dataset(self):
        global PROCESSING_FLAG, CURRENT_WORKSPACE
        
        if not self.selected_workspace.get():
            messagebox.showwarning("Peringatan", "Pilih workspace terlebih dahulu!")
            return
        
        workspace_name = self.selected_workspace.get()
        CURRENT_WORKSPACE = workspace_name
        PROCESSING_FLAG = True
        self.is_cancelled = False
        
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 60)
        self.log(f"🚀 Memulai generate dataset untuk: {workspace_name}")
        self.log(f"📊 Split ratio: {int(self.train_ratio.get())}% train / {int(self.valid_ratio.get())}% valid")
        self.log(f"📋 Format: {self.dataset_format.get()}")
        self.log("=" * 60)
        
        self.generate_btn.config(state=tk.DISABLED, bg="#555555", text="⏳ Processing...")
        
        def run_generation():
            try:
                if not self.copy_all_images_to_output(workspace_name):
                    raise Exception("Proses dibatalkan saat menyalin gambar")
                
                if not self.fix_bbox_clamp(workspace_name):
                    raise Exception("Proses dibatalkan saat memperbaiki bbox")
                
                if not self.fix_xml_structure(workspace_name):
                    raise Exception("Proses dibatalkan saat memperbaiki XML")
                
                if not self.split_and_generate(workspace_name):
                    raise Exception("Proses dibatalkan saat split dataset")
                
                if not self.is_cancelled:
                    self.log("=" * 60)
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Sukses! 🎉", 
                        f"Dataset {workspace_name} berhasil digenerate!\n\nCek folder: VOCDatasetOutput/"
                    ))
                
            except Exception as e:
                if self.is_cancelled:
                    self.log(f"\n⚠️ PROSES DIBATALKAN OLEH USER")
                    self.cleanup_workspace(workspace_name)
                else:
                    self.log(f"\n❌ ERROR: {str(e)}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}"))
            
            finally:
                PROCESSING_FLAG = False
                CURRENT_WORKSPACE = None
                self.root.after(0, lambda: self.generate_btn.config(
                    state=tk.NORMAL, 
                    bg=self.accent_green, 
                    text="🚀 GENERATE DATASET"
                ))
        
        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

def main():
    root = tk.Tk()
    app = DatasetGeneratorGUI(root)
    
    def on_closing():
        global PROCESSING_FLAG
        if PROCESSING_FLAG:
            if messagebox.askokcancel("Quit", "Proses sedang berjalan. Yakin ingin keluar?\nFile sementara akan dibersihkan."):
                app.handle_cancellation()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()