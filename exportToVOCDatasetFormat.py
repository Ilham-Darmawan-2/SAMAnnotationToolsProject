import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import xml.etree.ElementTree as ET
from PIL import Image
import random
from pathlib import Path

WORKSPACENAME = None

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
        
        # Setup GUI
        self.setup_gui()
        
        # Load workspaces
        self.load_workspaces()
    
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
        self.log("=" * 70)
        
        # Generate Button - Make it SUPER visible!
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
        self.root.update()
    
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
                # Parse format: {workspaceName}-{index}
                if '-' in folder_name:
                    workspace_name = folder_name.rsplit('-', 1)[0]
                    if workspace_name not in workspace_dict:
                        workspace_dict[workspace_name] = 0
                    workspace_dict[workspace_name] += 1
        
        self.workspaces = sorted(workspace_dict.keys())
        
        self.workspace_listbox.delete(0, tk.END)
        for ws in self.workspaces:
            count = workspace_dict[ws]
            self.workspace_listbox.insert(tk.END, f"{ws} ({count} folder)")
        
        self.log(f"✅ Ditemukan {len(self.workspaces)} workspace")
    
    def on_workspace_select(self, event):
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
        self.log(f"\n🔧 Memperbaiki bbox yang keluar dari gambar...")
        
        output_dir = f"output/{workspace_name}"
        if not os.path.exists(output_dir):
            self.log(f"⚠️  Folder {output_dir} tidak ditemukan!")
            return False
        
        fixed_count = 0
        margin = 3
        
        for xml_file in os.listdir(output_dir):
            if not xml_file.endswith('.xml'):
                continue
            
            xml_path = os.path.join(output_dir, xml_file)
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
                
                # Clamp dengan margin
                new_xmin = max(margin, min(xmin, width - margin))
                new_ymin = max(margin, min(ymin, height - margin))
                new_xmax = max(margin, min(xmax, width - margin))
                new_ymax = max(margin, min(ymax, height - margin))
                
                if new_xmin != xmin or new_ymin != ymin or new_xmax != xmax or new_ymax != ymax:
                    bbox.find('xmin').text = str(new_xmin)
                    bbox.find('ymin').text = str(new_ymin)
                    bbox.find('xmax').text = str(new_xmax)
                    bbox.find('ymax').text = str(new_ymax)
                    changed = True
            
            if changed:
                tree.write(xml_path)
                fixed_count += 1
        
        self.log(f"✅ Bbox diperbaiki: {fixed_count} file")
        return True
    
    def fix_xml_structure(self, workspace_name):
        self.log(f"🔧 Memperbaiki struktur XML...")
        
        ann_dir = f"output/{workspace_name}"
        fixed = 0
        
        for file in os.listdir(ann_dir):
            if not file.endswith(".xml"):
                continue
            
            path = os.path.join(ann_dir, file)
            tree = ET.parse(path)
            root = tree.getroot()
            changed = False
            
            for obj in root.findall("object"):
                tags = {child.tag: child for child in obj}
                
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
        
        self.log(f"✅ XML diperbaiki: {fixed} file")
    
    def split_and_generate(self, workspace_name):
        self.log(f"\n📊 Melakukan split dataset...")
        
        output_dir = f"output/{workspace_name}"
        dataset_root = f"datasetsOutput/{workspace_name}"
        
        # Get all XML files
        xml_files = [f for f in os.listdir(output_dir) if f.endswith('.xml')]
        random.shuffle(xml_files)
        
        total_files = len(xml_files)
        train_count = int(total_files * self.train_ratio.get() / 100)
        
        train_files = xml_files[:train_count]
        valid_files = xml_files[train_count:]
        
        self.log(f"   📦 Train: {len(train_files)} file")
        self.log(f"   📦 Valid: {len(valid_files)} file")
        
        # Create split folders
        train_dir = os.path.join(dataset_root, "train")
        valid_dir = os.path.join(dataset_root, "valid")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        
        # Copy files to split folders
        for xml_file in train_files:
            base_name = os.path.splitext(xml_file)[0]
            
            # Copy XML
            shutil.copy(
                os.path.join(output_dir, xml_file),
                os.path.join(train_dir, xml_file)
            )
            
            # Copy image
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                img_path = os.path.join(output_dir, base_name + ext)
                if os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(train_dir, base_name + ext))
                    break
        
        for xml_file in valid_files:
            base_name = os.path.splitext(xml_file)[0]
            
            # Copy XML
            shutil.copy(
                os.path.join(output_dir, xml_file),
                os.path.join(valid_dir, xml_file)
            )
            
            # Copy image
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                img_path = os.path.join(output_dir, base_name + ext)
                if os.path.exists(img_path):
                    shutil.copy(img_path, os.path.join(valid_dir, base_name + ext))
                    break
        
        self.log(f"✅ Split dataset selesai")
        
        # Generate VOC format
        self.generate_voc_format(dataset_root)
    
    def generate_voc_format(self, dataset_root):
        self.log(f"\n🏗️  Membuat format VOC...")
        
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
        image_exts = (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")
        
        for split_name in splits:
            ann_folder = os.path.join(dataset_root, split_name)
            txt_path = os.path.join(imagesets_dir, f"{split_name}.txt")
            
            file_names = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(ann_folder)
                if f.endswith(".xml")
            )
            
            with open(txt_path, "w") as f:
                for name in file_names:
                    f.write(name + "\n")
            
            self.log(f"   ✅ {split_name}.txt dibuat ({len(file_names)} file)")
            
            # Move files
            for file_name in os.listdir(ann_folder):
                src_path = os.path.join(ann_folder, file_name)
                
                if file_name.lower().endswith(image_exts):
                    shutil.copy(src_path, os.path.join(jpeg_dir, file_name))
                elif file_name.lower().endswith(".xml"):
                    shutil.copy(src_path, os.path.join(annot_dir, file_name))
            
            self.log(f"   ✅ File {split_name} dipindahkan")
        
        # Clean up
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root)
            self.log(f"🧹 Folder {dataset_root} dihapus")
        
        self.log(f"\n🎉 Dataset VOC siap dipakai YOLOX!")
        self.log(f"📍 Lokasi: {voc_output}/VOCdevkit/VOC2012/")
    
    def generate_dataset(self):
        if not self.selected_workspace.get():
            messagebox.showwarning("Peringatan", "Pilih workspace terlebih dahulu!")
            return
        
        workspace_name = self.selected_workspace.get()
        
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 60)
        self.log(f"🚀 Memulai generate dataset untuk: {workspace_name}")
        self.log(f"📊 Split ratio: {int(self.train_ratio.get())}% train / {int(self.valid_ratio.get())}% valid")
        self.log(f"📋 Format: {self.dataset_format.get()}")
        self.log("=" * 60)
        
        self.generate_btn.config(state=tk.DISABLED, bg="#555555", text="⏳ Processing...")
        
        try:
            # Step 1: Fix bbox
            if not self.fix_bbox_clamp(workspace_name):
                raise Exception("Gagal memperbaiki bbox")
            
            # Step 2: Fix XML structure
            self.fix_xml_structure(workspace_name)
            
            # Step 3: Split and generate
            self.split_and_generate(workspace_name)
            
            self.log("=" * 60)
            messagebox.showinfo("Sukses! 🎉", f"Dataset {workspace_name} berhasil digenerate!\n\nCek folder: VOCDatasetOutput/")
            
        except Exception as e:
            self.log(f"\n❌ ERROR: {str(e)}")
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
        
        finally:
            self.generate_btn.config(state=tk.NORMAL, bg=self.accent_green, text="🚀 GENERATE DATASET")

def main():
    root = tk.Tk()
    app = DatasetGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()