#!/usr/bin/env python3
"""
Full GUI Annotation Tool - WITH DYNAMIC CLASS MANAGEMENT
Requirements:
  - python3, opencv-python, numpy, tkinter, pillow
  - ultralytics (pip install ultralytics)
"""
import cv2
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np

from utils.config import (CLASSLIST, state, input_folder, colorsPalette, output_folder, class_manager, inference_root, model_path, model_folder)
from utils.file_handler import load_annotation_local
from utils.inferenceObjectDetection import inference_current
from utils.image_manager import (repeat_last_annotations, delete_current_image, 
                                 save_and_backup_bboxes)
import sys
import queue
import subprocess
import shutil

EPOCH = 5
BATCH = 4

class TkTerminalRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass

    def update(self):
        while not self.queue.empty():
            msg = self.queue.get()
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)

class TrainingConfigDialog:
    """Dialog untuk konfigurasi training parameters"""
    
    def __init__(self, parent, default_epoch=100, default_batch=16):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Training Configuration")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (400 // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (300 // 2)
        self.dialog.geometry(f"+{x}+{y}")
        
        # Dark theme colors
        bg_dark = "#1e1e1e"
        bg_secondary = "#2d2d2d"
        fg_light = "#e0e0e0"
        accent_blue = "#0d7377"
        accent_green = "#14a76c"
        
        self.dialog.configure(bg=bg_dark)
        
        # Title
        title_frame = tk.Frame(self.dialog, bg=accent_blue, height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="⚙️ Training Configuration",
            font=("Segoe UI", 16, "bold"),
            bg=accent_blue,
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Main content
        content_frame = tk.Frame(self.dialog, bg=bg_dark, padx=30, pady=10)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Epoch input
        epoch_frame = tk.Frame(content_frame, bg=bg_dark)
        epoch_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            epoch_frame,
            text="Epochs:",
            font=("Segoe UI", 11, "bold"),
            bg=bg_dark,
            fg=fg_light,
            width=10,
            anchor="w"
        ).pack(side=tk.LEFT)
        
        self.epoch_var = tk.IntVar(value=default_epoch)
        epoch_spinbox = tk.Spinbox(
            epoch_frame,
            from_=1,
            to=1000,
            textvariable=self.epoch_var,
            font=("Segoe UI", 11),
            width=15,
            bg=bg_secondary,
            fg=fg_light,
            buttonbackground=bg_secondary,
            relief=tk.FLAT,
            insertbackground=fg_light
        )
        epoch_spinbox.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            epoch_frame,
            text="(1-1000)",
            font=("Segoe UI", 9),
            bg=bg_dark,
            fg="#888888"
        ).pack(side=tk.LEFT)
        
        # Batch size input
        batch_frame = tk.Frame(content_frame, bg=bg_dark)
        batch_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            batch_frame,
            text="Batch Size:",
            font=("Segoe UI", 11, "bold"),
            bg=bg_dark,
            fg=fg_light,
            width=10,
            anchor="w"
        ).pack(side=tk.LEFT)
        
        self.batch_var = tk.IntVar(value=default_batch)
        batch_spinbox = tk.Spinbox(
            batch_frame,
            from_=1,
            to=128,
            textvariable=self.batch_var,
            font=("Segoe UI", 11),
            width=15,
            bg=bg_secondary,
            fg=fg_light,
            buttonbackground=bg_secondary,
            relief=tk.FLAT,
            insertbackground=fg_light
        )
        batch_spinbox.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            batch_frame,
            text="(1-128)",
            font=("Segoe UI", 9),
            bg=bg_dark,
            fg="#888888"
        ).pack(side=tk.LEFT)
        
        # Info text
        info_text = tk.Label(
            content_frame,
            text="💡 Batch size tergantung VRAM GPU.\n"
                 "Epoch lebih banyak = training lebih lama.",
            font=("Segoe UI", 9),
            bg=bg_dark,
            fg="#aaaaaa",
            justify=tk.LEFT
        )
        info_text.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=bg_dark)
        button_frame.pack(fill=tk.X, pady=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="❌ Cancel",
            font=("Segoe UI", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            cursor="hand2",
            relief=tk.FLAT,
            command=self.cancel,
            width=12
        )
        cancel_btn.pack(side=tk.LEFT, padx=5, ipady=25)
        
        start_btn = tk.Button(
            button_frame,
            text="🚀 Start Training",
            font=("Segoe UI", 11, "bold"),
            bg=accent_green,
            fg="white",
            activebackground="#12925f",
            cursor="hand2",
            relief=tk.FLAT,
            command=self.start,
            width=15
        )
        start_btn.pack(side=tk.RIGHT, padx=5, ipady=25)
        
        # Bind Enter key
        self.dialog.bind('<Return>', lambda e: self.start())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        
    def start(self):
        """Validate and save configuration"""
        try:
            epoch = self.epoch_var.get()
            batch = self.batch_var.get()
            
            if epoch < 1 or epoch > 1000:
                messagebox.showwarning(
                    "Invalid Input",
                    "Epochs harus antara 1-1000!",
                    parent=self.dialog
                )
                return
            
            if batch < 1 or batch > 128:
                messagebox.showwarning(
                    "Invalid Input",
                    "Batch size harus antara 1-128!",
                    parent=self.dialog
                )
                return
            
            self.result = {
                'epoch': epoch,
                'batch': batch
            }
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Invalid input: {str(e)}",
                parent=self.dialog
            )
    
    def cancel(self):
        """Cancel dialog"""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Show dialog and wait for result"""
        self.dialog.wait_window()
        return self.result

class AnnotationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAMTEK Object Detection Annotator Tools - Dynamic Class Manager")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2b2b2b')
        
        # Load images
        self.images = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.images.sort()
        
        if not self.images:
            messagebox.showerror("Error", f"No images found in {input_folder}")
            root.destroy()
            return
        
        # Variables
        self.current_img_pil = None
        self.canvas_image = None
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_x = 0
        self.start_y = 0
        self.rect_id = None
        self.bbox_rects = []
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        self.auto_annotate_running = False
        self.auto_annotate_interval = 3000  # default 3 detik dalam ms
        self.auto_annotate_job = None
        self.inference_in_progress = False  # 🔒 LOCK untuk mencegah inference overlap
        
        # Setup UI
        self.create_widgets()

        # Redirect print() ke terminal GUI
        self.stdout_redirector = TkTerminalRedirector(self.terminal_text)
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector

        # Loop updater terminal (non-blocking)
        self.root.after(50, self._update_terminal)

        # Wait for window to render before loading image
        self.root.update_idletasks()
        self.root.after(100, self.initial_load)

        # Bind keyboard shortcuts
        self.bind_shortcuts()
    
    def _update_terminal(self):
        self.stdout_redirector.update()
        self.root.after(50, self._update_terminal)
    
    def bind_shortcuts(self):
        """Bind all keyboard shortcuts"""
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('a', lambda e: self.prev_image())
        self.root.bind('d', lambda e: self.next_image())
        self.root.bind('r', lambda e: self.delete_selected_bbox())
        self.root.bind('s', lambda e: self.change_class_selected())
        self.root.bind('t', lambda e: self.start_training())
        self.root.bind('g', lambda e: self.run_inference())
        self.root.bind('e', lambda e: self.repeat_annotations())
        self.root.bind('b', lambda e: self.toggle_force_new_bbox())
        self.root.bind('p', lambda e: self.toggle_auto_annotation())
        self.root.bind('<Delete>', lambda e: self.delete_image())
        self.root.bind('<BackSpace>', lambda e: self.delete_image())
        self.root.bind('f', lambda e: self.toggle_auto_annotate())
        
        # Bind number keys for class selection (1-9)
        for i in range(9):
            self.root.bind(str(i+1), lambda e, idx=i: self.select_class_by_number(idx))
    
    def toggle_bbox_text(self):
        state.show_bbox_text = not state.show_bbox_text
        if state.show_bbox_text:
            self.text_label.config(text="🅣 Text: ON", fg='#00ff00')
        else:
            self.text_label.config(text="🅣 Text: OFF", fg='#ff4444')
        self.update_display()

    def show_auto_annotate_config_dialog(self):
        """Dialog untuk konfigurasi auto annotate"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Auto Annotate Configuration")
        dialog.geometry("450x280")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Title
        title_frame = tk.Frame(dialog, bg='#0d7377', height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="🤖 Auto Annotate Configuration",
            font=("Segoe UI", 16, "bold"),
            bg='#0d7377',
            fg="white"
        ).pack(pady=15)
        
        # Content
        content_frame = tk.Frame(dialog, bg='#2b2b2b', padx=30, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Interval input
        interval_frame = tk.Frame(content_frame, bg='#2b2b2b')
        interval_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(
            interval_frame,
            text="Interval (detik):",
            font=("Segoe UI", 11, "bold"),
            bg='#2b2b2b',
            fg='white',
            width=15,
            anchor="w"
        ).pack(side=tk.LEFT)
        
        interval_var = tk.DoubleVar(value=self.auto_annotate_interval / 1000)
        interval_spinbox = tk.Spinbox(
            interval_frame,
            from_=0.5,
            to=60.0,
            increment=0.5,
            textvariable=interval_var,
            font=("Segoe UI", 11),
            width=15,
            bg='#1e1e1e',
            fg='white',
            buttonbackground='#1e1e1e',
            relief=tk.FLAT,
            insertbackground='white'
        )
        interval_spinbox.pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            interval_frame,
            text="(0.5-60)",
            font=("Segoe UI", 9),
            bg='#2b2b2b',
            fg="#888888"
        ).pack(side=tk.LEFT)
        
        # Info
        info_text = tk.Label(
            content_frame,
            text="💡 Auto annotate akan:\n"
                "   • Menjalankan inference pada gambar saat ini\n"
                "   • Pindah ke gambar berikutnya\n"
                "   • Mengulangi proses dengan interval yang ditentukan\n\n"
                "⚠️ Tekan tombol 'Stop Auto' untuk menghentikan",
            font=("Segoe UI", 9),
            bg='#2b2b2b',
            fg="#aaaaaa",
            justify=tk.LEFT
        )
        info_text.pack(pady=15)
        
        result = {'start': False}
        
        def on_start():
            try:
                interval_sec = interval_var.get()
                if interval_sec < 0.5 or interval_sec > 60:
                    messagebox.showwarning(
                        "Invalid Input",
                        "Interval harus antara 0.5-60 detik!",
                        parent=dialog
                    )
                    return
                
                result['start'] = True
                result['interval'] = int(interval_sec * 1000)  # convert to ms
                dialog.destroy()
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Invalid input: {str(e)}",
                    parent=dialog
                )
        
        def on_cancel():
            result['start'] = False
            dialog.destroy()
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(
            button_frame,
            text="❌ Cancel",
            font=("Segoe UI", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            cursor="hand2",
            relief=tk.FLAT,
            command=on_cancel,
            width=12
        ).pack(side=tk.LEFT, padx=5, ipady=8)
        
        tk.Button(
            button_frame,
            text="🚀 Start",
            font=("Segoe UI", 11, "bold"),
            bg="#14a76c",
            fg="white",
            activebackground="#12925f",
            cursor="hand2",
            relief=tk.FLAT,
            command=on_start,
            width=15
        ).pack(side=tk.RIGHT, padx=5, ipady=8)
        
        dialog.bind('<Return>', lambda e: on_start())
        dialog.bind('<Escape>', lambda e: on_cancel())
        
        dialog.wait_window()
        return result

    def start_auto_annotate(self):
        """Mulai auto annotate process"""
        if self.auto_annotate_running:
            messagebox.showinfo(
                "Already Running",
                "Auto annotate sudah berjalan!",
                parent=self.root
            )
            return
        
        # Show config dialog
        config = self.show_auto_annotate_config_dialog()
        
        if not config['start']:
            return
        
        self.auto_annotate_interval = config['interval']
        self.auto_annotate_running = True
        
        # Update UI
        self.update_auto_annotate_status()
        
        print(f"[AUTO ANNOTATE] Started with interval: {self.auto_annotate_interval/1000}s")
        
        # Start the process
        self._auto_annotate_cycle()

    def _auto_annotate_cycle(self):
        """Satu cycle dari auto annotate process dengan safety lock"""
        if not self.auto_annotate_running:
            return
        
        # 🔒 CEK APAKAH INFERENCE MASIH BERJALAN
        if self.inference_in_progress:
            print(f"[AUTO ANNOTATE] ⚠️  Skipping cycle - inference still in progress")
            # Retry lagi setelah 1 detik
            self.auto_annotate_job = self.root.after(1000, self._auto_annotate_cycle)
            return
        
        try:
            # 🔒 LOCK - Tandai inference sedang berjalan
            self.inference_in_progress = True
            
            # Run inference on current image
            print(f"[AUTO ANNOTATE] 🔄 Processing image {state.current_index + 1}/{len(self.images)}")
            self.run_inference()
            
            # Move to next image
            self.save_current()
            
            # 🔒 UNLOCK - Inference selesai
            self.inference_in_progress = False
            
            # Pindah ke gambar berikutnya
            state.current_index = (state.current_index + 1) % len(self.images)
            self.load_current_image()
            self.update_display()
            
            print(f"[AUTO ANNOTATE] ✅ Completed. Next cycle in {self.auto_annotate_interval/1000}s")
            
            # Schedule next cycle
            self.auto_annotate_job = self.root.after(
                self.auto_annotate_interval,
                self._auto_annotate_cycle
            )
            
        except Exception as e:
            # 🔒 UNLOCK jika terjadi error
            self.inference_in_progress = False
            
            print(f"[AUTO ANNOTATE] ❌ Error: {str(e)}")
            self.stop_auto_annotate()
            messagebox.showerror(
                "Auto Annotate Error",
                f"Error during auto annotate:\n{str(e)}",
                parent=self.root
            )

    def stop_auto_annotate(self):
        """Stop auto annotate process"""
        if not self.auto_annotate_running:
            return
        
        self.auto_annotate_running = False
        
        # Cancel scheduled job
        if self.auto_annotate_job:
            self.root.after_cancel(self.auto_annotate_job)
            self.auto_annotate_job = None
        
        # 🔒 RESET LOCK saat stop
        self.inference_in_progress = False
        
        # Update UI
        self.update_auto_annotate_status()
        
        print("[AUTO ANNOTATE] Stopped")
        messagebox.showinfo(
            "Auto Annotate Stopped",
            "Auto annotate telah dihentikan.",
            parent=self.root
        )

    def toggle_auto_annotate(self):
        """Toggle auto annotate on/off"""
        if self.auto_annotate_running:
            self.stop_auto_annotate()
        else:
            self.start_auto_annotate()

    def update_auto_annotate_status(self):
        """Update status label untuk auto annotate"""
        if self.auto_annotate_running:
            self.auto_annotate_label.config(
                text=f"🔄 Auto Annotate: RUNNING ({self.auto_annotate_interval/1000}s)",
                fg='#00ff00'
            )
            self.auto_annotate_btn.config(
                text="⏸️ Stop Auto",
                bg='#cc0000'
            )
        else:
            self.auto_annotate_label.config(
                text="🤖 Auto Annotate: OFF",
                fg='#888888'
            )
            self.auto_annotate_btn.config(
                text="▶️ Start Auto",
                bg='#00aa00'
            )
    
    def create_widgets(self):
        # ========== TOP TOOLBAR ==========
        toolbar = tk.Frame(self.root, bg='#1e1e1e', height=60)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Image navigation
        nav_frame = tk.Frame(toolbar, bg='#1e1e1e')
        nav_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Button(nav_frame, text="◀ Prev (A)", command=self.prev_image, 
                 bg='#404040', fg='white', padx=15, pady=8, 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
        
        self.img_label = tk.Label(nav_frame, text="Image 1/1", 
                                 bg='#1e1e1e', fg='#00ff00', font=('Arial', 11, 'bold'))
        self.img_label.pack(side=tk.LEFT, padx=15)
        
        tk.Button(nav_frame, text="Next (D) ▶", command=self.next_image, 
                 bg='#404040', fg='white', padx=15, pady=8,
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
        
        # Tools
        tools_frame = tk.Frame(toolbar, bg='#1e1e1e')
        tools_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Button(tools_frame, text="🔄 Repeat (E)", command=self.repeat_annotations,
                 bg='#0066cc', fg='white', padx=10, pady=8,
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(tools_frame, text="🤖 Inference (G)", command=self.run_inference,
                 bg='#00aa00', fg='white', padx=10, pady=8,
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(tools_frame, text="🎓 Train (T)", command=self.start_training,
                 bg='#cc6600', fg='white', padx=10, pady=8,
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(tools_frame, text="🗑️ Delete Img (Del)", command=self.delete_image,
                 bg='#cc0000', fg='white', padx=10, pady=8,
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Status indicators
        status_frame = tk.Frame(toolbar, bg='#1e1e1e')
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.auto_label = tk.Label(status_frame, text="🤖 Auto: OFF", 
                               bg='#1e1e1e', fg='#888888', font=('Arial', 10))
        self.auto_label.pack(side=tk.RIGHT, padx=10)
        
        status_frame = tk.Frame(toolbar, bg='#1e1e1e')
        status_frame.pack(side=tk.RIGHT, padx=10)
        
        self.force_label = tk.Label(status_frame, text="📦 Force Box: OFF", 
                                    bg='#1e1e1e', fg='#888888', font=('Arial', 10))
        self.force_label.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(status_frame, text="Toggle Auto (P)", command=self.toggle_auto_annotation,
                 bg='#404040', fg='white', padx=8, pady=8,
                 font=('Arial', 8)).pack(side=tk.RIGHT, padx=2)
        
        tk.Button(status_frame, text="Force Box (B)", command=self.toggle_force_new_bbox,
                 bg='#404040', fg='white', padx=8, pady=8,
                 font=('Arial', 8)).pack(side=tk.RIGHT, padx=2)
        
        self.text_label = tk.Label(status_frame, text="🅣 Text: ON",
                    bg='#1e1e1e', fg='#00ff00', font=('Arial', 10))
        self.text_label.pack(side=tk.RIGHT, padx=10)

        tk.Button(status_frame, text="Toggle Text",
                command=self.toggle_bbox_text,
                bg='#404040', fg='white', padx=8, pady=8,
                font=('Arial', 8)).pack(side=tk.RIGHT, padx=2)
        
        self.auto_annotate_label = tk.Label(
            status_frame,
            text="🤖 Auto Annotate: OFF",
            bg='#1e1e1e',
            fg='#888888',
            font=('Arial', 10, 'bold')
        )
        self.auto_annotate_label.pack(side=tk.RIGHT, padx=10)
        
        self.auto_annotate_btn = tk.Button(
            status_frame,
            text="▶️ Start Auto",
            command=self.toggle_auto_annotate,
            bg='#00aa00',
            fg='white',
            padx=10,
            pady=8,
            font=('Arial', 9, 'bold')
        )
        self.auto_annotate_btn.pack(side=tk.RIGHT, padx=5)
        
        # ========== MAIN CONTENT AREA ==========
        content = tk.Frame(self.root, bg='#2b2b2b')
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left sidebar - Class selector
        left_panel = tk.Frame(content, bg='#1e1e1e', width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # ===== CLASS MANAGEMENT SECTION =====
        class_mgmt_frame = tk.Frame(left_panel, bg='#1e1e1e')
        class_mgmt_frame.pack(fill=tk.X, pady=10, padx=5)
        
        tk.Label(class_mgmt_frame, text="📋 Class Manager", bg='#1e1e1e', fg='white', 
                font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        btn_container = tk.Frame(class_mgmt_frame, bg='#1e1e1e')
        btn_container.pack(side=tk.RIGHT)
        
        tk.Button(btn_container, text="➕", command=self.add_class_dialog,
                 bg='#00aa00', fg='white', padx=10, pady=4, 
                 font=('Arial', 11, 'bold'),
                 relief=tk.RAISED, borderwidth=2).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_container, text="➖", command=self.delete_class_dialog,
                 bg='#cc0000', fg='white', padx=10, pady=4, 
                 font=('Arial', 11, 'bold'),
                 relief=tk.RAISED, borderwidth=2).pack(side=tk.LEFT, padx=2)
        
        # Class list with scrollbar
        tk.Label(left_panel, text="Select Class:", bg='#1e1e1e', fg='#cccccc',
                font=('Arial', 9)).pack(anchor=tk.W, padx=10, pady=(5, 2))
        
        class_frame = tk.Frame(left_panel, bg='#1e1e1e')
        class_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(class_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.class_listbox = tk.Listbox(class_frame, bg='#2b2b2b', fg='white',
                                        selectmode=tk.SINGLE, font=('Arial', 10),
                                        yscrollcommand=scrollbar.set, height=12,
                                        activestyle='dotbox', relief=tk.FLAT)
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.class_listbox.yview)
        
        # Populate classes
        self.refresh_class_list()
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)
        
        # Separator
        tk.Frame(left_panel, bg='#444444', height=2).pack(fill=tk.X, pady=10)
        
        # Visibility toggles
        visibility_header = tk.Frame(left_panel, bg='#1e1e1e')
        visibility_header.pack(fill=tk.X, padx=10, pady=(5, 5))
        
        tk.Label(visibility_header, text="👁️ Visibility", bg='#1e1e1e', fg='white',
                font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        
        tk.Button(visibility_header, text="All", command=self.show_all_classes,
                 bg='#404040', fg='white', padx=6, pady=2,
                 font=('Arial', 7)).pack(side=tk.RIGHT, padx=2)
        
        tk.Button(visibility_header, text="None", command=self.hide_all_classes,
                 bg='#404040', fg='white', padx=6, pady=2,
                 font=('Arial', 7)).pack(side=tk.RIGHT, padx=2)
        
        # Visibility scroll frame
        vis_canvas = tk.Canvas(left_panel, bg='#1e1e1e', height=200, highlightthickness=0)
        vis_scrollbar = tk.Scrollbar(left_panel, orient="vertical", command=vis_canvas.yview)
        self.visibility_frame = tk.Frame(vis_canvas, bg='#1e1e1e')
        
        self.visibility_frame.bind(
            "<Configure>",
            lambda e: vis_canvas.configure(scrollregion=vis_canvas.bbox("all"))
        )
        
        vis_canvas.create_window((0, 0), window=self.visibility_frame, anchor="nw")
        vis_canvas.configure(yscrollcommand=vis_scrollbar.set)
        
        vis_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        vis_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.refresh_visibility_toggles()
        
        # Canvas area
        canvas_container = tk.Frame(content, bg='#000000', relief=tk.SUNKEN, borderwidth=2)
        canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Configure>', self.on_canvas_resize)

        # ========== BOTTOM TERMINAL PANEL ==========
        terminal_frame = tk.Frame(self.root, bg='#000000', height=180)
        terminal_frame.pack(side=tk.BOTTOM, fill=tk.X)
        terminal_frame.pack_propagate(False)

        tk.Label(
            terminal_frame,
            text="🖥️ Terminal Log",
            bg='#000000',
            fg='#00ff00',
            font=('Courier', 10, 'bold')
        ).pack(anchor=tk.W, padx=10, pady=(5, 0))

        self.terminal_text = tk.Text(
            terminal_frame,
            bg='#000000',
            fg='#00ff00',
            font=('Courier', 9),
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT
        )
        self.terminal_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Right sidebar - Info & shortcuts
        right_panel = tk.Frame(content, bg='#1e1e1e', width=280)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)
        
        tk.Label(right_panel, text="ℹ️ Information", bg='#1e1e1e', fg='white',
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        self.info_text = tk.Text(right_panel, bg='#2b2b2b', fg='#00ff00',
                                font=('Courier', 9), height=25, wrap=tk.WORD,
                                relief=tk.FLAT, padx=5, pady=5)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        tk.Label(right_panel, text="⌨️ Shortcuts", bg='#1e1e1e', fg='white',
                font=('Arial', 11, 'bold')).pack(pady=(15, 5))
        
        shortcuts = tk.Text(right_panel, bg='#2b2b2b', fg='white',
                           font=('Courier', 8), height=16, wrap=tk.WORD,
                           relief=tk.FLAT, padx=5, pady=5)
        shortcuts.pack(fill=tk.X, padx=5, pady=(0, 5))
        shortcuts.insert('1.0', """
        Navigation:
        A/← Previous    D/→ Next

        Annotation:
        Click+Drag: Draw Box
        Click Box: Select
        Drag Box: Move
        Drag Corner: Resize
        R: Delete Selected Box
        S: Change Class

        Tools:
        E: Repeat Last    G: Inference
        T: Train Model    B: Force Box
        P: Auto Mode      Del: Delete Img
        F: Auto Annotate  (Start/Stop)

        Classes:
        1-9: Quick Select
        ➕➖: Add/Remove Class
        """)
        shortcuts.config(state=tk.DISABLED)
    
    def refresh_class_list(self):
        """Refresh class listbox with current classes"""
        global CLASSLIST, colorsPalette
        
        # Get latest from class manager
        CLASSLIST = class_manager.get_classes()
        colorsPalette = class_manager.get_colors()
        
        # Clear listbox
        self.class_listbox.delete(0, tk.END)
        
        # Populate
        for i, cls in enumerate(CLASSLIST):
            display_text = f"{i+1}. {cls}"
            self.class_listbox.insert(tk.END, display_text)
            if i < len(colorsPalette):
                hex_color = self.rgb_to_hex(colorsPalette[i])
                self.class_listbox.itemconfig(i, bg=hex_color, fg='white')
        
        # Select first class if available
        if CLASSLIST:
            self.class_listbox.select_set(0)
            state.current_class = CLASSLIST[0]
            print(f"[GUI] Loaded {len(CLASSLIST)} classes")
    
    def refresh_visibility_toggles(self):
        """Refresh visibility checkboxes"""
        # Clear existing widgets
        for widget in self.visibility_frame.winfo_children():
            widget.destroy()
        
        # Recreate checkboxes
        self.visibility_vars = {}
        for cls in CLASSLIST:
            var = tk.BooleanVar(value=state.visible_class.get(cls, True))
            self.visibility_vars[cls] = var
            
            frame = tk.Frame(self.visibility_frame, bg='#1e1e1e')
            frame.pack(anchor=tk.W, pady=2, fill=tk.X)
            
            cb = tk.Checkbutton(frame, text=cls, variable=var,
                               bg='#1e1e1e', fg='white', selectcolor='#2b2b2b',
                               font=('Arial', 9), command=self.update_display,
                               activebackground='#1e1e1e', activeforeground='white')
            cb.pack(side=tk.LEFT, padx=2)
        
        # Update state visibility dict
        state.visible_class = {cls: True for cls in CLASSLIST}
    
    def show_all_classes(self):
        """Show all classes"""
        for cls, var in self.visibility_vars.items():
            var.set(True)
            state.visible_class[cls] = True
        self.update_display()
    
    def hide_all_classes(self):
        """Hide all classes"""
        for cls, var in self.visibility_vars.items():
            var.set(False)
            state.visible_class[cls] = False
        self.update_display()
    
    def add_class_dialog(self):
        global CLASSLIST
        """Show dialog to add new class"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Class")
        dialog.geometry("400x200")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        tk.Label(dialog, text="Add New Class", bg='#2b2b2b', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=15)
        
        tk.Label(dialog, text="Class Name:", bg='#2b2b2b', fg='white',
                font=('Arial', 10)).pack(pady=5)
        
        entry = tk.Entry(dialog, font=('Arial', 11), width=30)
        entry.pack(pady=10)
        entry.focus()
        
        hint = tk.Label(dialog, text="(Allowed: letters, numbers, underscore, dash)",
                       bg='#2b2b2b', fg='#888888', font=('Arial', 8))
        hint.pack()
        
        def on_submit():
            new_class = entry.get().strip()
            if new_class:
                success, message = class_manager.add_class(new_class)
                
                if success:
                    messagebox.showinfo("Success", message, parent=dialog)
                    self.refresh_class_list()
                    self.refresh_visibility_toggles()
                    self.update_display()
                    self.update_info()
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", message, parent=dialog)
                    entry.delete(0, tk.END)
                    entry.focus()
        
        btn_frame = tk.Frame(dialog, bg='#2b2b2b')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Add", command=on_submit,
                 bg='#00aa00', fg='white', padx=20, pady=8,
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                 bg='#666666', fg='white', padx=20, pady=8,
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        entry.bind('<Return>', lambda e: on_submit())
        entry.bind('<Escape>', lambda e: dialog.destroy())
        CLASSLIST = class_manager.get_classes()
    
    def delete_class_dialog(self):
        global CLASSLIST
        """Show dialog to delete class"""
        selection = self.class_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", 
                                 "Pilih class yang ingin dihapus terlebih dahulu!",
                                 parent=self.root)
            return
        
        idx = selection[0]
        if idx >= len(CLASSLIST):
            return
        
        class_to_delete = CLASSLIST[idx]
        
        # Count how many annotations will be affected
        affected_count = sum(1 for bbox in state.bboxes if bbox[4] == class_to_delete)
        
        # Confirmation dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Confirm Delete Class")
        dialog.geometry("450x250")
        dialog.configure(bg='#2b2b2b')
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        tk.Label(dialog, text="⚠️ Delete Class Warning", bg='#2b2b2b', fg='#ff6666',
                font=('Arial', 14, 'bold')).pack(pady=15)
        
        msg = f"Menghapus class '{class_to_delete}' akan:\n\n"
        msg += f"• Menghapus class dari dataset model assistant\n"
        msg += f"• Menghapus {affected_count} annotation(s) di gambar ini\n"
        msg += f"• Update semua label files di dataset\n"
        msg += f"• TIDAK DAPAT dipulihkan!\n\n"
        msg += f"Apakah Anda yakin?"
        
        tk.Label(dialog, text=msg, bg='#2b2b2b', fg='white',
                font=('Arial', 10), justify=tk.LEFT).pack(pady=10, padx=20)
        
        def on_confirm():
            success, message = class_manager.delete_class(class_to_delete)
            
            if success:
                # Update current bboxes - remove deleted class
                state.bboxes = [bbox for bbox in state.bboxes if bbox[4] != class_to_delete]
                
                self.refresh_class_list()
                self.refresh_visibility_toggles()
                self.update_display()
                self.update_info()
                
                dialog.destroy()
                messagebox.showinfo("Success", message, parent=self.root)
            else:
                dialog.destroy()
                messagebox.showerror("Error", message, parent=self.root)
        
        btn_frame = tk.Frame(dialog, bg='#2b2b2b')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="Yes, Delete", command=on_confirm,
                 bg='#cc0000', fg='white', padx=20, pady=8,
                 font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                 bg='#666666', fg='white', padx=20, pady=8,
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        
        CLASSLIST = class_manager.get_classes()
    
    def rgb_to_hex(self, rgb):
        """Convert BGR tuple to hex color"""
        return f'#{rgb[2]:02x}{rgb[1]:02x}{rgb[0]:02x}'
    
    def initial_load(self):
        """Initial load after window is ready"""
        self.load_current_image()
        self.update_display()
    
    def load_current_image(self):
        """Load current image and annotations"""
        img_name = self.images[state.current_index]
        img_path = os.path.join(input_folder, img_name)
        
        orig = cv2.imread(img_path)
        if orig is None:
            messagebox.showerror("Error", f"Cannot load {img_name}")
            return
        
        state.orig_shape = orig.shape
        h, w = state.orig_shape[:2]

        # Get canvas size
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 100:
            canvas_w = 1000
        if canvas_h < 100:
            canvas_h = 700
        
        # Calculate scale to fit
        padding = 20
        scale_w = (canvas_w - padding * 2) / w
        scale_h = (canvas_h - padding * 2) / h
        scale = min(scale_w, scale_h, 1.0)
        
        state.display_scale = scale
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        state.frame = cv2.resize(orig, (new_w, new_h))
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        self.current_img_pil = Image.fromarray(frame_rgb)
        
        # Center image
        self.canvas_offset_x = (canvas_w - new_w) // 2
        self.canvas_offset_y = (canvas_h - new_h) // 2
        
        # Load annotations
        state.bboxes = load_annotation_local(img_name)
        
        self.update_info()
    
    def update_display(self):
        """Redraw canvas with image and bboxes"""
        if self.current_img_pil is None:
            return
        
        img_copy = self.current_img_pil.copy()
        img_array = np.array(img_copy)
        
        # Draw all bboxes
        for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
            # Check visibility
            if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                continue
            
            # Get class color
            class_index = class_manager.get_class_index(cls)
            if class_index == -1:
                class_index = 0
            
            colors = class_manager.get_colors()
            color = colors[class_index] if class_index < len(colors) else (255, 0, 0)
            
            # Highlight selected bbox
            if i == state.selected_bbox:
                color = (60, 60, 200)  # Red for selected
            
            color_rgb = (color[2], color[1], color[0])  # BGR to RGB
            
            # Draw rectangle
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Draw label background
            label = cls
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if state.show_bbox_text:
                cv2.rectangle(img_array, (x1, y1-label_h-8), (x1+label_w+8, y1), color_rgb, -1)
                cv2.putText(img_array, label, (x1+4, y1-4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Force new bbox indicator
        if state.force_new_bbox:
            h, w = img_array.shape[:2]
            cv2.rectangle(img_array, (5, 5), (w-5, h-5), (200, 60, 60), 3)
        
        # Convert to PhotoImage
        img_with_boxes = Image.fromarray(img_array)
        self.photo = ImageTk.PhotoImage(img_with_boxes)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, 
                                anchor=tk.NW, image=self.photo)
        
        # Update image counter
        self.img_label.config(text=f"Image {state.current_index + 1}/{len(self.images)}")
    
    def update_info(self):
        """Update info panel"""
        img_name = self.images[state.current_index]
        
        # Count bboxes per class
        class_counts = {cls: 0 for cls in CLASSLIST}
        total_visible = 0
        total_hidden = 0
        
        for _, _, _, _, cls in state.bboxes:
            if cls in class_counts:
                class_counts[cls] += 1
                if self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    total_visible += 1
                else:
                    total_hidden += 1
        
        # Calculate dataset progress
        annotated_count = 0
        for img in self.images:
            xml_path = os.path.join(output_folder, os.path.splitext(img)[0] + ".xml")
            if os.path.exists(xml_path):
                annotated_count += 1
        
        progress_pct = (annotated_count / len(self.images) * 100) if self.images else 0
        
        # Build info text
        info = f"══════════════════════\n"
        info += f"CURRENT IMAGE\n"
        info += f"══════════════════════\n"
        info += f"📁 {img_name}\n"
        info += f"📐 {state.orig_shape[1]}x{state.orig_shape[0]}px\n"
        info += f"🔍 Scale: {state.display_scale:.2f}x\n\n"
        
        info += f"══════════════════════\n"
        info += f"ANNOTATIONS\n"
        info += f"══════════════════════\n"
        info += f"📦 Total: {len(state.bboxes)}\n"
        info += f"👁️  Visible: {total_visible}\n"
        if total_hidden > 0:
            info += f"🚫 Hidden: {total_hidden}\n"
        info += "\n"
        
        info += f"══════════════════════\n"
        info += f"CLASS DISTRIBUTION\n"
        info += f"══════════════════════\n"
        if any(class_counts.values()):
            for cls, count in class_counts.items():
                if count > 0:
                    visible = self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get()
                    icon = "✅" if visible else "⬜"
                    info += f"{icon} {cls}: {count}\n"
        else:
            info += "(No annotations)\n"
        
        info += f"\n══════════════════════\n"
        info += f"DATASET PROGRESS\n"
        info += f"══════════════════════\n"
        info += f"📊 {annotated_count}/{len(self.images)}\n"
        info += f"📈 {progress_pct:.1f}% Complete\n"
        info += f"⏳ {len(self.images) - annotated_count} Remaining\n\n"
        
        info += f"══════════════════════\n"
        info += f"CLASSES ({len(CLASSLIST)})\n"
        info += f"══════════════════════\n"
        for i, cls in enumerate(CLASSLIST[:10]):  # Show first 10
            current = "→" if cls == state.current_class else " "
            info += f"{current} {i+1}. {cls}\n"
        if len(CLASSLIST) > 10:
            info += f"... +{len(CLASSLIST)-10} more\n"
        
            info += f"\n╔══════════════════════╗\n"
            info += f"STATUS\n"
            info += f"╚══════════════════════╝\n"
            
            if self.auto_annotate_running:
                info += f"🔄 Auto Annotate: RUNNING\n"
                info += f"   Interval: {self.auto_annotate_interval/1000}s\n"
            
            if state.automated_annotation:
                info += f"🤖 Auto Mode: ON\n"
            if state.force_new_bbox:
                info += f"📦 Force Box: ON\n"
            if state.training_running:
                info += f"🎯 Training: RUNNING\n"
            if state.selected_bbox is not None:
                info += f"✏️  Selected: Box #{state.selected_bbox + 1}\n"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info)
        self.info_text.config(state=tk.DISABLED)
    
    def get_canvas_coords(self, event):
        """Convert event coords to image coords"""
        if self.current_img_pil is None:
            return None, None
        
        x = event.x - self.canvas_offset_x
        y = event.y - self.canvas_offset_y
        
        img_w = self.current_img_pil.width
        img_h = self.current_img_pil.height
        
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        
        return x, y
    
    def on_mouse_down(self, event):
        global CLASSLIST # Get latest from class manager 
        CLASSLIST = class_manager.get_classes()
        """Handle mouse button down"""
        if len(CLASSLIST) == 0:
            messagebox.showwarning("No Class", "Belum ada class. Tambahkan class dulu sebelum annotasi.")
            return
        
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        state.ix, state.iy = x, y
        state.selected_bbox = None
        state.resizing = False
        state.resize_mode = None
        
        # Force new bbox mode
        if state.force_new_bbox:
            self.drawing = True
            self.start_x, self.start_y = x, y
            return
        
        # Find clicked bbox (smallest first for overlaps)
        clicked = []
        for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                clicked.append((area, i))
        
        if clicked:
            # Select smallest bbox
            _, state.selected_bbox = min(clicked, key=lambda a: a[0])
            x1, y1, x2, y2, _ = state.bboxes[state.selected_bbox]
            
            # Check if clicking on corner handles for resizing
            handle_size = 10
            if abs(x - x1) < handle_size and abs(y - y1) < handle_size:
                state.resizing = True
                state.resize_mode = 'tl'
            elif abs(x - x2) < handle_size and abs(y - y1) < handle_size:
                state.resizing = True
                state.resize_mode = 'tr'
            elif abs(x - x1) < handle_size and abs(y - y2) < handle_size:
                state.resizing = True
                state.resize_mode = 'bl'
            elif abs(x - x2) < handle_size and abs(y - y2) < handle_size:
                state.resizing = True
                state.resize_mode = 'br'
            else:
                # Moving mode
                self.moving = True
            
            self.update_display()
            return
        
        # Start drawing new bbox
        self.drawing = True
        self.start_x, self.start_y = x, y
    
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        if self.drawing:
            # Redraw with temporary bbox
            self.update_display()
            
            img_array = np.array(self.current_img_pil)
            cv2.rectangle(img_array, (self.start_x, self.start_y), (x, y), (255, 0, 0), 2)
            
            # Redraw existing boxes
            for i, (x1, y1, x2, y2, cls) in enumerate(state.bboxes):
                if not self.visibility_vars.get(cls, tk.BooleanVar(value=True)).get():
                    continue
                
                class_index = class_manager.get_class_index(cls)
                if class_index == -1:
                    class_index = 0
                colors = class_manager.get_colors()
                color = colors[class_index] if class_index < len(colors) else (255, 0, 0)
                color_rgb = (color[2], color[1], color[0])
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color_rgb, 2)
            
            temp_img = Image.fromarray(img_array)
            self.photo = ImageTk.PhotoImage(temp_img)
            
            self.canvas.delete("all")
            self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, 
                                    anchor=tk.NW, image=self.photo)
        
        elif self.moving and state.selected_bbox is not None:
            # Move bbox
            dx, dy = x - state.ix, y - state.iy
            state.bboxes[state.selected_bbox][0] += dx
            state.bboxes[state.selected_bbox][1] += dy
            state.bboxes[state.selected_bbox][2] += dx
            state.bboxes[state.selected_bbox][3] += dy
            state.ix, state.iy = x, y
            self.update_display()
        
        elif state.resizing and state.selected_bbox is not None:
            # Resize bbox
            x1, y1, x2, y2, cls = state.bboxes[state.selected_bbox]
            if state.resize_mode == 'tl':
                state.bboxes[state.selected_bbox][0] = min(x, x2 - 5)
                state.bboxes[state.selected_bbox][1] = min(y, y2 - 5)
            elif state.resize_mode == 'tr':
                state.bboxes[state.selected_bbox][2] = max(x, x1 + 5)
                state.bboxes[state.selected_bbox][1] = min(y, y2 - 5)
            elif state.resize_mode == 'bl':
                state.bboxes[state.selected_bbox][0] = min(x, x2 - 5)
                state.bboxes[state.selected_bbox][3] = max(y, y1 + 5)
            elif state.resize_mode == 'br':
                state.bboxes[state.selected_bbox][2] = max(x, x1 + 5)
                state.bboxes[state.selected_bbox][3] = max(y, y1 + 5)
            self.update_display()
    
    def on_mouse_up(self, event):
        global CLASSLIST # Get latest from class manager 
        CLASSLIST = class_manager.get_classes()
        """Handle mouse button down"""
        if len(CLASSLIST) == 0:
            messagebox.showwarning("No Class", "Belum ada class. Tambahkan class dulu sebelum annotasi.")
            return
        """Handle mouse button up"""
        x, y = self.get_canvas_coords(event)
        if x is None:
            return
        
        if self.drawing:
            # Finalize new bbox
            x1 = min(self.start_x, x)
            y1 = min(self.start_y, y)
            x2 = max(self.start_x, x)
            y2 = max(self.start_y, y)
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            if w >= 8 and h >= 8:
                state.bboxes.append([x1, y1, x2, y2, state.current_class])
                print(f"[GUI] Added bbox: {state.current_class}")
                self.update_info()
            else:
                print("[GUI] Skipped tiny bbox (<8px)")
        
        self.drawing = False
        self.moving = False
        state.resizing = False
        state.force_new_bbox = False
        self.update_display()
        self.update_force_label()
    
    def on_mouse_move(self, event):
        """Handle mouse move for cursor"""
        pass
    
    def on_canvas_resize(self, event):
        """Handle canvas resize"""
        if self.current_img_pil is not None:
            canvas_w = event.width
            canvas_h = event.height
            img_w = self.current_img_pil.width
            img_h = self.current_img_pil.height
            
            self.canvas_offset_x = (canvas_w - img_w) // 2
            self.canvas_offset_y = (canvas_h - img_h) // 2
            
            self.update_display()
    
    def on_class_select(self, event):
        """Handle class selection from listbox"""
        selection = self.class_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(CLASSLIST):
                state.current_class = CLASSLIST[idx]
                print(f"[GUI] Selected class: {state.current_class}")
                self.update_info()
    
    def select_class_by_number(self, idx):
        """Select class by number key (0-8)"""
        if 0 <= idx < len(CLASSLIST):
            self.class_listbox.selection_clear(0, tk.END)
            self.class_listbox.select_set(idx)
            self.class_listbox.see(idx)
            state.current_class = CLASSLIST[idx]
            print(f"[GUI] Quick selected class {idx+1}: {state.current_class}")
            self.update_info()
    
    def next_image(self):
        """Navigate to next image"""
        self.save_current()
        state.current_index = (state.current_index + 1) % len(self.images)
        
        if state.automated_annotation:
            state.auto_annotation = True
        
        self.load_current_image()
        
        if state.auto_annotation:
            self.run_inference()
            state.auto_annotation = False
        
        self.update_display()
    
    def prev_image(self):
        """Navigate to previous image"""
        self.save_current()
        state.current_index = (state.current_index - 1) % len(self.images)
        
        if state.automated_annotation:
            state.auto_annotation = True
        
        self.load_current_image()
        
        if state.auto_annotation:
            self.run_inference()
            state.auto_annotation = False
        
        self.update_display()
    
    def save_current(self):
        """Save current annotations"""
        img_name = self.images[state.current_index]
        save_and_backup_bboxes(img_name, state.orig_shape, 
                              cv2.cvtColor(np.array(self.current_img_pil), cv2.COLOR_RGB2BGR), CLASSLIST)
    
    def delete_selected_bbox(self):
        """Delete selected bounding box"""
        if state.selected_bbox is not None:
            deleted_class = state.bboxes[state.selected_bbox][4]
            del state.bboxes[state.selected_bbox]
            state.selected_bbox = None
            print(f"[GUI] Deleted bbox: {deleted_class}")
            self.update_display()
            self.update_info()
    
    def change_class_selected(self):
        """Change class of selected bbox to next class"""
        if state.selected_bbox is not None and CLASSLIST:
            current_class = state.bboxes[state.selected_bbox][4]
            idx = class_manager.get_class_index(current_class)
            if idx == -1:
                idx = 0
            idx = (idx + 1) % len(CLASSLIST)
            new_class = CLASSLIST[idx]
            state.bboxes[state.selected_bbox][4] = new_class
            print(f"[GUI] Changed class: {current_class} → {new_class}")
            self.update_display()
            self.update_info()
    
    def monitor_training_process(self):
        """Cek apakah subprocess training sudah selesai"""

        if hasattr(state, "training_process"):
            process = state.training_process

            # poll() == None artinya masih jalan
            if process.poll() is None:
                # cek lagi 2 detik kemudian
                self.root.after(2000, self.monitor_training_process)
            else:
                # training selesai
                state.training_running = False

                messagebox.showinfo(
                    "Training Finished",
                    "YOLO Training process has completed!",
                    parent=self.root
                )
    
    def start_training(self):
        """Start YOLO training di terminal baru dengan argparse"""
        global CLASSLIST, model_folder, model_path
        self.save_current()
        
        # Get latest from class manager
        CLASSLIST = class_manager.get_classes()

        # Cek apakah process masih jalan
        if state.training_running and hasattr(state, "training_process"):
            if state.training_process.poll() is None:
                messagebox.showwarning(
                    "Training Running",
                    "Training already running. Please wait...",
                    parent=self.root
                )
                return
            else:
                # Process sudah selesai
                state.training_running = False

        # ========================================
        # TAMPILKAN DIALOG KONFIGURASI
        # ========================================
        config_dialog = TrainingConfigDialog(
            self.root,
            default_epoch=EPOCH,  # default dari global variable
            default_batch=BATCH
        )
        config = config_dialog.show()
        
        # Jika user cancel
        if config is None:
            return
        
        # Update epoch dan batch dari dialog
        current_epoch = config['epoch']
        current_batch = config['batch']
        # ========================================

        state.training_running = True

        train_script = os.path.join(
            os.path.dirname(__file__),
            "utils",
            "training.py"
        )

        dataset_root = inference_root
        images_folder = input_folder
        classlist = CLASSLIST

        class_args = []
        for c in classlist:
            class_args.append(c)

        cmd = [
            sys.executable,
            train_script,

            "--dataset_root", dataset_root,
            "--images_folder", images_folder,

            "--model_path", model_path,
            "--model_folder", model_folder,

            "--epochs", str(current_epoch),  # Gunakan nilai dari dialog
            "--batch", str(current_batch),   # Gunakan nilai dari dialog

            "--classlist"
        ] + class_args

        try:
            terminal_cmd = None

            if shutil.which("gnome-terminal"):
                terminal_cmd = ["gnome-terminal", "--wait", "--"] + cmd
            elif shutil.which("xterm"):
                terminal_cmd = ["xterm", "-hold", "-e"] + cmd
            elif shutil.which("konsole"):
                terminal_cmd = ["konsole", "-e"] + cmd
            else:
                terminal_cmd = cmd

            # SIMPAN PROCESS OBJECT
            process = subprocess.Popen(terminal_cmd)

            # simpan ke state
            state.training_process = process

            messagebox.showinfo(
                "Training Started",
                f"Training started in new terminal!\n\n"
                f"📊 Epochs: {current_epoch}\n"
                f"📦 Batch Size: {current_batch}\n\n"
                f"Check terminal for progress.",
                parent=self.root
            )

            # Mulai monitoring proses di background
            self.monitor_training_process()

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to start training:\n{str(e)}",
                parent=self.root
            )

        state.training_running = False
    
    def run_inference(self):
        """Run inference on current image"""
        self.save_current()
        print("[GUI] Running inference...")
        inference_current(self.images, state.current_index, conf=0.3)
        self.update_display()
        self.update_info()
        print("[GUI] Inference completed!")
    
    def repeat_annotations(self):
        global CLASSLIST, colorsPalette
        
        # Get latest from class manager
        CLASSLIST = class_manager.get_classes()
        """Repeat annotations from previous image"""
        repeat_last_annotations(self.images, state.current_index, CLASSLIST)
        self.update_display()
        self.update_info()
        print("[GUI] Repeated annotations from previous image")
    
    def toggle_force_new_bbox(self):
        """Toggle force new bbox mode"""
        state.force_new_bbox = not state.force_new_bbox
        self.update_force_label()
        self.update_display()
        print(f"[GUI] Force new bbox: {'ON' if state.force_new_bbox else 'OFF'}")
    
    def update_force_label(self):
        """Update force bbox indicator label"""
        if state.force_new_bbox:
            self.force_label.config(text="📦 Force Box: ON", fg='#00ff00')
        else:
            self.force_label.config(text="📦 Force Box: OFF", fg='#888888')
    
    def toggle_auto_annotation(self):
        """Toggle auto annotation mode"""
        state.automated_annotation = not state.automated_annotation
        if state.automated_annotation:
            self.auto_label.config(text="🤖 Auto: ON", fg='#00ff00')
        else:
            self.auto_label.config(text="🤖 Auto: OFF", fg='#888888')
        print(f"[GUI] Auto annotation: {'ON' if state.automated_annotation else 'OFF'}")
    
    def delete_image(self):
        """Delete current image and all its annotations"""
        img_name = self.images[state.current_index]
        if messagebox.askyesno("Confirm Delete", 
                              f"Delete image '{img_name}' and all annotations?\n\nThis cannot be undone!",
                              parent=self.root):
            result = delete_current_image(self.images, state.current_index)
            if result[0] is None:
                messagebox.showinfo("Dataset Empty", 
                                  "All images deleted. Exiting application...",
                                  parent=self.root)
                self.root.destroy()
                return
            
            state.current_index, self.images = result
            self.load_current_image()
            self.update_display()
            print(f"[GUI] Deleted image: {img_name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationGUI(root)
    root.mainloop()