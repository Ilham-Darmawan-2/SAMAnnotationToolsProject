"""
Dynamic Class Manager for Annotation Tool
Handles loading, saving, adding, and deleting classes with color generation
"""
import os
import random
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import colorsys

class ClassManager:
    def __init__(self, workspace_name: str):
        self.workspace_name = workspace_name
        self.config_dir = "configs"
        self.class_file = os.path.join(self.config_dir, f"{workspace_name}.txt")
        self.inference_root = f"inference/{workspace_name}"
        self.labels_folder = os.path.join(self.inference_root, "labels")
        self.data_yaml_path = os.path.join(self.inference_root, "data.yaml")
        self.output_folder = f"output/{workspace_name}"
        
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.labels_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.classes: List[str] = []
        self.colors: List[Tuple[int, int, int]] = []
        
        # Load or initialize classes
        self._load_classes()
        self._generate_colors()
    
    def _load_classes(self):
        """Load classes from file or create empty file"""
        if os.path.exists(self.class_file):
            with open(self.class_file, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            print(f"[ClassManager] Loaded {len(self.classes)} classes from {self.class_file}")
        else:
            self.classes = []
            self._save_classes()
            print(f"[ClassManager] Created new class file: {self.class_file}")
    
    def _save_classes(self):
        """Save classes to file"""
        with open(self.class_file, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")
        print(f"[ClassManager] Saved {len(self.classes)} classes to {self.class_file}")
    
    def _generate_colors(self):
        """Generate strong, non-white, high-contrast colors for each class"""
        self.colors = []
        random.seed(42)

        n = len(self.classes)

        for i in range(n):
            # Hue terdistribusi merata (golden angle)
            h = (i * 0.61803398875) % 1.0  

            # Saturation & Value dijaga supaya gak pucat
            s = random.uniform(0.75, 0.95)   # warna pekat
            v = random.uniform(0.45, 0.75)   # hindari terlalu terang

            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            # Convert ke 0-255 (BGR buat OpenCV)
            color = (int(b * 255), int(g * 255), int(r * 255))

            self.colors.append(color)

        print(f"[ClassManager] Generated {len(self.colors)} high-contrast colors")
    
    def add_class(self, class_name: str) -> Tuple[bool, str]:
        """
        Add new class
        Returns: (success, message)
        """
        # Validasi: cek spasi dan simbol
        if not class_name.replace('_', '').replace('-', '').isalnum():
            return False, "Class name hanya boleh huruf, angka, underscore (_) dan dash (-)"
        
        # Validasi: cek jika kosong
        if not class_name.strip():
            return False, "Class name tidak boleh kosong"
        
        # Validasi: cek duplikat (case sensitive)
        if class_name in self.classes:
            return False, f"Class '{class_name}' sudah ada"
        
        # Tambah class di index paling akhir
        self.classes.append(class_name)
        self._save_classes()
        self._generate_colors()  # Regenerate colors
        self._update_data_yaml()  # Update data.yaml
        
        return True, f"Class '{class_name}' berhasil ditambahkan"
    
    def delete_class(self, class_name: str) -> Tuple[bool, str]:
        """
        Delete class and update all label files (YOLO and XML)
        Returns: (success, message)
        """
        if class_name not in self.classes:
            return False, f"Class '{class_name}' tidak ditemukan"
        
        old_index = self.classes.index(class_name)
        
        # Buat mapping perubahan index
        # Class sebelum yang dihapus tetap sama, class setelahnya turun 1
        index_mapping = {}
        for i, cls in enumerate(self.classes):
            if i < old_index:
                index_mapping[i] = i  # Tetap sama
            elif i > old_index:
                index_mapping[i] = i - 1  # Turun 1
            # i == old_index akan dihapus, tidak perlu mapping
        
        # Hapus class dari list
        self.classes.remove(class_name)
        self._save_classes()
        self._generate_colors()
        
        # Update semua label files (YOLO .txt)
        yolo_updated = self._update_yolo_label_files(old_index, index_mapping)
        
        # Update semua XML files
        xml_updated = self._update_xml_files(class_name)
        
        # Update data.yaml
        self._update_data_yaml()
        
        return True, f"Class '{class_name}' dihapus. {yolo_updated} YOLO labels dan {xml_updated} XML files diperbarui"
    
    def _update_yolo_label_files(self, deleted_index: int, index_mapping: Dict[int, int]) -> int:
        """
        Update all YOLO label files after class deletion
        Returns: number of files updated
        """
        if not os.path.exists(self.labels_folder):
            return 0
        
        updated_count = 0
        label_files = [f for f in os.listdir(self.labels_folder) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(self.labels_folder, label_file)
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                modified = False
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_idx = int(parts[0])
                        
                        # Skip jika class yang dihapus
                        if old_class_idx == deleted_index:
                            modified = True
                            continue
                        
                        # Update index jika perlu
                        if old_class_idx in index_mapping:
                            new_class_idx = index_mapping[old_class_idx]
                            parts[0] = str(new_class_idx)
                            modified = True
                        
                        new_lines.append(' '.join(parts) + '\n')
                
                # Tulis ulang jika ada perubahan
                if modified:
                    with open(label_path, 'w') as f:
                        f.writelines(new_lines)
                    updated_count += 1
            
            except Exception as e:
                print(f"[ClassManager] Error updating YOLO {label_file}: {e}")
        
        print(f"[ClassManager] Updated {updated_count} YOLO label files")
        return updated_count
    
    def _update_xml_files(self, deleted_class_name: str) -> int:
        """
        Update all XML annotation files by removing objects with deleted class
        Returns: number of files updated
        """
        if not os.path.exists(self.output_folder):
            return 0
        
        updated_count = 0
        xml_files = [f for f in os.listdir(self.output_folder) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(self.output_folder, xml_file)
            
            try:
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Find all <object> elements
                objects_to_remove = []
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text == deleted_class_name:
                        objects_to_remove.append(obj)
                
                # Remove objects with deleted class
                if objects_to_remove:
                    for obj in objects_to_remove:
                        root.remove(obj)
                    
                    # Save updated XML
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    updated_count += 1
                    print(f"[ClassManager] Removed {len(objects_to_remove)} object(s) from {xml_file}")
            
            except Exception as e:
                print(f"[ClassManager] Error updating XML {xml_file}: {e}")
        
        print(f"[ClassManager] Updated {updated_count} XML files")
        return updated_count
    
    def _update_data_yaml(self):
        """Update data.yaml with current classes"""
        if not os.path.exists(self.data_yaml_path):
            # Buat data.yaml baru jika belum ada
            train_path = os.path.join(self.inference_root, "train/images")
            val_path = os.path.join(self.inference_root, "val/images")
        else:
            # Parse existing data.yaml untuk ambil train dan val path
            train_path = ""
            val_path = ""
            try:
                with open(self.data_yaml_path, 'r') as f:
                    for line in f:
                        if line.startswith('train:'):
                            train_path = line.split('train:')[1].strip()
                        elif line.startswith('val:'):
                            val_path = line.split('val:')[1].strip()
            except Exception as e:
                print(f"[ClassManager] Error reading data.yaml: {e}")
                train_path = os.path.join(self.inference_root, "train/images")
                val_path = os.path.join(self.inference_root, "val/images")
        
        # Jika path masih kosong, set default
        if not train_path:
            train_path = os.path.join(self.inference_root, "train/images")
        if not val_path:
            val_path = os.path.join(self.inference_root, "val/images")
        
        # Tulis data.yaml
        with open(self.data_yaml_path, 'w') as f:
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")
        
        print(f"[ClassManager] Updated {self.data_yaml_path}")
    
    def get_classes(self) -> List[str]:
        """Get list of classes"""
        return self.classes.copy()
    
    def get_colors(self) -> List[Tuple[int, int, int]]:
        """Get list of colors"""
        return self.colors.copy()
    
    def get_class_index(self, class_name: str) -> int:
        """Get index of class, returns -1 if not found"""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return -1
    
    def refresh(self):
        """Reload classes from file"""
        self._load_classes()
        self._generate_colors()