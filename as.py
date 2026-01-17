import os
import xml.etree.ElementTree as ET

def fix_bbox_margin(xml_folder):
    """
    Memperbaiki bbox yang terlalu dekat dengan tepi frame.
    
    Logika:
    - Jika jarak bbox ke tepi < 25px, maka pepetin ke tepi (margin 1px)
    - Jika jarak >= 25px, biarkan apa adanya
    
    Args:
        xml_folder: Path ke folder yang berisi file XML
    """
    
    if not os.path.exists(xml_folder):
        print(f"❌ Folder {xml_folder} tidak ditemukan!")
        return
    
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    
    if not xml_files:
        print(f"⚠️  Tidak ada file XML di folder {xml_folder}")
        return
    
    print(f"🔍 Ditemukan {len(xml_files)} file XML")
    print(f"📁 Folder: {xml_folder}")
    print("=" * 70)
    
    fixed_count = 0
    total_bbox_fixed = 0
    threshold = 25  # Jarak threshold (px)
    margin = 1      # Margin final (px)
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            if size is None:
                print(f"⚠️  {xml_file}: tidak ada tag <size>, skip")
                continue
            
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            file_changed = False
            bbox_count = 0
            
            # Process each object
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                # Get current bbox values
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Store original values
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = xmin, ymin, xmax, ymax
                
                # Check and fix left edge (xmin)
                if xmin < threshold:
                    xmin = margin
                
                # Check and fix top edge (ymin)
                if ymin < threshold:
                    ymin = margin
                
                # Check and fix right edge (xmax)
                if (width - xmax) < threshold:
                    xmax = width - margin
                
                # Check and fix bottom edge (ymax)
                if (height - ymax) < threshold:
                    ymax = height - margin
                
                # Update if changed
                if (xmin != orig_xmin or ymin != orig_ymin or 
                    xmax != orig_xmax or ymax != orig_ymax):
                    
                    bbox.find('xmin').text = str(int(xmin))
                    bbox.find('ymin').text = str(int(ymin))
                    bbox.find('xmax').text = str(int(xmax))
                    bbox.find('ymax').text = str(int(ymax))
                    
                    file_changed = True
                    bbox_count += 1
                    
                    # Show details
                    class_name = obj.find('name').text if obj.find('name') is not None else "unknown"
                    print(f"   📦 {class_name}:")
                    if xmin != orig_xmin:
                        print(f"      xmin: {int(orig_xmin)} → {int(xmin)} (jarak ke kiri: {int(orig_xmin)}px)")
                    if ymin != orig_ymin:
                        print(f"      ymin: {int(orig_ymin)} → {int(ymin)} (jarak ke atas: {int(orig_ymin)}px)")
                    if xmax != orig_xmax:
                        print(f"      xmax: {int(orig_xmax)} → {int(xmax)} (jarak ke kanan: {int(width - orig_xmax)}px)")
                    if ymax != orig_ymax:
                        print(f"      ymax: {int(orig_ymax)} → {int(ymax)} (jarak ke bawah: {int(height - orig_ymax)}px)")
            
            # Save if changed
            if file_changed:
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                fixed_count += 1
                total_bbox_fixed += bbox_count
                print(f"✅ {xml_file}: {bbox_count} bbox diperbaiki (size: {width}x{height})")
                print()
        
        except Exception as e:
            print(f"❌ Error di {xml_file}: {str(e)}")
    
    print("=" * 70)
    print(f"🎉 Selesai!")
    print(f"   📄 File diperbaiki: {fixed_count}/{len(xml_files)}")
    print(f"   📦 Total bbox diperbaiki: {total_bbox_fixed}")
    print(f"   ⚙️  Threshold: {threshold}px | Margin: {margin}px")


def analyze_bbox_distribution(xml_folder):
    """
    Analisis distribusi jarak bbox ke tepi frame.
    Berguna untuk menentukan threshold yang tepat.
    """
    
    if not os.path.exists(xml_folder):
        print(f"❌ Folder {xml_folder} tidak ditemukan!")
        return
    
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    
    distances = {
        'left': [],
        'top': [],
        'right': [],
        'bottom': []
    }
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            if size is None:
                continue
            
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is None:
                    continue
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                distances['left'].append(xmin)
                distances['top'].append(ymin)
                distances['right'].append(width - xmax)
                distances['bottom'].append(height - ymax)
        
        except:
            continue
    
    print("\n📊 ANALISIS JARAK BBOX KE TEPI FRAME")
    print("=" * 70)
    
    for edge, dist_list in distances.items():
        if dist_list:
            close_boxes = sum(1 for d in dist_list if d < 25)
            very_close = sum(1 for d in dist_list if d < 10)
            touching = sum(1 for d in dist_list if d < 2)
            
            print(f"\n{edge.upper()}:")
            print(f"   Total bbox: {len(dist_list)}")
            print(f"   Jarak < 25px: {close_boxes} ({close_boxes/len(dist_list)*100:.1f}%)")
            print(f"   Jarak < 10px: {very_close} ({very_close/len(dist_list)*100:.1f}%)")
            print(f"   Jarak < 2px:  {touching} ({touching/len(dist_list)*100:.1f}%)")
            print(f"   Min: {min(dist_list):.1f}px | Max: {max(dist_list):.1f}px")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # === CONFIG ===
    XML_FOLDER = "output/vehicle"  # Ganti dengan folder XML kamu
    
    # === PILIH MODE ===
    
    # Mode 1: Analisis dulu (optional - untuk cek data)
    print("🔍 MODE ANALISIS")
    analyze_bbox_distribution(XML_FOLDER)
    
    # Mode 2: Fix bbox
    print("\n\n🔧 MODE PERBAIKAN")
    response = input("\nLanjut perbaiki bbox? (y/n): ")
    
    if response.lower() == 'y':
        fix_bbox_margin(XML_FOLDER)
    else:
        print("❌ Dibatalkan")