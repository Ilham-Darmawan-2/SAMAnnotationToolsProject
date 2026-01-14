"""
Image navigation and management
"""
import os
import copy
from .config import state, output_folder, inference_images, inference_labels, input_folder

def repeat_last_annotations(images, current_index, classList):
    """Repeat annotations from previous image"""
    from .file_handler import save_pascal_voc, save_yolo_label_and_image
    
    if not state.prev_bboxes:
        print("[INFO] No previous annotations to repeat.")
        return
    
    print("[INFO] Reapplying previous annotations...")
    state.bboxes = [b.copy() for b in state.prev_bboxes]
    save_pascal_voc(images[current_index], state.frame.shape)
    save_yolo_label_and_image(images[current_index], state.frame, classList)

def delete_current_image(images, current_index):
    """Delete current image and all associated files"""
    import cv2
    import numpy as np
    
    if not images:
        print("[WARN] No images left.")
        return current_index, images

    img_name = images[current_index]
    base_name = os.path.splitext(img_name)[0]

    # Paths to all related files
    input_path = os.path.join(input_folder, img_name)
    xml_path = os.path.join(output_folder, base_name + ".xml")
    infer_img_path = os.path.join(inference_images, img_name)
    infer_label_path = os.path.join(inference_labels, base_name + ".txt")

    # Delete files if they exist
    for f in [input_path, xml_path, infer_img_path, infer_label_path]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Deleted: {f}")

    # Remove from list
    del images[current_index]

    # Clear bboxes
    state.bboxes.clear()

    # Determine new index
    if current_index >= len(images):
        current_index = max(0, len(images) - 1)

    # If no images left, show notification
    if not images:
        print("[INFO] All images deleted.")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "No images left", (400, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Annotator", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None, None

    return current_index, images

def save_and_backup_bboxes(img_name, orig_shape, orig, ClassList):
    """Save annotations and backup current bboxes"""
    from .file_handler import save_pascal_voc, save_yolo_label_and_image
    
    save_pascal_voc(img_name, orig_shape)
    save_yolo_label_and_image(img_name, orig, ClassList)
    state.prev_bboxes = copy.deepcopy(state.bboxes)