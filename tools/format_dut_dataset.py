import os
import json
import cv2
import shutil
import random  # <--- NEW: Imported the random module

def parse_sot_bounding_box(line_data: str) -> list:
    """Parses SOT bounding box format into a list of floats."""
    normalized_line = line_data.strip().replace(',', ' ')
    coordinate_parts = normalized_line.split()
    if len(coordinate_parts) < 4:
        return None
    return [float(coordinate_parts[0]), float(coordinate_parts[1]), float(coordinate_parts[2]), float(coordinate_parts[3])]

def reformat_and_generate_pipeline(dataset_root: str, source_img_dir_name: str, source_gt_dir_name: str, target_dir_name: str = "test", num_sequences: int = 5):
    """
    Step 1: Reformats a random selection of sequences into the strict MOT format.
    """
    source_img_dir = os.path.join(dataset_root, source_img_dir_name)
    source_gt_dir = os.path.join(dataset_root, source_gt_dir_name)
    target_dir = os.path.join(dataset_root, target_dir_name)

    print(f"🚀 [Step 1] Formatting {num_sequences} RANDOM sequences for evaluation...")
    
    # Clean up the old 'test' folder to avoid mixing old and new sequences
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Grab all valid sequence directories
    all_sequences = sorted([d for d in os.listdir(source_img_dir) if os.path.isdir(os.path.join(source_img_dir, d))])
    
    # --- NEW: Randomly select the sequences ---
    # We use min() just in case num_sequences is larger than the total available sequences
    safe_num_sequences = min(num_sequences, len(all_sequences))
    mini_sequences = sorted(random.sample(all_sequences, safe_num_sequences))
    
    print(f"🎲 Randomly selected sequences: {mini_sequences}")
    # ------------------------------------------

    for seq_name in mini_sequences:
        seq_path = os.path.join(source_img_dir, seq_name)
            
        target_seq_dir = os.path.join(target_dir, seq_name)
        target_img_dir = os.path.join(target_seq_dir, "img1")
        target_gt_dir = os.path.join(target_seq_dir, "gt")
        
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_gt_dir, exist_ok=True)
        
        # Format images to 6 digits (.jpg)
        for img_name in os.listdir(seq_path):
            if img_name.lower().endswith(('.jpg', '.png')):
                frame_num = int(img_name.split('.')[0])
                new_img_name = f"{frame_num:06d}.jpg"
                shutil.copy2(os.path.join(seq_path, img_name), os.path.join(target_img_dir, new_img_name))
                
        # Handle Ground Truth Translation
        gt_source_file_1 = os.path.join(source_gt_dir, f"{seq_name}_gt.txt")
        gt_source_file_2 = os.path.join(source_gt_dir, f"{seq_name}.txt")
        
        gt_source_actual = None
        if os.path.exists(gt_source_file_1):
            gt_source_actual = gt_source_file_1
        elif os.path.exists(gt_source_file_2):
            gt_source_actual = gt_source_file_2
        
        gt_target_file = os.path.join(target_gt_dir, "gt.txt")
        
        if gt_source_actual:
            with open(gt_source_actual, 'r') as f_in, open(gt_target_file, 'w') as f_out:
                for frame_idx, line in enumerate(f_in.readlines()):
                    bbox = parse_sot_bounding_box(line)
                    if bbox and bbox[2] > 0 and bbox[3] > 0:
                        # Convert to strict MOT format
                        f_out.write(f"{frame_idx + 1},1,{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,1,1\n")
        else:
            print(f"⚠️ WARNING: Ground Truth file not found for {seq_name}")

    print("✅ [Step 1] Physical reformatting and GT translation complete.")
    return target_dir, mini_sequences

def generate_dut_coco_format(dataset_root: str, formatted_test_dir: str, output_json_path: str, valid_sequences: list):
    """
    Step 2: Generates the COCO JSON for the selected sequences.
    """
    print(f"🚀 [Step 2] Generating COCO JSON for the selected sequences...")
    
    # Delete the old JSON file to prevent data pollution
    if os.path.exists(output_json_path):
        print(f"🧹 Deleting old JSON file at '{output_json_path}'...")
        os.remove(output_json_path)
    
    coco_format = {'images': [], 'annotations': [], 'videos': [], 'categories': [{'id': 1, 'name': 'UAV'}]}
    global_image_id, global_annotation_id = 0, 0
    
    for video_id, sequence_name in enumerate(valid_sequences, start=1):
        coco_format['videos'].append({'id': video_id, 'file_name': sequence_name})
        
        sequence_images_dir = os.path.join(formatted_test_dir, sequence_name, "img1")
        gt_txt_path = os.path.join(formatted_test_dir, sequence_name, "gt", "gt.txt")
        
        image_files = sorted([f for f in os.listdir(sequence_images_dir) if f.lower().endswith(('.jpg', '.png'))])
        num_images = len(image_files)
        
        ground_truth_lines = []
        if os.path.exists(gt_txt_path):
            with open(gt_txt_path, 'r') as gt_file:
                ground_truth_lines = gt_file.readlines()
            
        for frame_index, image_name in enumerate(image_files):
            global_image_id += 1
            image_path = os.path.join(sequence_images_dir, image_name)
            
            frame_image = cv2.imread(image_path)
            if frame_image is None: continue
            height, width = frame_image.shape[:2]
            
            coco_format['images'].append({
                'file_name': f"{sequence_name}/img1/{image_name}",
                'id': global_image_id,
                'frame_id': frame_index + 1,
                'prev_image_id': global_image_id - 1 if frame_index > 0 else -1,
                'next_image_id': global_image_id + 1 if frame_index < num_images - 1 else -1,
                'video_id': video_id, 'height': height, 'width': width
            })
            
            if frame_index < len(ground_truth_lines):
                parts = ground_truth_lines[frame_index].strip().split(',')
                if len(parts) >= 6:
                    bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                    if bbox[2] > 0 and bbox[3] > 0:
                        global_annotation_id += 1
                        coco_format['annotations'].append({
                            'id': global_annotation_id, 'category_id': 1, 'image_id': global_image_id,
                            'track_id': 1, 'bbox': bbox, 'conf': 1.0, 'iscrowd': 0, 'area': bbox[2] * bbox[3]
                        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file)
    print("✅ [Step 2] COCO JSON successfully generated.")

if __name__ == '__main__':
    DATASET_ROOT = "dataset/DUT Anti-UAV"
    
    # 1. Reformat the files and get the list of the 5 active sequences
    formatted_dir, active_sequences = reformat_and_generate_pipeline(
        dataset_root=DATASET_ROOT,
        source_img_dir_name="Anti-UAV-Tracking-V0",
        source_gt_dir_name="Anti-UAV-Tracking-V0GT",
        target_dir_name="test",
        num_sequences=7
    )
    
    # 2. Build the JSON using ONLY those randomized sequences
    generate_dut_coco_format(
        dataset_root=DATASET_ROOT,
        formatted_test_dir=formatted_dir,
        output_json_path=os.path.join(DATASET_ROOT, "annotations", "test.json"),
        valid_sequences=active_sequences
    )