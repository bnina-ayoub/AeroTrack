import os
import numpy as np
import json
import cv2

# --- PARAMÈTRES À AJUSTER ---
DATA_PATH = 'dataset/UAVSwarm'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
# Tu peux ajouter 'train_half' et 'val_half' si tu souhaites diviser tes vidéos d'entraînement
SPLITS = ['train', 'test'] 

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, split)
        
        # Si le dossier n'existe pas (ex: pas de dossier test), on l'ignore
        if not os.path.exists(data_path):
            print(f"Le dossier {data_path} n'existe pas, passage au suivant.")
            continue
            
        out_path = os.path.join(OUT_PATH, f'{split}.json')
        out = {
            'images': [], 
            'annotations': [], 
            'videos': [],
            'categories': [{'id': 1, 'name': 'UAV'}] # Catégorie adaptée à ton projet
        }
        
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
                
            video_cnt += 1
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'img1')
            ann_path = os.path.join(seq_path, 'gt/gt.txt')
            
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            
            # Paramétrage de la plage d'images (utile si tu utilises train_half/val_half)
            image_range = [0, num_images - 1]

            # 1. Enregistrement des images
            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                
                img_file = os.path.join(data_path, f'{seq}/img1/{i + 1:06d}.jpg')
                img = cv2.imread(img_file)
                if img is None:
                    continue
                    
                height, width = img.shape[:2]
                image_info = {
                    'file_name': f'{seq}/img1/{i + 1:06d}.jpg',
                    'id': image_cnt + i + 1,
                    'frame_id': i + 1 - image_range[0],
                    'prev_image_id': image_cnt + i if i > 0 else -1,
                    'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                    'video_id': video_cnt,
                    'height': height, 
                    'width': width
                }
                out['images'].append(image_info)
                
            print(f'{seq}: {num_images} images ajoutées')
            
            # 2. Enregistrement des annotations (si le fichier gt.txt existe)
            if os.path.exists(ann_path):
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                
                print(f'{seq}: Traitement des annotations...')
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                        
                    track_id = int(anns[i][1])
                    ann_cnt += 1
                    
                    # Gestion des IDs de tracking
                    if not track_id == tid_last:
                        tid_curr += 1
                        tid_last = track_id
                        
                    ann = {
                        'id': ann_cnt,
                        'category_id': 1, # Toujours 1 pour UAV
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]) if anns.shape[1] > 6 else 1.0,
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])
                    }
                    out['annotations'].append(ann)
            else:
                print(f'{seq}: Aucun fichier gt.txt trouvé (normal pour le set de test).')
                
            image_cnt += num_images
            
        print(f'Chargement de {split} terminé : {len(out["images"])} images et {len(out["annotations"])} annotations.')
        
        # Sauvegarde du fichier JSON
        with open(out_path, 'w') as f:
            json.dump(out, f)