import os
import json
from PIL import Image

def convert_visdrone_to_coco(visdrone_dir, out_json_path, is_train=True):
    print(f"Converting VisDrone in {visdrone_dir} to COCO format...")
    img_dir = os.path.join(visdrone_dir, 'images')
    ann_dir = os.path.join(visdrone_dir, 'annotations')
    
    # 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    categories = [
        {"id": 1, "name": "pedestrian", "supercategory": "person"},
        {"id": 2, "name": "people", "supercategory": "person"},
        {"id": 4, "name": "car", "supercategory": "vehicle"},
        {"id": 5, "name": "van", "supercategory": "vehicle"},
        {"id": 6, "name": "truck", "supercategory": "vehicle"},
        {"id": 9, "name": "bus", "supercategory": "vehicle"}
    ]
    valid_cat_ids = {c["id"] for c in categories}
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    ann_id = 1
    
    for img_id, img_file in enumerate(img_files, 1):
        if img_id % 1000 == 0:
            print(f"Processed {img_id}/{len(img_files)} images...")
            
        img_path = os.path.join(img_dir, img_file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {img_file}: {e}")
            continue
            
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        
        txt_file = img_file.replace('.jpg', '.txt')
        txt_path = os.path.join(ann_dir, txt_file)
        
        if not os.path.exists(txt_path):
            continue
            
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                    
                bbox_x, bbox_y, bbox_w, bbox_h, score, category = map(int, parts[:6])
                
                # Only keep valid categories (car, person variants)
                if category not in valid_cat_ids:
                    continue
                    
                # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category,
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
                
    with open(out_json_path, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"Done! Saved to {out_json_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")

if __name__ == "__main__":
    train_dir = "/work/nthujerry123/sam3/datasets/uav/VisDrone2019-DET-train"
    val_dir = "/work/nthujerry123/sam3/datasets/uav/VisDrone2019-DET-val"
    
    os.makedirs("/work/nthujerry123/sam3/datasets/visdrone/annotations", exist_ok=True)
    
    if os.path.exists(train_dir):
        convert_visdrone_to_coco(train_dir, "/work/nthujerry123/sam3/datasets/visdrone/annotations/train_coco.json")
    if os.path.exists(val_dir):
        convert_visdrone_to_coco(val_dir, "/work/nthujerry123/sam3/datasets/visdrone/annotations/val_coco.json")
