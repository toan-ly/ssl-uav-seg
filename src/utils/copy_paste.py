import json
import cv2
import numpy as np
import base64
import zlib
import os
from pathlib import Path
from collections import defaultdict
import random

class CAPAugmentation:
    def __init__(self, dataset_path, output_path, instances_per_class=5):
        """
        Cross-Image Copy-Paste Augmentation for Supervisely format
        
        Args:
            dataset_path: Path to dataset root (contains img and ann folders)
            output_path: Where to save augmented images/annotations
            instances_per_class: How many instances to extract per class
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.instances_per_class = instances_per_class
        self.img_dir = os.path.join(dataset_path, 'img')
        self.ann_dir = os.path.join(dataset_path, 'ann')
        self.object_pool = defaultdict(list)  # {class_name: [(img, mask, bbox), ...]}
        
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'img'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'ann'), exist_ok=True)
    
    def decode_bitmap(self, bitmap_data):
        """Decode Supervisely bitmap mask (cropped region)"""
        data = base64.b64decode(bitmap_data)
        decoded = zlib.decompress(data)
        # Data is packed bits, unpack them
        mask = np.unpackbits(np.frombuffer(decoded, dtype=np.uint8))
        return mask
    
    def encode_bitmap(self, mask):
        """Encode mask to Supervisely bitmap format"""
        compressed = zlib.compress(mask.tobytes())
        return base64.b64encode(compressed).decode('utf-8')
    
    def get_bbox_from_mask(self, mask):
        """Extract bounding box from binary mask"""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return [x_min, y_min, x_max, y_max]
    
    def extract_object_from_mask(self, img, mask, bbox):
        """Extract object region using mask"""
        x_min, y_min, x_max, y_max = bbox
        obj_img = img[y_min:y_max+1, x_min:x_max+1].copy()
        obj_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
        return obj_img, obj_mask
    
    def build_object_pool(self):
        """
        Extract and store object instances from all images
        Pool structure: {class_name: [(img, mask, origin_bbox), ...]}
        """
        ann_files = sorted(os.listdir(self.ann_dir))
        
        for ann_file in ann_files:
            if not ann_file.endswith('.json'):
                continue
            
            # Handle xxx.png.json format
            img_name = ann_file.replace('.json', '')  # Remove .json to get xxx.png
            ann_path = os.path.join(self.ann_dir, ann_file)
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            with open(ann_path) as f:
                ann_data = json.load(f)
            
            # Get image dimensions from annotation
            img_h = ann_data.get('size', {}).get('height', img.shape[0])
            img_w = ann_data.get('size', {}).get('width', img.shape[1])
            
            # Extract each object
            for obj in ann_data.get('objects', []):
                if obj.get('geometryType') != 'bitmap':
                    continue
                
                class_name = obj.get('classTitle', 'unknown')
                bitmap = obj.get('bitmap', {})
                origin = bitmap.get('origin', [0, 0])  # [x, y] position
                
                try:
                    # Decode mask (this is the cropped region)
                    mask_bits = self.decode_bitmap(bitmap['data'])
                    
                    # Extract region from image to get actual dimensions
                    x_orig, y_orig = origin
                    region = img[y_orig:, x_orig:].copy()  # Crop from origin
                    region_h, region_w = region.shape[:2]
                    
                    # Reshape mask to region dimensions
                    try:
                        mask = mask_bits[:region_h * region_w].reshape((region_h, region_w))
                    except ValueError:
                        # Try different dimension order
                        mask = mask_bits[:region_h * region_w].reshape((region_w, region_h)).T
                    
                    # Get bbox within the cropped region
                    bbox_in_region = self.get_bbox_from_mask(mask)
                    if bbox_in_region is None:
                        continue
                    
                    # Extract object from region
                    obj_img, obj_mask = self.extract_object_from_mask(region, mask, bbox_in_region)
                    
                    # Store in pool
                    self.object_pool[class_name].append({
                        'img': obj_img,
                        'mask': obj_mask,
                        'bbox': bbox_in_region,
                        'origin': origin
                    })
                except Exception as e:
                    print(f"Error processing {ann_file} object {obj.get('id')}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Limit to N instances per class
        for class_name in self.object_pool:
            if len(self.object_pool[class_name]) > self.instances_per_class:
                self.object_pool[class_name] = random.sample(
                    self.object_pool[class_name], 
                    self.instances_per_class
                )
        
        print(f"Object pool created:")
        for class_name, instances in self.object_pool.items():
            print(f"  {class_name}: {len(instances)} instances")
    
    def paste_object_on_image(self, img, obj_img, obj_mask, pos_x, pos_y):
        """Paste object onto image at given position"""
        h, w = obj_img.shape[:2]
        img_h, img_w = img.shape[:2]
        
        # Check bounds
        if pos_x + w > img_w or pos_y + h > img_h or pos_x < 0 or pos_y < 0:
            return False
        
        # Create mask for blending
        mask_3ch = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR)
        mask_3ch = mask_3ch.astype(float) / 255.0
        
        # Blend using alpha blending
        region = img[pos_y:pos_y+h, pos_x:pos_x+w]
        blended = (obj_img.astype(float) * mask_3ch + 
                   region.astype(float) * (1 - mask_3ch)).astype(np.uint8)
        
        img[pos_y:pos_y+h, pos_x:pos_x+w] = blended
        
        return True
    
    def update_annotations(self, ann_data, obj_mask, pos_x, pos_y, class_name):
        """Add new object to annotation"""
        new_obj = {
            'id': max([obj.get('id', 0) for obj in ann_data.get('objects', [])], default=0) + 1,
            'classId': 49320,  # Adjust as needed
            'description': '',
            'geometryType': 'bitmap',
            'classTitle': class_name,
            'bitmap': {
                'data': self.encode_bitmap(obj_mask),
                'origin': [pos_x, pos_y]
            },
            'tags': []
        }
        ann_data['objects'].append(new_obj)
    
    def augment_image(self, img_name, num_pastes=2):
        """
        Augment single image by pasting random objects
        
        Args:
            img_name: Image filename (e.g., xxx.png)
            num_pastes: Number of objects to paste
        """
        img_path = os.path.join(self.img_dir, img_name)
        ann_path = os.path.join(self.ann_dir, img_name + '.json')
        
        if not os.path.exists(img_path) or not os.path.exists(ann_path):
            return False
        
        img = cv2.imread(img_path)
        with open(ann_path) as f:
            ann_data = json.load(f)
        
        img_h, img_w = img.shape[:2]
        pasted = 0
        
        # Try to paste N objects
        for _ in range(num_pastes * 5):  # Retry if placement fails
            if pasted >= num_pastes:
                break
            
            # Random class from pool
            class_name = random.choice(list(self.object_pool.keys()))
            obj_data = random.choice(self.object_pool[class_name])
            
            obj_img = obj_data['img']
            obj_mask = obj_data['mask']
            obj_h, obj_w = obj_img.shape[:2]
            
            # Random position (ensure it fits)
            pos_x = random.randint(0, max(0, img_w - obj_w))
            pos_y = random.randint(0, max(0, img_h - obj_h))
            
            # Try to paste
            if self.paste_object_on_image(img, obj_img, obj_mask, pos_x, pos_y):
                self.update_annotations(ann_data, obj_mask, pos_x, pos_y, class_name)
                pasted += 1
        
        return pasted > 0
    
    def augment_dataset(self, augment_ratio=0.5, num_pastes=2):
        """
        Augment dataset
        
        Args:
            augment_ratio: Fraction of images to augment
            num_pastes: Number of objects to paste per image
        """
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])
        num_to_augment = int(len(img_files) * augment_ratio)
        selected = random.sample(img_files, num_to_augment)
        
        for idx, img_name in enumerate(selected):
            print(f"Augmenting {idx+1}/{num_to_augment}: {img_name}")
            
            img_path = os.path.join(self.img_dir, img_name)
            ann_path = os.path.join(self.ann_dir, img_name + '.json')
            img = cv2.imread(img_path)
            
            with open(ann_path) as f:
                ann_data = json.load(f)
            
            # Augment
            if self.augment_image(img_name, num_pastes):
                # Save augmented image
                img_base = os.path.splitext(img_name)[0]
                out_img_path = os.path.join(self.output_path, 'img', 
                                           f'aug_{img_name}')
                out_ann_path = os.path.join(self.output_path, 'ann', 
                                           f'aug_{img_name}.json')
                
                cv2.imwrite(out_img_path, img)
                with open(out_ann_path, 'w') as f:
                    json.dump(ann_data, f, indent=2)

# Usage
if __name__ == '__main__':
    '`p'
    dataset_path=Path('E:\FALL2025\DEEP LEARNING\Project\uav-project\data\train'),
    output_path='./dataset_augmented',
    instances_per_class=5
    cap = CAPAugmentation(
        
    )
    
    # Build object pool from all images
    cap.build_object_pool()
    
    # Augment dataset
    cap.augment_dataset(augment_ratio=0.5, num_pastes=2)