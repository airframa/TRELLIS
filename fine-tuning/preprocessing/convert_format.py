#!/usr/bin/env python3
"""
3DRealCar to TRELLIS format conversion.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import trimesh
from PIL import Image
import torch

# Add TRELLIS root to path
trellis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, trellis_root)
from dataset_toolkits.utils import get_file_hash

class RealCar3DToTrellisConverter:
    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create TRELLIS directory structure
        (self.output_dir / "renders").mkdir(exist_ok=True)
        (self.output_dir / "voxels").mkdir(exist_ok=True)
        
        # Initialize SAM for background removal
        self.setup_car_detector()
        
    # Add this to clean_converter.py after saving transforms.json:
    def verify_conversion(self, renders_dir, sha256):
        """Verify the conversion matches expected format"""
        transforms_path = renders_dir / "transforms.json"
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Check a few cameras
        for i in range(min(5, len(transforms['frames']))):
            frame = transforms['frames'][i]
            c2w = np.array(frame['transform_matrix'])
            cam_pos = c2w[:3, 3]
            distance = np.linalg.norm(cam_pos)
            print(f"  Camera {i}: distance = {distance:.3f}")
        
        # Check mesh bounds
        mesh = trimesh.load(str(renders_dir / "mesh.ply"))
        print(f"  Mesh bounds: {mesh.bounds}")
        print(f"  Mesh centered: {np.abs(mesh.vertices.mean(axis=0)).max() < 0.1}")
    
    def process_single_car(self, car_dir):
        """Process a single car directory from 3DRealCar format to TRELLIS format"""
        print(f"Processing: {car_dir.name}")
        
        # Find mesh file
        mesh_candidates = [
            car_dir / "textured_output.obj",
            car_dir / "export_refined.obj", 
            car_dir / "export.obj",
        ]
        
        mesh_file = None
        for candidate in mesh_candidates:
            if candidate.exists():
                mesh_file = candidate
                break
        
        if not mesh_file:
            print(f"No mesh file found in {car_dir}")
            return None
        
        # Compute SHA256 hash
        # sha256 = get_file_hash(str(mesh_file))

        # Compute unique SHA256 hash
        # Include sample directory name to handle duplicate mesh files
        import hashlib
        unique_id = f"{str(mesh_file)}_{car_dir.name}"
        sha256 = hashlib.sha256(unique_id.encode('utf-8')).hexdigest()

        # Load the transform from info.json
        info_path = car_dir / "info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Extract the 4x4 matrix
        t = info['transformToWorldMap']
        transform_world = np.array([
            [t['m11'], t['m12'], t['m13'], t['m14']],
            [t['m21'], t['m22'], t['m23'], t['m24']],
            [t['m31'], t['m32'], t['m33'], t['m34']],
            [t['m41'], t['m42'], t['m43'], t['m44']]
        ])
        
        # Load mesh and transform
        mesh = trimesh.load(str(mesh_file), process=False)
        mesh.apply_transform(transform_world)
        
        # CRITICAL FIX: Normalize based on mesh bounds only
        mesh_bounds = mesh.bounds
        center = (mesh_bounds[1] + mesh_bounds[0]) / 2.0
        extent = mesh_bounds[1] - mesh_bounds[0]
        scale_factor = 1.0 / np.max(extent)  # Fit in unit cube
        
        # Apply normalization to mesh
        mesh.vertices = (mesh.vertices - center) * scale_factor
        
        # Convert to Blender coordinates
        vertices = mesh.vertices.copy()
        mesh.vertices[:, 1] = -vertices[:, 2]  # Z -> -Y
        mesh.vertices[:, 2] = vertices[:, 1]   # Y -> Z
        
        # Save normalized mesh
        # Verify mesh is within bounds BEFORE saving
        mesh_bounds_final = mesh.bounds
        if not np.all(np.abs(mesh_bounds_final) <= 0.5001):  # Small tolerance
            print(f"Warning: Mesh slightly outside unit cube: {mesh_bounds_final}")
            # Force clamp if needed
            mesh.vertices = np.clip(mesh.vertices, -0.5, 0.5)
        
        # Save normalized mesh
        renders_dir = self.output_dir / "renders" / sha256
        renders_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(str(renders_dir / "mesh.ply"))
        
        # First pass: collect camera positions to understand their distribution
        camera_positions = []
        json_files = sorted(car_dir.glob("frame_*.json"))
        for json_file in json_files[:150]:  # Sample first 10 for analysis
            with open(json_file, 'r') as f:
                cam_params = json.load(f)
            if 'cameraPoseARFrame' in cam_params:
                c2w = np.array(cam_params['cameraPoseARFrame']).reshape(4, 4)
                c2w_aligned = np.linalg.inv(transform_world) @ c2w
                cam_pos = (c2w_aligned[:3, 3] - center) * scale_factor
                # Convert to Blender coords
                cam_pos_blender = np.array([cam_pos[0], -cam_pos[2], cam_pos[1]])
                camera_positions.append(cam_pos_blender)
        
        # Calculate camera rescaling factor
        cam_positions_np = np.array(camera_positions)
        avg_cam_distance = np.mean(np.linalg.norm(cam_positions_np, axis=1))
        camera_scale_factor = 2.0 / avg_cam_distance  # Target distance of 2.0
        
        print(f"Original avg camera distance: {avg_cam_distance:.3f}")
        print(f"Camera rescale factor: {camera_scale_factor:.3f}")
        
        # Process camera data with rescaling
        transforms_data = self.process_camera_data(
            car_dir, renders_dir, scale_factor, center, transform_world, camera_scale_factor
        )
        
        # Save transforms.json with TRELLIS-compatible parameters
        transforms_data.update({
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            # Note: scale should reflect the combined effect
            "scale": float(scale_factor / camera_scale_factor),
            "offset": center.tolist()
        })
        
        transforms_path = renders_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms_data, f, indent=2)

        self.verify_conversion(renders_dir, sha256)
        
        # Create metadata entry
        metadata = {
            'sha256': sha256,
            'aesthetic_score': self.calculate_aesthetic_score(renders_dir),
            'rendered': True,
            'voxelized': False,
            'num_voxels': 0,
            'local_path': str(mesh_file.relative_to(self.source_dir)),
            'source_dataset': '3DRealCar',
            'original_id': car_dir.name,
            'captions': '["3D car model from 3DRealCar dataset"]',  
        }
        
        return metadata
    
    def process_camera_data(self, car_dir, renders_dir, scale_factor, center, transform_world, camera_scale_factor):
        """Process camera data and convert to TRELLIS format"""
        transforms_data = {"frames": []}
        
        # Find all frame JSON files
        json_files = sorted(car_dir.glob("frame_*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    camera_params = json.load(f)
                
                frame_num = json_file.stem.split('_')[-1]
                
                # Find corresponding image
                image_path = self.find_image_file(car_dir, frame_num)
                if not image_path:
                    continue
                
                # Process camera parameters - pass camera_scale_factor here
                frame_data = self.process_camera_frame(
                    camera_params, image_path, frame_num, 
                    renders_dir, scale_factor, center, transform_world, camera_scale_factor
                )
                
                if frame_data:
                    transforms_data["frames"].append(frame_data)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
        return transforms_data
    
    def process_camera_frame(self, camera_params, image_path, frame_num, renders_dir, 
                        scale_factor, center, transform_world, camera_scale_factor):
        if 'cameraPoseARFrame' not in camera_params:
            return None
            
        # Get AR camera pose
        c2w_ar = np.array(camera_params['cameraPoseARFrame']).reshape(4, 4)
        c2w_aligned = np.linalg.inv(transform_world) @ c2w_ar
        
        # Apply normalization
        cam_pos = c2w_aligned[:3, 3]
        cam_pos_normalized = (cam_pos - center) * scale_factor
        
        # Convert to Blender coordinates
        cam_pos_blender = np.array([
            cam_pos_normalized[0],
            -cam_pos_normalized[2],
            cam_pos_normalized[1]
        ])
        
        # CRITICAL: Rescale camera position to standard distance
        cam_pos_blender = cam_pos_blender * camera_scale_factor
        
        # Verify distance
        cam_distance = np.linalg.norm(cam_pos_blender)
        if cam_distance < 1.5 or cam_distance > 2.5:
            print(f"Warning: Camera {frame_num} at unusual distance {cam_distance:.2f}")
        
        # Build rotation matrix (looking at origin)
        forward = -cam_pos_blender / np.linalg.norm(cam_pos_blender)
        up = np.array([0, 0, 1])
        right = np.cross(up, forward)
        if np.linalg.norm(right) < 0.001:
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        actual_up = np.cross(forward, right)
        
        # Build c2w matrix
        c2w_blender = np.eye(4)
        c2w_blender[:3, 0] = right
        c2w_blender[:3, 1] = actual_up
        c2w_blender[:3, 2] = -forward
        c2w_blender[:3, 3] = cam_pos_blender
        
        frame_idx = int(frame_num)
        
        # Process image
        output_image_path = renders_dir / f"frame_{frame_idx:05d}.png"
        self.convert_and_resize_image(image_path, output_image_path, target_size=512)
        
        # Calculate FOV for the RESIZED image and RESCALED camera
        if 'intrinsics' in camera_params and len(camera_params['intrinsics']) == 9:
            K_orig = np.array(camera_params['intrinsics']).reshape(3, 3)
            fx_orig = K_orig[0, 0]
            
            # Get original image dimensions
            orig_img = Image.open(image_path)
            orig_width, orig_height = orig_img.size
            
            # Original FOV
            fov_orig = 2 * np.arctan(orig_width / (2 * fx_orig))
            
            # Adjust FOV for camera rescaling
            # When camera moves farther, FOV decreases to maintain view
            fov_adjusted = 2 * np.arctan(np.tan(fov_orig/2) / camera_scale_factor)
        else:
            fov_adjusted = 0.6981317007977318  # Default 40 degrees
        
        return {
            "file_path": f"./frame_{frame_idx:05d}.png",
            "transform_matrix": c2w_blender.tolist(),
            "camera_angle_x": fov_adjusted
        }

    def setup_car_detector(self):
        """Initialize SAM with YOLO for car detection"""
        try:
            # Use YOLO to detect car bounding box first
            from ultralytics import YOLO
            
            # Get paths to local models
            third_party_dir = os.path.join(trellis_root, 'third_party')
            yolo_path = os.path.join(third_party_dir, 'yolov8x.pt')
            sam_checkpoint = os.path.join(third_party_dir, "sam_vit_h_4b8939.pth")
            
            # Load YOLO
            if os.path.exists(yolo_path):
                print(f"Loading YOLO from {yolo_path}")
                self.detector = YOLO(model=yolo_path, verbose=True)
            else:
                print(f"YOLO not found, downloading...")
                self.detector = YOLO('yolov8x.pt')
            
            # Load SAM
            from segment_anything import sam_model_registry, SamPredictor
            import torch
            
            if not os.path.exists(sam_checkpoint):
                raise FileNotFoundError(
                    f"SAM model not found at {sam_checkpoint}\n"
                    f"Please download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                )
            
            print(f"Loading SAM from {sam_checkpoint}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device=device)
            
            self.sam_predictor = SamPredictor(sam)
            print("SAM and YOLO loaded successfully")
        except ImportError:
            print("Installing dependencies...")
            os.system("pip install ultralytics")
            os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
            self.setup_car_detector()

    def segment_car(self, image_path):
        """Detect car with YOLO, then segment precisely with SAM"""
        import cv2
        import numpy as np
        
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # First, detect cars with YOLO
        results = self.detector(str(image_path))
        
        car_boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # Class 2 is 'car' in COCO dataset
                    if class_id == 2 and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        center_x = (x1 + x2) / 2 / w
                        center_y = (y1 + y2) / 2 / h
                        
                        car_boxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'area': area,
                            'center_x': center_x,
                            'center_y': center_y,
                            'confidence': confidence
                        })
        
        if not car_boxes:
            print(f"No cars detected in {image_path}")
            return None
        
        # Select the best car (largest and most central)
        best_car = max(car_boxes, key=lambda x: x['area'] * (1 - abs(x['center_x'] - 0.5)) * (1 - abs(x['center_y'] - 0.5)))
        
        # Use SAM to segment within the detected car bbox
        self.sam_predictor.set_image(image_rgb)
        
        # Get box coordinates
        input_box = np.array(best_car['bbox'])
        
        # Generate mask using SAM with the box prompt
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        # Select best mask (highest score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        return best_mask

    def convert_and_resize_image(self, input_path, output_path, target_size=512):
        """Convert image with transparent background like TRELLIS"""
        mask = self.segment_car(input_path)
        
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)
        
        # Create RGBA image with transparent background
        img_rgba = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)
        
        if mask is not None:
            # Apply mask - copy RGB where mask is True, set alpha channel
            img_rgba[:, :, :3] = img_np
            img_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
            
            # Find bounding box of mask
            y_coords, x_coords = np.where(mask)
            if len(x_coords) > 0:
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                # Add padding (10% of dimensions)
                width = x_max - x_min
                height = y_max - y_min
                padding_x = int(width * 0.1)
                padding_y = int(height * 0.1)
                
                x_min = max(0, x_min - padding_x)
                x_max = min(img_np.shape[1], x_max + padding_x)
                y_min = max(0, y_min - padding_y)
                y_max = min(img_np.shape[0], y_max + padding_y)
                
                # Crop to car region
                img_rgba = img_rgba[y_min:y_max, x_min:x_max]
        else:
            # No mask - keep full image opaque
            print(f"Warning: No car mask found for {input_path}, keeping full image")
            img_rgba[:, :, :3] = img_np
            img_rgba[:, :, 3] = 255
        
        img_pil = Image.fromarray(img_rgba, 'RGBA')
        
        # Make square with transparent padding
        width, height = img_pil.size
        max_dim = max(width, height)
        
        # Create square image with fully transparent background
        square_img = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        square_img.paste(img_pil, (paste_x, paste_y))
        
        # Resize to target size
        final_img = square_img.resize((target_size, target_size), Image.LANCZOS)
        final_img.save(output_path, 'PNG')

    
    def calculate_fov(self, camera_params):
        """Calculate field of view from camera intrinsics"""
        if 'intrinsics' in camera_params and len(camera_params['intrinsics']) == 9:
            K = np.array(camera_params['intrinsics']).reshape(3, 3)
            fx = K[0, 0]
            image_width = K[0, 2] * 2  # Principal point gives half-width
            return 2 * np.arctan(image_width / (2 * fx))
        
        return 0.6981317007977318  # Default 40 degrees in radians
    
    def find_image_file(self, car_dir, frame_num):
        """Find the image file corresponding to a frame"""
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = car_dir / f"frame_{frame_num}{ext}"
            if candidate.exists():
                return candidate
        return None
    
    def convert_image(self, input_path, output_path):
        """Convert image to TRELLIS format (RGBA PNG)"""
        img = Image.open(input_path).convert("RGB")
        img_rgba = Image.new("RGBA", img.size)
        img_rgba.paste(img, (0, 0))
        img_rgba.putalpha(255)  # Fully opaque
        img_rgba.save(output_path)
    
    def calculate_aesthetic_score(self, renders_dir):
        """Calculate aesthetic score (placeholder)"""
        return 7.5
    
    def convert_all_cars(self):
        """Convert all cars in the source directory"""
        all_metadata = []
        
        # Check if source_dir is a single car or directory of cars
        if any(self.source_dir.glob("frame_*.json")):
            # Single car
            metadata = self.process_single_car(self.source_dir)
            if metadata:
                all_metadata.append(metadata)
        else:
            # Directory of cars
            car_dirs = [d for d in self.source_dir.iterdir() 
                       if d.is_dir() and any(d.glob("frame_*.json"))]
            
            for car_dir in tqdm(car_dirs, desc="Converting cars"):
                try:
                    metadata = self.process_single_car(car_dir)
                    if metadata:
                        all_metadata.append(metadata)
                except Exception as e:
                    print(f"Error processing {car_dir}: {e}")
                    continue
        
        # Save metadata
        if all_metadata:
            df = pd.DataFrame(all_metadata)
            df.to_csv(self.output_dir / "metadata.csv", index=False)

            # Create instances.txt for voxelization
            instances_file = self.output_dir / "instances.txt"
            with open(instances_file, 'w') as f:
                for metadata in all_metadata:
                    f.write(f"{metadata['sha256']}\n")
            print(f"Created instances file: {instances_file}")

            print(f"Successfully converted {len(all_metadata)} cars")
        
        return all_metadata

def main():
    parser = argparse.ArgumentParser(description="Convert 3DRealCar data to TRELLIS format")
    parser.add_argument("--source_dir", required=True, help="Path to 3DRealCar source directory")
    parser.add_argument("--output_dir", required=True, help="Path to output TRELLIS directory")
    
    args = parser.parse_args()
    
    converter = RealCar3DToTrellisConverter(args.source_dir, args.output_dir)
    converter.convert_all_cars()

if __name__ == "__main__":
    main()