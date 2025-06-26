# suppress TensorFlow INFO and WARNING logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERRO
# imports
import argparse
import json
import random
import numpy as np
import pandas as pd
import cv2
import trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import open_clip
import timm
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
from numba import njit
import subprocess
# Add the TRELLIS root directory to sys.path
# Go up two levels from preprocessing/ to fine-tuning/ to TRELLIS root
trellis_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.abspath(trellis_root))
from dataset_toolkits.utils import get_file_hash
from trellis.datasets import SparseStructure, SparseFeat2Render, SLat2Render

# Numba-optimized version of project_dino_features_to_voxels
@njit
def compute_patch_centers_numba(grid_w, grid_h, patch_size):
    """Compute patch center coordinates efficiently"""
    num_patches = grid_w * grid_h
    px = np.empty(num_patches, dtype=np.float32)
    py = np.empty(num_patches, dtype=np.float32)
    
    idx = 0
    for j in range(grid_h):
        for i in range(grid_w):
            px[idx] = (i + 0.5) * patch_size
            py[idx] = (j + 0.5) * patch_size
            idx += 1
    
    return px, py

@njit
def pixels_to_rays_numba(px, py, fx, fy, cx, cy):
    """Convert pixel coordinates to normalized ray directions in camera frame"""
    num_rays = len(px)
    rays_cam = np.empty((num_rays, 3), dtype=np.float32)
    
    for i in range(num_rays):
        x_cam = (px[i] - cx) / fx
        y_cam = (py[i] - cy) / fy
        z_cam = 1.0
        
        # Normalize
        norm = np.sqrt(x_cam*x_cam + y_cam*y_cam + z_cam*z_cam)
        rays_cam[i, 0] = x_cam / norm
        rays_cam[i, 1] = y_cam / norm
        rays_cam[i, 2] = z_cam / norm
    
    return rays_cam

@njit
def transform_rays_to_world_numba(rays_cam, R, T):
    """Transform rays from camera space to world space"""
    num_rays = rays_cam.shape[0]
    rays_world = np.empty((num_rays, 3), dtype=np.float32)
    origin_world = T.astype(np.float32)
    
    for i in range(num_rays):
        # rays_world[i] = R @ rays_cam[i]
        for j in range(3):
            rays_world[i, j] = (R[j, 0] * rays_cam[i, 0] + 
                               R[j, 1] * rays_cam[i, 1] + 
                               R[j, 2] * rays_cam[i, 2])
    
    return rays_world, origin_world

@njit
def ray_aabb_intersection_batch_numba(rays_world, origin_world, voxel_min, voxel_max, resolution):
    """Batch ray-AABB intersection with early termination"""
    num_rays = rays_world.shape[0]
    valid_coords = np.empty((num_rays, 3), dtype=np.int32)
    valid_mask = np.empty(num_rays, dtype=np.bool_)
    valid_count = 0
    
    for i in range(num_rays):
        ray_dir = rays_world[i]
        
        # Ray-AABB intersection
        # Handle division by zero by adding small epsilon
        eps = 1e-8
        t_min_x = (voxel_min[0] - origin_world[0]) / (ray_dir[0] + eps if abs(ray_dir[0]) > eps else eps)
        t_max_x = (voxel_max[0] - origin_world[0]) / (ray_dir[0] + eps if abs(ray_dir[0]) > eps else eps)
        
        t_min_y = (voxel_min[1] - origin_world[1]) / (ray_dir[1] + eps if abs(ray_dir[1]) > eps else eps)
        t_max_y = (voxel_max[1] - origin_world[1]) / (ray_dir[1] + eps if abs(ray_dir[1]) > eps else eps)
        
        t_min_z = (voxel_min[2] - origin_world[2]) / (ray_dir[2] + eps if abs(ray_dir[2]) > eps else eps)
        t_max_z = (voxel_max[2] - origin_world[2]) / (ray_dir[2] + eps if abs(ray_dir[2]) > eps else eps)
        
        # Ensure t_min < t_max for each axis
        if t_min_x > t_max_x:
            t_min_x, t_max_x = t_max_x, t_min_x
        if t_min_y > t_max_y:
            t_min_y, t_max_y = t_max_y, t_min_y
        if t_min_z > t_max_z:
            t_min_z, t_max_z = t_max_z, t_min_z
        
        # Find overall intersection
        t_enter = max(max(t_min_x, t_min_y), max(t_min_z, 0.0))
        t_exit = min(min(t_max_x, t_max_y), t_max_z)
        
        if t_enter < t_exit and t_exit > 0:
            # Ray intersects the voxel volume
            t_sample = (t_enter + t_exit) * 0.5
            
            # Compute 3D point
            point_3d_x = origin_world[0] + ray_dir[0] * t_sample
            point_3d_y = origin_world[1] + ray_dir[1] * t_sample
            point_3d_z = origin_world[2] + ray_dir[2] * t_sample
            
            # Convert to voxel coordinates
            voxel_coord_x = int((point_3d_x - voxel_min[0]) / (voxel_max[0] - voxel_min[0]) * resolution)
            voxel_coord_y = int((point_3d_y - voxel_min[1]) / (voxel_max[1] - voxel_min[1]) * resolution)
            voxel_coord_z = int((point_3d_z - voxel_min[2]) / (voxel_max[2] - voxel_min[2]) * resolution)
            
            # Check bounds
            if (voxel_coord_x >= 0 and voxel_coord_x < resolution and
                voxel_coord_y >= 0 and voxel_coord_y < resolution and
                voxel_coord_z >= 0 and voxel_coord_z < resolution):
                
                valid_coords[valid_count, 0] = voxel_coord_x
                valid_coords[valid_count, 1] = voxel_coord_y
                valid_coords[valid_count, 2] = voxel_coord_z
                valid_mask[i] = True
                valid_count += 1
            else:
                valid_mask[i] = False
        else:
            valid_mask[i] = False
    
    return valid_coords[:valid_count], valid_mask, valid_count


class RealCar3DProcessor:
    def __init__(self, source_dir, output_dir, sample_views=150, run_verifications=False):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sample_views = sample_views
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_verifications = run_verifications 

        # Create root-level directories as expected by TRELLIS
        (self.output_dir / "renders").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "voxels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(parents=True, exist_ok=True)

    def convert_to_trellis(self):
        """Convert all cars in source directory to TRELLIS format"""
        source_path = Path(self.source_dir)
        all_metadata = []
        
        # Check if source_dir itself contains car data
        if any(source_path.glob("frame_*.json")):
            # Single car case
            try:
                metadata = self.execute_conversion(source_path)
                all_metadata.append(metadata)
            except Exception as e:
                print(f"Error processing {source_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            # Multiple cars case
            car_dirs = []
            for d in source_path.iterdir():
                if d.is_dir() and any(d.glob("frame_*.json")):
                    car_dirs.append(d)
            
            if car_dirs:
                print(f"Found {len(car_dirs)} cars to process")
                for car_dir in tqdm(car_dirs, desc="Converting to TRELLIS format", leave=False):
                    try:
                        metadata = self.execute_conversion(car_dir)
                        all_metadata.append(metadata)
                    except Exception as e:
                        print(f"Error processing {car_dir.name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        return all_metadata

    def execute_conversion(self, car_dir):
        """Process a single car from 3DRealCar to TRELLIS format"""
        car_dir = Path(car_dir)
        original_car_id = car_dir.name

        # Try different possible mesh locations
        mesh_candidates = [
            car_dir / "textured_output.obj",
            car_dir / "export_refined.obj", 
            car_dir / "export.obj",
            car_dir / "colmap_processed" / "meshed-delaunay.ply"
        ]

        mesh_file = None
        mesh = None
        for mesh_path in mesh_candidates:
            if mesh_path.exists():
                # print(f"Loading mesh from: {mesh_path}")
                mesh = trimesh.load(str(mesh_path), process=False)
                mesh_file = mesh_path
                break

        if mesh is None or mesh_file is None:
            raise FileNotFoundError(f"No mesh file found in {car_dir}")

        # Compute SHA256 hash of the mesh file - TRELLIS standard
        sha256 = get_file_hash(str(mesh_file))

        # Create directories using SHA256 as identifier (TRELLIS standard)
        renders_dir = self.output_dir / "renders" / sha256
        renders_dir.mkdir(parents=True, exist_ok=True)

        # Normalize mesh to unit cube centered at origin
        # Store the transformation parameters
        original_bounds = mesh.bounds.copy()
        original_center = (original_bounds[1] + original_bounds[0]) / 2.0
        max_extent = np.max(original_bounds[1] - original_bounds[0])

        # Center the mesh
        mesh.vertices -= original_center
        # Scale uniformly
        mesh.vertices /= max_extent

        # Save the mesh in the expected location
        mesh_output_path = renders_dir / "mesh.ply"
        mesh.export(str(mesh_output_path))

        # Load camera parameters and find corresponding images
        frames_data = self.load_frames_data(car_dir)

        if not frames_data:
            raise ValueError(f"No frame data found in {car_dir}")

        # Sample views uniformly
        sampled_indices = self.sample_uniform_views(len(frames_data), target=min(self.sample_views, len(frames_data)))

        # Path to your output JSON file
        transforms_path = renders_dir / "transforms.json"

        # Load existing transforms if available
        if transforms_path.exists():
            with open(transforms_path, "r") as f:
                transforms = json.load(f)
        else:
            transforms = {"frames": []}

        # Store camera parameters for sampled frames
        sampled_camera_params = {}

        # Process each sampled view
        for idx, frame_idx in enumerate(sampled_indices):
            frame_data = frames_data[frame_idx]

            # Define output image path
            output_image_name = f"frame_{idx:05d}.png"
            output_image_path = renders_dir / output_image_name

            # # Skip if already exists
            if output_image_path.exists():
                 continue

            # Use OpenCV to load .jpg quickly
            img_bgr = cv2.imread(str(frame_data['image_path']))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Convert to PIL and add alpha
            img_pil = Image.fromarray(img_rgb)
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')

            # Save as .png
            output_image_name = f"frame_{idx:05d}.png"
            output_image_path = renders_dir / output_image_name
            img_pil.save(output_image_path)

            # Convert camera parameters to TRELLIS format
            cam_trellis, K, c2w = self.convert_camera_params(
                frame_data['camera_params'], 
                max_extent,  # mesh_scale
                original_center,  # mesh_center
                frame_data['frame_num']
            )
            
            # Add frame to transforms with intrinsics
            frame_info = {
                "file_path": f"./{output_image_name}",
                "transform_matrix": cam_trellis['transform_matrix'],
                "camera_angle_x": cam_trellis['camera_angle_x']
            }
            
            # Add original intrinsics if available
            if K is not None:
                frame_info["intrinsics"] = K.tolist()
            
            transforms["frames"].append(frame_info)

            # Store camera parameters for DINOv2 processing
            sampled_camera_params[idx] = {
                'K': K,
                'c2w': c2w,
                'original_frame_idx': frame_idx
            }

        # Save transforms.json
        with open(renders_dir / "transforms.json", 'w') as f:
            json.dump(transforms, f, indent=2)

        # Create voxel representation
        voxel_info = self.create_voxel_representation(mesh, resolution=64)

        # Ensure voxel coordinates are within bounds (0-63)
        assert np.all(voxel_info >= 0) and np.all(voxel_info < 64), \
            f"Voxel coordinates out of bounds: min={voxel_info.min()}, max={voxel_info.max()}"

        # Save voxel data as PLY in the root voxels directory using SHA256
        voxel_path = self.output_dir / "voxels" / f"{sha256}.ply"

        # Create PLY file with vertex positions
        with open(voxel_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(voxel_info)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for coord in voxel_info:
                # Convert from voxel grid coordinates to normalized range [-0.5, 0.5]
                x, y, z = (coord.astype(float) / 64.0) - 0.5
                f.write(f"{x} {y} {z}\n")

        # Calculate aesthetic score
        aesthetic_score = self.calculate_aesthetic_score(mesh, renders_dir)

        # Create metadata CSV entry - Full TRELLIS compatibility
        metadata = {
            'sha256': sha256,
            'aesthetic_score': aesthetic_score,
            'rendered': True,
            'voxelized': True, 
            'num_voxels': len(voxel_info),
            'num_views': len(transforms["frames"]),
            'feature_dinov2_vitl14_reg': False,  # Changed to False
            'latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16': False,  # Added
            'cond_rendered': False,
            'captions': None,
            'local_path': str(mesh_file.relative_to(self.source_dir)),
            # Custom fields for reference
            'source_dataset': '3DRealCar',
            'original_id': original_car_id,
        }
        
        return metadata

    def convert_camera_params(self, cam_3drealcar, mesh_scale, mesh_center, frame_num):
        """Convert 3DRealCar camera format to TRELLIS format"""
        camera_trellis = {}
        K = None
        c2w = None

        # Default camera_angle_x if not found
        camera_trellis['camera_angle_x'] = 1.0  # Default FOV

        # Extract intrinsics
        if 'intrinsics' in cam_3drealcar:
            intrinsics_flat = cam_3drealcar['intrinsics']
            if len(intrinsics_flat) == 9:
                K = np.array(intrinsics_flat).reshape(3, 3)
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]

                # Ensure fx is not zero
                if fx > 0:
                    # Assuming image size from intrinsics (principal point ~= image_size/2)
                    image_width = cx * 2
                    image_height = cy * 2

                    # Convert to FOV
                    fov_x = 2 * np.arctan(image_width / (2 * fx))
                    camera_trellis['camera_angle_x'] = float(fov_x)
                else:
                    print(f"Warning: Invalid focal length fx={fx}, using default FOV")
        else:
            print("Warning: No intrinsics found in camera parameters, using default FOV")

        # Extract camera pose
        if 'cameraPoseARFrame' in cam_3drealcar:
            pose_flat = cam_3drealcar['cameraPoseARFrame']
            if len(pose_flat) == 16:
                c2w_original = np.array(pose_flat).reshape(4, 4)
                
                # Replace the 180° rotation with proper coordinate system conversion
                # Instead of rotating, convert from CV to CG coordinate system:
                conversion_mat = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],  # Flip Y axis
                    [0, 0, -1, 0],   # Flip Z axis
                    [0, 0, 0, 1]
                ])
                c2w = c2w_original @ conversion_mat
                
                # Then apply normalization as before:
                c2w[:3, 3] -= mesh_center
                c2w[:3, 3] /= mesh_scale
                
                camera_trellis['transform_matrix'] = c2w.tolist()
            else:
                print(f"Warning: Invalid cameraPoseARFrame length: {len(pose_flat)}")
                # Default identity matrix
                camera_trellis['transform_matrix'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        else:
            print("Warning: No cameraPoseARFrame found, using identity matrix")
            camera_trellis['transform_matrix'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        return camera_trellis, K, c2w
    
    def project_dino_features_to_voxels_optimized(self, *, features, camera_intrinsics, camera_extrinsics, 
                                            image_size, resolution=64, return_debug=False, mesh_bounds=None):
        """
        Optimized version using Numba for faster ray-voxel projection.
        """
        # Convert to numpy for Numba compatibility
        K = camera_intrinsics.cpu().numpy().astype(np.float32)
        c2w_input = camera_extrinsics.cpu().numpy().astype(np.float32)
        c2w = c2w_input
        
        # 1. Determine patch grid shape and compute centers
        N = features.shape[0]
        patch_size = 14
        grid_w = image_size // patch_size  # 37
        grid_h = image_size // patch_size  # 37
        
        # Handle token count mismatch
        expected_tokens = grid_h * grid_w
        if features.shape[0] != expected_tokens:
            features = features[:expected_tokens]  # Truncate extra tokens
        
        # 2. Compute patch centers using Numba
        px, py = compute_patch_centers_numba(grid_w, grid_h, patch_size)
        
        # 3. Convert pixels to rays using Numba
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        rays_cam = pixels_to_rays_numba(px, py, fx, fy, cx, cy)
        
        # 4. Transform rays to world space using Numba
        R = c2w[:3, :3].astype(np.float32)
        T = c2w[:3, 3].astype(np.float32)
        rays_world, origin_world = transform_rays_to_world_numba(rays_cam, R, T)
        
        # 5. Set up voxel bounds
        if mesh_bounds is not None:
            voxel_min = mesh_bounds[0].astype(np.float32)
            voxel_max = mesh_bounds[1].astype(np.float32)
        else:
            voxel_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
            voxel_max = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        # 6. Perform batch ray-AABB intersection using Numba
        valid_coords, valid_mask, valid_count = ray_aabb_intersection_batch_numba(
            rays_world, origin_world, voxel_min, voxel_max, resolution
        )
        
        # 7. Extract corresponding features
        if valid_count > 0:
            coords = valid_coords
            feats = features[valid_mask[:len(features)]]
        else:
            coords = np.array([]).reshape(0, 3).astype(np.int32)
            feats = np.array([]).reshape(0, features.shape[1])
        
        if return_debug:
            debug_info = {
                'out_of_bounds': N - valid_count,
                'grid_shape': (grid_h, grid_w),
                'patch_size': patch_size,
                'camera_position': T,
                'total_patches': N,
                'valid_patches': valid_count,
                'mesh_bounds': (voxel_min, voxel_max)
            }
            return coords, feats, debug_info

        return coords, feats

    def extract_dino_features_all(self):
        """Extract DINOv2 features for all cars in the dataset efficiently"""
        print("🔍 Extracting DINOv2 features for all cars...")
        
        metadata_path = self.output_dir / "metadata.csv"
        if not metadata_path.exists():
            print("No metadata.csv found. Run execute_conversion first.")
            return
            
        df = pd.read_csv(metadata_path)
        
        # Load DINOv2 model only once
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DINOv2 model on {device}...")
        
        model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True)
        model.eval().to(device)
        
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        # Process each car
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting DINOv2 features", leave=False):
            sha256 = row['sha256']
            
            # Check if features already exist
            feat_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
            feat_file = feat_dir / f"{sha256}.npz"
            if feat_file.exists():
                print(f"DINOv2 features already exist. Skipping extraction.")
                continue
                
            try:
                self._extract_features_for_car(
                    sha256=sha256,
                    model=model,
                    transform=transform,
                    device=device
                )
            except Exception as e:
                print(f"❌ Error processing {sha256}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("✅ DINOv2 feature extraction complete!")

    def _extract_features_for_car(self, sha256, model, transform, device):
        """Extract features for a single car"""
        renders_dir = self.output_dir / "renders" / sha256
        
        # Load mesh
        mesh_path = renders_dir / "mesh.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        mesh = trimesh.load(str(mesh_path), process=False)
        mesh_bounds = mesh.bounds
        
        # Load transforms.json to get camera info
        transforms_path = renders_dir / "transforms.json"
        if not transforms_path.exists():
            raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
            
        with open(transforms_path, 'r') as f:
            transforms_data = json.load(f)
        
        # Get all frame files
        frame_files = sorted(renders_dir.glob("frame_*.png"))
        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {renders_dir}")
        
        # Create frame path to transforms mapping
        frame_to_transform = {}
        for i, frame_info in enumerate(transforms_data['frames']):
            filename = Path(frame_info['file_path']).name
            frame_to_transform[filename] = {
                'transform_matrix': np.array(frame_info['transform_matrix']),
                'camera_angle_x': frame_info.get('camera_angle_x', 1.0),
                'intrinsics': np.array(frame_info['intrinsics']) if 'intrinsics' in frame_info else None,
                'index': i
            }
        
        # Sample frames for feature extraction
        num_frames_to_sample = min(100, len(frame_files))
        if len(frame_files) > num_frames_to_sample:
            sampled_frames = random.sample(frame_files, num_frames_to_sample)
        else:
            sampled_frames = frame_files
        
        # Extract features
        patchtokens_all = []
        indices_all = []
        resolution = 64
        
        # Verification tracking
        voxel_hit_count = np.zeros((resolution, resolution, resolution), dtype=int)
        total_valid_projections = 0
        total_out_of_bounds = 0
        
        for frame_path in sampled_frames:
            
            frame_name = frame_path.name
            
            if frame_name not in frame_to_transform:
                continue
                
            # Get camera info
            cam_info = frame_to_transform[frame_name]
            c2w = cam_info['transform_matrix']
            
            # Use original intrinsics if available, otherwise reconstruct from FOV
            if cam_info['intrinsics'] is not None:
                K_original = cam_info['intrinsics']
                
                # CRITICAL: Create adjusted intrinsics for 518x518 image
                original_width = K_original[0, 2] * 2
                original_height = K_original[1, 2] * 2
                image_size = 518
                
                scale_x = image_size / original_width
                scale_y = image_size / original_height
                
                K = K_original.copy()
                K[0, 0] *= scale_x  # fx
                K[1, 1] *= scale_y  # fy
                K[0, 2] = image_size / 2  # cx
                K[1, 2] = image_size / 2  # cy
            else:
                # Fallback: reconstruct from FOV
                camera_angle_x = cam_info['camera_angle_x']
                image_size = 518
                fx = image_size / (2 * np.tan(camera_angle_x / 2))
                fy = fx  # Assume square pixels
                cx = cy = image_size / 2
                
                K = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            
            # Load and process image
            image = Image.open(frame_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = model.forward_features(tensor)
                features = features.squeeze(0).cpu().numpy()
            
            # Project features to voxels with debug info
            coords, feats, debug_info = self.project_dino_features_to_voxels_optimized(
                features=features,
                camera_intrinsics=torch.tensor(K),
                camera_extrinsics=torch.tensor(c2w),
                image_size=518,
                resolution=resolution,
                mesh_bounds=mesh_bounds,
                return_debug=True
            )
            
            # Update verification tracking
            total_valid_projections += len(coords)
            total_out_of_bounds += debug_info['out_of_bounds']
            
            for coord in coords:
                voxel_hit_count[coord[0], coord[1], coord[2]] += 1
            
            if len(coords) > 0:
                patchtokens_all.append(feats)
                indices_all.append(coords)
        
        # Save features if we have any
        if patchtokens_all and indices_all:
            patchtokens = np.concatenate(patchtokens_all, axis=0)
            indices = np.concatenate(indices_all, axis=0)
            
            # Save features
            feat_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
            feat_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                feat_dir / f"{sha256}.npz",
                patchtokens=patchtokens.astype(np.float32),
                indices=indices.astype(np.int32)
            )
            
            # print(f"✓ Saved {len(indices)} features for {sha256[:8]}...")
        
            # Quick verification calls
            if self._run_verifications: 

                # Print verification report
                print(f"\n📊 Ray-casting Verification Report for {sha256[:8]}:")
                # Use first 10 frames for consistency with original report
                num_frames_for_report = min(10, len(sampled_frames))
                total_patches = num_frames_for_report * 1374  # Correct number: 1374 patches
                print(f"Total patches processed: {total_patches}")
                print(f"Valid projections: {total_valid_projections}")
                print(f"Out of bounds projections: {total_out_of_bounds}")

                if (total_valid_projections + total_out_of_bounds) > 0:
                    success_rate = total_valid_projections / (total_valid_projections + total_out_of_bounds) * 100
                    print(f"Projection success rate: {success_rate:.1f}%")
                else:
                    print("Projection success rate: N/A (no projections)")

                unique_voxels_hit = np.sum(voxel_hit_count > 0)
                print(f"Unique voxels with features: {unique_voxels_hit}")
                print(f"Voxel coverage: {unique_voxels_hit / (resolution**3) * 100:.3f}%")  # Show 3 decimal places

                # Check spatial distribution
                x_coverage = np.any(voxel_hit_count > 0, axis=(1, 2))
                y_coverage = np.any(voxel_hit_count > 0, axis=(0, 2))
                z_coverage = np.any(voxel_hit_count > 0, axis=(0, 1))

                print(f"X-axis coverage: {np.sum(x_coverage)}/{resolution} slices")
                print(f"Y-axis coverage: {np.sum(y_coverage)}/{resolution} slices")
                print(f"Z-axis coverage: {np.sum(z_coverage)}/{resolution} slices")
                
                # Optionally save visualization
                if unique_voxels_hit > 0:
                    self._save_coverage_visualization(voxel_hit_count, renders_dir / "dino_coverage_debug.png")

                # Reconstruct sampled_camera_params for verification
                sampled_camera_params = {}
                for i, (fname, cam_info) in enumerate(frame_to_transform.items()):
                    if cam_info['intrinsics'] is not None:
                        sampled_camera_params[i] = {
                            'K': cam_info['intrinsics'],
                            'c2w': cam_info['transform_matrix'],
                            'original_frame_idx': i
                        }
                
                print("\n🔍 Running additional verifications...")
                self.verify_raycasting_reprojection(renders_dir, sampled_camera_params, mesh)
                self.verify_mesh_and_voxels(mesh, sha256, self.output_dir)
        else:
            print(f"⚠️ No valid features extracted for {sha256}")

    def verify_mesh_and_voxels(self, mesh, sha256, output_dir):
        """Verify mesh normalization and voxel alignment"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: Mesh vertices
        ax1 = fig.add_subplot(131, projection='3d')
        vertices = mesh.vertices[::100]  # Sample every 100th vertex for visibility
        ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.5, c='blue')
        ax1.set_xlim(-0.6, 0.6)
        ax1.set_ylim(-0.6, 0.6)
        ax1.set_zlim(-0.6, 0.6)
        ax1.set_title(f'Mesh Vertices (sampled)\nBounds: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[1][0]:.2f}]')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Voxel representation
        ax2 = fig.add_subplot(132, projection='3d')
        voxel_path = output_dir / "voxels" / f"{sha256}.ply"
        if voxel_path.exists():
            # Read the PLY file manually
            voxel_coords = []
            with open(voxel_path, 'r') as f:
                lines = f.readlines()
                # Find where header ends
                header_end = 0
                for i, line in enumerate(lines):
                    if line.strip() == "end_header":
                        header_end = i + 1
                        break
                # Read vertex data
                for line in lines[header_end:]:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        voxel_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            
            if voxel_coords:
                voxel_coords = np.array(voxel_coords)
                ax2.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], 
                        s=1, alpha=0.5, c='green')
                ax2.set_title(f'Voxel Representation\n{len(voxel_coords)} voxels')
        else:
            ax2.set_title('Voxel Representation\n(not yet created)')
        
        ax2.set_xlim(-0.6, 0.6)
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_zlim(-0.6, 0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot 3: Feature voxel locations
        ax3 = fig.add_subplot(133, projection='3d')
        feat_path = output_dir / "features" / "dinov2_vitl14_reg" / f"{sha256}.npz"
        if feat_path.exists():
            data = np.load(feat_path)
            indices = data['indices']
            if len(indices) > 0:
                # Convert voxel indices to 3D coordinates
                voxel_centers = (indices.astype(float) / 64.0) - 0.5
                ax3.scatter(voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2], 
                        s=1, alpha=0.5, c='red')
                ax3.set_title(f'DINO Feature Voxels\n{len(indices)} features')
            else:
                ax3.set_title('DINO Feature Voxels\n(no features)')
        else:
            ax3.set_title('DINO Feature Voxels\n(not yet created)')
        
        ax3.set_xlim(-0.6, 0.6)
        ax3.set_ylim(-0.6, 0.6)
        ax3.set_zlim(-0.6, 0.6)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        debug_path = output_dir / "renders" / sha256 / "mesh_voxel_alignment_debug.png"
        plt.savefig(debug_path, dpi=150)
        plt.close()
        print(f"💾 Saved mesh-voxel alignment debug to {debug_path}")

    def _save_coverage_visualization(self, voxel_hit_count, output_path):
        """Save a visualization of the voxel coverage"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY projection (top view)
        xy_projection = np.max(voxel_hit_count, axis=2)
        im1 = axes[0].imshow(xy_projection.T, cmap='hot', origin='lower')
        axes[0].set_title('XY Projection (Top View)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0])
        
        # XZ projection (side view)
        xz_projection = np.max(voxel_hit_count, axis=1)
        im2 = axes[1].imshow(xz_projection.T, cmap='hot', origin='lower')
        axes[1].set_title('XZ Projection (Side View)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[1])
        
        # YZ projection (front view)
        yz_projection = np.max(voxel_hit_count, axis=0)
        im3 = axes[2].imshow(yz_projection.T, cmap='hot', origin='lower')
        axes[2].set_title('YZ Projection (Front View)')
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"💾 Saved coverage visualization to {output_path}")

    def verify_raycasting_reprojection(self, renders_dir, sampled_camera_params, mesh):
        """Verify that voxels projected from camera 0 can be reprojected back correctly"""
        print("\n🔍 Running reprojection verification...")
        
        if 0 not in sampled_camera_params:
            print("No camera 0 found for reprojection test")
            return
        
        K_original = sampled_camera_params[0]['K']
        c2w = sampled_camera_params[0]['c2w']
        
        # CRITICAL: Adjust intrinsics for 518x518 image
        original_width = K_original[0, 2] * 2
        original_height = K_original[1, 2] * 2
        image_size = 518
        
        scale_x = image_size / original_width
        scale_y = image_size / original_height
        
        K = K_original.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] = image_size / 2  # cx
        K[1, 2] = image_size / 2  # cy
        
        print(f"\nOriginal intrinsics for {original_width:.0f}x{original_height:.0f} image:")
        print(f"fx={K_original[0,0]:.1f}, fy={K_original[1,1]:.1f}, cx={K_original[0,2]:.1f}, cy={K_original[1,2]:.1f}")
        
        print(f"\nAdjusted intrinsics for {image_size}x{image_size} image:")
        print(f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        
        # Test with mesh center first
        # Add coordinate system conversion to test point
        conversion_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        test_point = conversion_mat @ np.array([0, 0, 0, 1])  # Apply CV->CG conversion
        w2c = np.linalg.inv(c2w)
        
        # Project test point
        p_cam = w2c @ test_point
        print(f"\nTest point [0,0,0] in camera coords: {p_cam[:3]}")
        
        if p_cam[2] <= 0:
            print("ERROR: Mesh center is behind camera! Camera orientation is wrong.")
            return
        
        # Project to image with adjusted intrinsics
        p_img = K @ p_cam[:3]
        p_img = p_img[:2] / p_img[2]
        print(f"Test point projects to image: ({p_img[0]:.1f}, {p_img[1]:.1f})")
        print(f"Image size: 518x518")
        print(f"Is in bounds? {0 <= p_img[0] < 518 and 0 <= p_img[1] < 518}")
        
        # Now test the full reprojection pipeline
        print("\nTesting reprojection of camera 0 features...")
        
        # Create dummy features for camera 0
        N = 1374  # DINOv2 patches for 518x518 image
        dummy_features = np.zeros((N, 1024), dtype=np.float32)
        
        # Project features from camera 0 to get voxel coordinates
        coords_cam0, _, debug_info = self.project_dino_features_to_voxels(
            features=dummy_features,
            camera_intrinsics=torch.tensor(K),
            camera_extrinsics=torch.tensor(c2w),
            image_size=518,
            resolution=64,
            return_debug=True,
            mesh_bounds=mesh.bounds
        )
        
        if len(coords_cam0) == 0:
            print("No valid projections from camera 0")
            return
        
        print(f"Camera 0 projected to {len(coords_cam0)} voxels")
        
        # Now reproject these voxels back to camera 0
        resolution = 64
        mesh_bounds = mesh.bounds
        
        # Convert voxel indices to world coordinates
        voxel_centers = coords_cam0.astype(np.float32) / resolution  # [0, 1]
        voxel_centers = mesh_bounds[0] + voxel_centers * (mesh_bounds[1] - mesh_bounds[0])
        
        # Project to camera space
        points_homo = np.hstack([voxel_centers, np.ones((len(voxel_centers), 1))])
        points_cam = (w2c @ points_homo.T).T[:, :3]
        
        # All points should be in front of camera
        z_positive = points_cam[:, 2] > 0
        print(f"Points in front of camera: {np.sum(z_positive)}/{len(points_cam)}")
        
        if not np.all(z_positive):
            print("WARNING: Some reprojected points are behind camera!")
        
        # Project to image plane (only for points in front)
        points_cam_valid = points_cam[z_positive]
        points_img = (K @ points_cam_valid.T).T
        points_img = points_img[:, :2] / points_img[:, 2:3]
        
        # Check bounds
        h, w = 518, 518
        in_bounds = (points_img[:, 0] >= 0) & (points_img[:, 0] < w) & \
                    (points_img[:, 1] >= 0) & (points_img[:, 1] < h)
        
        success_rate = np.sum(in_bounds) / len(coords_cam0) * 100
        
        print(f"\n✓ Reprojection success: {np.sum(in_bounds)}/{len(coords_cam0)} ({success_rate:.1f}%)")
        
        if success_rate < 80:
            print("⚠️ WARNING: Low reprojection success rate indicates potential issues")
        
        # Also save the simple reprojection visualization
        self._visualize_reprojection(renders_dir, points_img[in_bounds], "reprojection_check.png")

    def _visualize_reprojection(self, renders_dir, points_2d, filename):
        # Load first frame (original resolution)
        first_frame = renders_dir / "frame_00000.png"
        if not first_frame.exists():
            return
            
        img = Image.open(first_frame)
        orig_width, orig_height = img.size
        scale_x = orig_width / 518
        scale_y = orig_height / 518
        
        # Scale points to original image size
        scaled_points = points_2d.copy()
        scaled_points[:, 0] *= scale_x
        scaled_points[:, 1] *= scale_y
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.scatter(scaled_points[:, 0], scaled_points[:, 1], c='red', s=1, alpha=0.5)
        plt.title('Reprojected Voxel Centers on Frame 0')
        plt.axis('off')
        
        output_path = renders_dir / filename
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"💾 Saved reprojection visualization to {output_path}")

    def load_frames_data(self, car_dir):
        """Load all frame data (images and camera parameters) in parallel"""
        frames_data = []

        json_files = sorted(car_dir.glob("frame_*.json"))

        # Possible image locations
        image_dirs = [
            car_dir,
            car_dir / "colmap_processed" / "images",
            car_dir / "images",
        ]

        def process_json(json_file):
            frame_num = json_file.stem.split('_')[-1]

            # Find corresponding image
            image_path = None
            for img_dir in image_dirs:
                if not img_dir.exists():
                    continue
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = img_dir / f"frame_{frame_num}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break
                if image_path:
                    break

            if not image_path:
                return None

            try:
                with open(json_file, 'r') as f:
                    camera_params = json.load(f)

                return {
                    'image_path': image_path,
                    'camera_params': camera_params,
                    'frame_num': frame_num
                }
            except Exception as e:
                print(f"⚠️ Error reading {json_file.name}: {e}")
                return None

        # Parallel loading
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_json, json_files))

        # Filter out failed loads
        frames_data = [r for r in results if r is not None]

        return frames_data

    def calculate_aesthetic_score(self, mesh, renders_dir):
        """
        Calculate aesthetic score using LAION aesthetic predictor.
        Randomly samples frames and averages their scores.
        """
        try:
            # Check if we have the required model files
            weights_path = "third_party/improved_aesthetic_predictor/sac+logos+ava1-l14-linearMSE.pth"
            if not Path(weights_path).exists():
                print(f"Warning: LAION weights not found at {weights_path}, using fallback method")

            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Model definition (matching the working architecture)
            class AestheticPredictor(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(768, 1024),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(1024, 128),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(128, 64),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(64, 16),
                        torch.nn.Linear(16, 1)
                    )

                def forward(self, x):
                    return self.layers(x)

            # Load CLIP model (only once, cache if possible)
            if not hasattr(self, '_clip_model'):
                # print("Loading CLIP model for aesthetic scoring...")
                self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                    model_name="ViT-L-14",
                    pretrained="laion2b_s32b_b82k"
                )
                self._clip_model = self._clip_model.to(device)
                self._clip_model.eval()

                # Load aesthetic regressor
                self._regressor = AestheticPredictor().to(device)
                checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
                self._regressor.load_state_dict(checkpoint)
                self._regressor.eval()
                # print("LAION aesthetic predictor loaded successfully")

            # Find all frame files
            frame_pattern = str(renders_dir / "frame_*.png")
            all_frames = glob.glob(frame_pattern)

            if len(all_frames) == 0:
                print(f"Warning: No frame files found in {renders_dir}")
                return self._fallback_aesthetic_score(mesh, renders_dir)

            # Sample frames (max 10 for efficiency, min 3 for reliability)
            num_samples = min(10, max(3, len(all_frames)))
            if len(all_frames) > num_samples:
                selected_frames = random.sample(all_frames, num_samples)
            else:
                selected_frames = all_frames

            # Process frames and collect scores
            scores = []
            for frame_path in selected_frames:
                try:
                    # Load and preprocess image
                    image = Image.open(frame_path).convert("RGB")
                    image_tensor = self._preprocess(image).unsqueeze(0).to(device)

                    # Get CLIP embedding
                    with torch.no_grad():
                        embedding = self._clip_model.encode_image(image_tensor).float()

                    # Normalize embedding
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                    # Predict aesthetic score
                    with torch.no_grad():
                        score = self._regressor(embedding).item()

                    scores.append(score)

                except Exception as e:
                    print(f"Warning: Failed to process frame {frame_path}: {e}")
                    continue

            if not scores:
                print("Warning: No frames could be processed, using fallback method")
                return self._fallback_aesthetic_score(mesh, renders_dir)

            # Calculate average score
            avg_score = sum(scores) / len(scores)

            # LAION scores typically range from 1-10, but can go higher
            # Clamp to reasonable range for consistency
            final_score = max(1.0, min(10.0, avg_score))

            # print(f"LAION aesthetic score: {final_score:.3f} (from {len(scores)} frames, "
            #     f"range: {min(scores):.3f}-{max(scores):.3f})")

            return final_score

        except ImportError as e:
            print(f"Warning: Required libraries not available ({e}), using fallback method")
            return self._fallback_aesthetic_score(mesh, renders_dir)
        except Exception as e:
            print(f"Warning: Error in LAION aesthetic scoring ({e}), using fallback method")
            return self._fallback_aesthetic_score(mesh, renders_dir)

    def _fallback_aesthetic_score(self, mesh, renders_dir):
        """
        Fallback method for aesthetic scoring when LAION predictor is not available.
        Uses simple heuristics based on mesh quality.
        """
        score = 7.5  # Base score

        # Adjust based on mesh quality
        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            # More vertices might indicate higher detail
            vertex_score = min(2.0, len(mesh.vertices) / 50000)
            score += vertex_score

        # Check if mesh is watertight
        if hasattr(mesh, 'is_watertight') and mesh.is_watertight:
            score += 0.5

        # Clamp to reasonable range
        score = max(1.0, min(10.0, score))

        print(f"Using fallback aesthetic score: {score:.3f}")
        return score

    def create_voxel_representation(self, mesh, resolution=64):
        """Create a simple voxel representation of the mesh"""
        # Get mesh bounds
        bounds = mesh.bounds

        # Create voxel grid
        voxel_size = (bounds[1] - bounds[0]).max() / resolution

        # Get vertex positions normalized to voxel grid
        voxel_coords = ((mesh.vertices - bounds[0]) / voxel_size).astype(np.int32)

        # Ensure coordinates are within bounds [0, resolution-1]
        voxel_coords = np.clip(voxel_coords, 0, resolution - 1)

        # Remove duplicates
        unique_voxels = np.unique(voxel_coords, axis=0)

        return unique_voxels

    def sample_uniform_views(self, total_views, target=150):
        """Sample views uniformly from available views"""
        if total_views <= target:
            return list(range(total_views))

        # Uniform sampling
        indices = np.linspace(0, total_views-1, target, dtype=int)
        return indices.tolist()
    
    def run_encode_latents(self, force=False):
        # Path to the expected output directory for latents
        latent_dir = (
            self.output_dir / "latents" / "dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
        )

        print(str(self.output_dir))

        # General check: if any .npz file exists, assume latents are done
        if latent_dir.exists() and any(latent_dir.glob("*.npz")) and not force:
            print("Latent features already exist. Skipping encoding.")
            print("✅ Latent features encoding complete!")
            return
        
        cmd = [
            "python", "dataset_toolkits/encode_latent.py",
            "--output_dir", str(self.output_dir),
            "--feat_model", "dinov2_vitl14_reg",
            "--enc_pretrained", "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"
        ]

        # Optional: print the command for debug
        print("🔍 Encoding latent features for all cars")
        # print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Optional: show output or handle errors
        if result.returncode != 0:
            print("Error running encode_latents:")
            print(result.stderr)
        else:
            print("encode_latents completed successfully.")
            print(result.stdout)

        print("✅ Latent features encoding complete!")

    def update_metadata(self, update_type, check_dir=None, column_name=None, print_summary=False):
        """
        General metadata update function.
        
        Args:
            update_type: Type of update ('features' or 'latents')
            check_dir: Directory to check for files (optional)
            column_name: Specific column to update (optional)
        """
        def _log(msg):
            if print_summary:
                _log(msg)

        metadata_path = self.output_dir / "metadata.csv"
        
        if not metadata_path.exists():
            print(f"Warning: metadata.csv not found at {metadata_path}")
            return
            
        df = pd.read_csv(metadata_path)
        
        if update_type == 'features':
            # Update DINOv2 features
            if check_dir is None:
                check_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
            if column_name is None:
                column_name = 'feature_dinov2_vitl14_reg'
                
            _log(f"Updating {column_name} in metadata...")
            updated_count = 0
            
            for idx, row in df.iterrows():
                sha256 = row['sha256']
                feat_file = check_dir / f"{sha256}.npz"
                if feat_file.exists():
                    df.loc[idx, column_name] = True
                    updated_count += 1
                else:
                    df.loc[idx, column_name] = False
                    
            _log(f"Updated {updated_count} entries with {column_name}=True")
            
        elif update_type == 'latents':
            # Update latent encodings from CSV files
            latent_csv_files = list(self.output_dir.glob("latent_*.csv"))
            _log(f"Found {len(latent_csv_files)} latent CSV files")
            
            for latent_csv in latent_csv_files:
                _log(f"Processing: {latent_csv.name}")
                latent_col_name = latent_csv.stem.rsplit('_', 1)[0]
                
                df_latent = pd.read_csv(latent_csv)
                
                if latent_col_name not in df.columns:
                    df[latent_col_name] = False
                    
                for _, latent_row in df_latent.iterrows():
                    sha256 = latent_row['sha256']
                    has_latent = latent_row[latent_col_name]
                    mask = df['sha256'] == sha256
                    if mask.any():
                        df.loc[mask, latent_col_name] = has_latent
                        
            # Verify latent files exist
            self._verify_latent_files(df)
        
        # Save updated metadata
        df.to_csv(metadata_path, index=False)
        print(f"✅ Updated metadata saved to: {metadata_path}")
        
        # Print summary
        if print_summary:
            self._print_metadata_summary(df)

    def _verify_latent_files(self, df):
        """Helper method to verify latent files exist"""
        latent_cols = [col for col in df.columns if col.startswith('latent_')]
        
        for latent_col in latent_cols:
            # Parse column name to get directory structure
            parts = latent_col.replace('latent_', '').split('_')
            feat_model_parts = []
            enc_model_parts = []
            found_enc = False
            
            for i, part in enumerate(parts):
                if 'enc' in part and not found_enc:
                    found_enc = True
                    enc_model_parts = parts[i:]
                elif not found_enc:
                    feat_model_parts.append(part)
                    
            feat_model = '_'.join(feat_model_parts)
            enc_model = '_'.join(enc_model_parts)
            
            latent_dir = self.output_dir / "latents" / f"{feat_model}_{enc_model}"
            if latent_dir.exists():
                # print(f"\nVerifying {latent_col} files in {latent_dir.name}...")
                verified_count = 0
                
                for idx, row in df.iterrows():
                    if row.get(latent_col, False):
                        sha256 = row['sha256']
                        latent_file = latent_dir / f"{sha256}.npz"
                        if latent_file.exists():
                            verified_count += 1
                        else:
                            print(f"  WARNING: Missing latent file for {sha256}")
                            df.loc[idx, latent_col] = False
                # print(f"  Verified {verified_count} latent files exist")

    def _print_metadata_summary(self, df):
        """Helper method to print metadata summary"""
        print("\n📊 Metadata Summary:")
        print(f"Total instances: {len(df)}")
        
        # Latent columns
        for col in df.columns:
            if col.startswith('latent_') or col.startswith('feature_'):
                count = df[col].sum()
                print(f"  {col}: {count} instances")
        
        # TRELLIS standard columns
        trellis_cols = ['rendered', 'voxelized', 'aesthetic_score']
        for col in trellis_cols:
            if col in df.columns:
                if col == 'aesthetic_score':
                    mean_score = df[col].mean()
                    print(f"  {col}: mean = {mean_score:.2f}")
                else:
                    count = df[col].sum() if df[col].dtype == bool else df[col].notna().sum()
                    print(f"  {col}: {count} instances")

    def validate_trellis_format(self, source_dir, output_dir, verbose=False):
        """
        Validate that 3DRealCar data was correctly converted to TRELLIS format.
        
        Args:
            source_dir: Source directory containing 3DRealCar data
            output_dir: Output directory where TRELLIS format data was saved
        """
        def _log(msg):
            if verbose:
                _log(msg)
        
        print(f"Validating conversion from 3DRealCar to TRELLIS format...")
        _log(f"Source directory: {source_dir}")
        _log(f"Output directory: {output_dir}")
        
        # Step 1: Check if the preprocessing completed successfully
        output_path = Path(output_dir)
        metadata_path = output_path / "metadata.csv"
        
        if not metadata_path.exists():
            _log(f"❌ Metadata file not found at {metadata_path}. Preprocessing may have failed.")
            return False
        
        # Step 2: Inspect metadata file
        import pandas as pd
        try:
            metadata = pd.read_csv(metadata_path)
            _log(f"✓ Successfully loaded metadata with {len(metadata)} entries")
            _log(f"  Metadata columns: {', '.join(metadata.columns)}")
        except Exception as e:
            _log(f"❌ Failed to load metadata: {e}")
            return False
        
        if len(metadata) == 0:
            _log(f"❌ Metadata is empty. No cars were processed successfully.")
            return False
        
        # Step 3: Check for a sample car ID
        sample_car_id = metadata['sha256'].iloc[0]
        _log(f"Using sample car ID: {sample_car_id}")
        
        # Step 4: Check expected TRELLIS directory structure (FIXED)
        required_dirs = [
            output_path / "renders" / sample_car_id,
            output_path / "voxels",
            output_path / "features" / "dinov2_vitl14_reg",
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                _log(f"❌ Required directory not found: {dir_path}")
                return False
            else:
                _log(f"✓ Found directory: {dir_path}")
        
        # Step 5: Check for transforms.json (FIXED path)
        transforms_path = output_path / "renders" / sample_car_id / "transforms.json"
        if not transforms_path.exists():
            _log(f"❌ transforms.json not found at {transforms_path}")
            return False
        
        try:
            with open(transforms_path, 'r') as f:
                transforms = json.load(f)
            _log(f"✓ Successfully loaded transforms.json with {len(transforms['frames'])} frames")
        except Exception as e:
            _log(f"❌ Failed to load transforms.json: {e}")
            return False
        
        # Step 6: Check for voxel data (FIXED path)
        voxel_ply_path = output_path / "voxels" / f"{sample_car_id}.ply"
        if not voxel_ply_path.exists():
            _log(f"❌ Voxel PLY file not found at {voxel_ply_path}")
            return False
        else:
            _log(f"✓ Found voxel PLY file: {voxel_ply_path}")
        
        # Step 7: Check for rendered images (FIXED path)
        first_frame = transforms['frames'][0]['file_path']
        image_path = output_path / "renders" / sample_car_id / first_frame.replace("./", "")
        
        if not image_path.exists():
            _log(f"❌ Rendered image not found at {image_path}")
            return False
        else:
            _log(f"✓ Found rendered image: {image_path}")
        
        # Step 8: Check mesh.ply (FIXED path)
        mesh_path = output_path / "renders" / sample_car_id / "mesh.ply"
        if not mesh_path.exists():
            _log(f"❌ Mesh file not found at {mesh_path}")
            return False
        else:
            _log(f"✓ Found mesh file: {mesh_path}")
        
        # Step 9: Check for DINOv2 features (FIXED path)
        features_path = output_path / "features" / "dinov2_vitl14_reg" / f"{sample_car_id}.npz"
        if not features_path.exists():
            _log(f"❌ DINOv2 features not found at {features_path}")
        else:
            _log(f"✓ Found DINOv2 features: {features_path}")
        
        # Step 10: Try loading the data with TRELLIS dataloaders
        try:
            # Test SparseStructure dataloader
            _log("\nTesting TRELLIS dataloaders...")
            _log("1. Testing SparseStructure dataloader")
            dataset_ss = SparseStructure(
                roots=str(output_dir),
                resolution=64,
                min_aesthetic_score=0.0  # Set to 0 to include all processed cars
            )
            _log(f"✓ Successfully loaded SparseStructure dataset with {len(dataset_ss)} instances")
            _log(f"  Dataset info:\n{dataset_ss}")
            
            # Try loading a sample
            try:
                sample = dataset_ss[0]
                _log(f"✓ Successfully loaded a sample from SparseStructure dataset")
                _log(f"  Sample contains keys: {list(sample.keys())}")
                _log(f"  Sample SS shape: {sample['ss'].shape}")
            except Exception as e:
                _log(f"❌ Failed to load a sample from SparseStructure dataset: {e}")
                import traceback
                traceback._log_exc()
        except Exception as e:
            _log(f"❌ Failed to initialize SparseStructure dataset: {e}")
            import traceback
            traceback._log_exc()
        
        # Test SparseFeat2Render dataloader if DinoV2 features were extracted
        try:
            _log("\n2. Testing SparseFeat2Render dataloader")
            
            if features_path.exists():
                dataset_feat = SparseFeat2Render(
                    roots=str(output_dir),
                    image_size=224,
                    model='dinov2_vitl14_reg',
                    resolution=64,
                    min_aesthetic_score=0.0
                )
                _log(f"✓ Successfully loaded SparseFeat2Render dataset with {len(dataset_feat)} instances")
                
                # Try loading a sample
                try:
                    sample = dataset_feat[0]
                    _log(f"✓ Successfully loaded a sample from SparseFeat2Render dataset")
                    _log(f"  Sample contains keys: {list(sample.keys())}")
                    _log(f"  Sample feature shape: {sample['feats'].shape}")
                    _log(f"  Sample coords shape: {sample['coords'].shape}")
                except Exception as e:
                    _log(f"❌ Failed to load a sample from SparseFeat2Render dataset: {e}")
                    import traceback
                    traceback._log_exc()
            else:
                _log("ℹ️ Skipping SparseFeat2Render test - DinoV2 features not found")
        except Exception as e:
            _log(f"❌ Failed to initialize SparseFeat2Render dataset: {e}")
            import traceback
            traceback._log_exc()
        
        # Test SLat2Render dataloader if latents were extracted
        try:
            _log("\n3. Testing SLat2Render dataloader")
            latents_path = output_path / "latents"
            
            if latents_path.exists() and any(latents_path.glob("*/*.npz")):
                latent_model = os.listdir(latents_path)[0]
                dataset_slat = SLat2Render(
                    roots=str(output_dir),
                    image_size=224,
                    latent_model=latent_model,
                    min_aesthetic_score=0.0
                )
                _log(f"✓ Successfully loaded SLat2Render dataset with {len(dataset_slat)} instances")
                
                # Try loading a sample
                try:
                    sample = dataset_slat[0]
                    _log(f"✓ Successfully loaded a sample from SLat2Render dataset")
                    _log(f"  Sample contains keys: {list(sample.keys())}")
                except Exception as e:
                    _log(f"❌ Failed to load a sample from SLat2Render dataset: {e}")
                    import traceback
                    traceback._log_exc()
            else:
                _log("ℹ️ Skipping SLat2Render test - latent features not found")
        except Exception as e:
            _log(f"❌ Failed to initialize SLat2Render dataset: {e}")
            import traceback
            traceback._log_exc()
        
        _log("\n✅ Validation complete!")
        return True

# Main preprocessing script
def preprocess_3drealcar(source_dir, output_dir):
    processor = RealCar3DProcessor(source_dir, output_dir, run_verifications=False)
    
    # Step 1: Convert source directory to TRELLIS format
    all_metadata = processor.convert_to_trellis()
    
    if not all_metadata:
        print(f"No car data found in {source_dir}")
        return
        
    # Save initial metadata
    df = pd.DataFrame(all_metadata)
    df.to_csv(processor.output_dir / "metadata.csv", index=False)
    print(f"✅ Converted to TRELLIS format for {len(all_metadata)} cars")
    
    # Step 2: Extract DINOv2 features for all cars
    processor.extract_dino_features_all()
    
    # Step 3: Update metadata with feature information
    processor.update_metadata('features')
    
    # Step 4: Encode latents
    processor.run_encode_latents()
    
    # Step 5: Update metadata with latent information
    processor.update_metadata('latents')
    
    # Step 6: Validate TRELLIS format
    validation_success = processor.validate_trellis_format(
        source_dir=str(processor.source_dir),
        output_dir=str(processor.output_dir)
    )
    
    if not validation_success:
        print("⚠️ Validation failed: please check the logs.")
    else:
        print("✅ Validation complete: all TRELLIS format checks passed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 3DRealCar dataset into TRELLIS format.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the 3DRealCar source directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")

    args = parser.parse_args()

    preprocess_3drealcar(
        source_dir=args.source_dir,
        output_dir=args.output_dir
    )