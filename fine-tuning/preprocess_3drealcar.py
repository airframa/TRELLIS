# Fixed preprocess_3drealcar.py with proper camera_angle_x handling
import json
import random
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import timm
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path
# Add the TRELLIS root directory to sys.path
import sys
import os
trellis_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, trellis_root)
from dataset_toolkits.utils import get_file_hash


class RealCar3DProcessor:
    def __init__(self, source_dir, output_dir, sample_views=150):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sample_views = sample_views
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create root-level directories as expected by TRELLIS
        (self.output_dir / "renders").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "voxels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "features").mkdir(parents=True, exist_ok=True)

    def process_car(self, car_dir):
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
                print(f"Loading mesh from: {mesh_path}")
                mesh = trimesh.load(str(mesh_path), force='mesh')
                mesh_file = mesh_path
                break

        if mesh is None or mesh_file is None:
            raise FileNotFoundError(f"No mesh file found in {car_dir}")

        # Compute SHA256 hash of the mesh file - TRELLIS standard
        sha256 = get_file_hash(str(mesh_file))
        print(f"Computed SHA256 for {mesh_file.name}: {sha256}")
        print(f"Original car ID: {original_car_id} -> SHA256: {sha256}")

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

        # Store the transformation parameters for later use
        mesh_center = mesh.vertices.mean(axis=0)  # Should be ~0 after centering
        mesh_scale = max_extent
        
        # Verify normalization
        print(f"Normalized mesh bounds: {mesh.bounds}")
        bbox_center = mesh.bounds.mean(axis=0)
        assert np.allclose(bbox_center, 0, atol=1e-6), f"Mesh bounding box not centered: {bbox_center}"
        mean_center = mesh.vertices.mean(axis=0)
        if not np.allclose(mean_center, 0, atol=1e-2):
            print(f"⚠️ Mesh vertex mean is off-center: {mean_center} (expected ~0)")
        assert np.max(np.abs(mesh.vertices)) <= 0.5 + 1e-6, "Mesh not properly scaled"

        # Save the mesh in the expected location
        mesh_output_path = renders_dir / "mesh.ply"
        mesh.export(str(mesh_output_path))

        # Load camera parameters and find corresponding images
        frames_data = self.load_frames_data(car_dir)

        if not frames_data:
            raise ValueError(f"No frame data found in {car_dir}")

        print(f"Found {len(frames_data)} valid frames")

        # Sample views uniformly
        sampled_indices = self.sample_uniform_views(len(frames_data), target=min(self.sample_views, len(frames_data)))

        # Prepare transforms.json for TRELLIS format
        transforms = {
            "frames": []
        }

        # Store camera parameters for sampled frames
        sampled_camera_params = {}

        # Process each sampled view
        for idx, frame_idx in enumerate(sampled_indices):
            frame_data = frames_data[frame_idx]

            # Load and save image in TRELLIS format
            img = Image.open(frame_data['image_path'])
            if img.mode != 'RGBA':
                # Create alpha channel if not present
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Add full opacity alpha channel
                img.putalpha(255)

            # Save as PNG with proper naming
            output_image_name = f"frame_{idx:05d}.png"
            output_image_path = renders_dir / output_image_name
            img.save(output_image_path)

            # Convert camera parameters to TRELLIS format
            cam_trellis, K, c2w = self.convert_camera_params(
                frame_data['camera_params'], 
                max_extent,  # mesh_scale
                original_center,  # mesh_center
                frame_data['frame_num']
            )
            
            # Add frame to transforms (only transform_matrix goes in frames)
            transforms["frames"].append({
                "file_path": f"./{output_image_name}",
                "transform_matrix": cam_trellis['transform_matrix'],
                "camera_angle_x": cam_trellis['camera_angle_x']
            })

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

        # Extract DINOv2 features (pass sampled_camera_params instead of frames_data)
        self.extract_dino_features(renders_dir, mesh, sha256, sampled_camera_params)

        # Run additional verification about raycasting and reprojection
        self.verify_raycasting_reprojection(renders_dir, sampled_camera_params, mesh)

        # Add this new verification call
        self.verify_mesh_and_voxels(mesh, sha256, self.output_dir)

        # Create metadata CSV entry - Full TRELLIS compatibility
        metadata = {
            'sha256': sha256,
            'aesthetic_score': aesthetic_score,
            'rendered': True,
            'voxelized': True, 
            'num_voxels': len(voxel_info),
            'num_views': len(transforms["frames"]),
            'feature_dinov2_vitl14_reg': True,
            'cond_rendered': False,
            'captions': None,
            'local_path': str(mesh_file.relative_to(self.source_dir)),
            # Custom fields for reference
            'source_dataset': '3DRealCar',
            'original_id': original_car_id,
        }

        print(f"Processed {len(transforms['frames'])} images for SHA256: {sha256}")
        print(f"Original ID: {original_car_id} -> SHA256: {sha256}")
        
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

    def project_dino_features_to_voxels(self, *, features, camera_intrinsics, camera_extrinsics, image_size, resolution=64, return_debug=False, mesh_bounds=None):
        """
        Projects DINOv2 features to 3D voxel coordinates via raycasting.
        """
        K = camera_intrinsics.cpu().numpy()  # Already adjusted!
        c2w_input = camera_extrinsics.cpu().numpy()
        c2w = c2w_input
        
        # 1. Determine patch grid shape
        N = features.shape[0]
        possible_dims = []
        for h in range(1, int(np.sqrt(N)) + 10):
            if N % h == 0:
                w = N // h
                possible_dims.append((h, w))
        grid_h, grid_w = min(possible_dims, key=lambda x: abs(x[0] - x[1]))

        # 2. Compute patch center pixels
        # Fix grid calculation - DINOv2 always outputs 37x37 grid for 518x518 images
        patch_size = 14
        grid_w = image_size // patch_size  # 37
        grid_h = image_size // patch_size  # 37
        
        # Handle token count mismatch
        expected_tokens = grid_h * grid_w
        if features.shape[0] != expected_tokens:
            features = features[:expected_tokens]  # Truncate extra tokens
        
        # Compute patch centers
        xs = (np.arange(grid_w) + 0.5) * patch_size
        ys = (np.arange(grid_h) + 0.5) * patch_size
        px, py = np.meshgrid(xs, ys)
        px = px.flatten()
        py = py.flatten()

        # 3. Convert pixels to rays in camera frame
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        x_cam = (px - cx) / fx
        y_cam = (py - cy) / fy
        rays_cam = np.stack([x_cam, y_cam, np.ones_like(x_cam)], axis=-1)
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

        # 4. Transform rays to world space
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        rays_world = (R @ rays_cam.T).T
        origin_world = T[None, :]

        # 5. Ray-mesh intersection approach
        # Instead of fixed depth, we'll cast rays through the voxel volume
        # and sample points along the ray that fall within the mesh bounds
        
        if mesh_bounds is not None:
            # Use actual mesh bounds (should be close to [-0.5, 0.5]^3 after normalization)
            voxel_min = mesh_bounds[0]
            voxel_max = mesh_bounds[1]
        else:
            # Default normalized bounds
            voxel_min = np.array([-0.5, -0.5, -0.5])
            voxel_max = np.array([0.5, 0.5, 0.5])
        
        # Find ray-box intersections for each ray
        valid_coords = []
        valid_feats = []
        
        for i in range(len(rays_world)):
            ray_origin = origin_world[0]
            ray_dir = rays_world[i]
            
            # Ray-AABB intersection
            t_min = (voxel_min - ray_origin) / (ray_dir + 1e-8)
            t_max = (voxel_max - ray_origin) / (ray_dir + 1e-8)
            
            t_enter = np.maximum(np.minimum(t_min, t_max), 0)
            t_exit = np.maximum(t_min, t_max)
            
            t_near = np.max(t_enter)
            t_far = np.min(t_exit)
            
            if t_near < t_far and t_far > 0:
                # Ray intersects the voxel volume
                # Sample a point in the middle of the intersection
                t_sample = (t_near + t_far) * 0.5
                point_3d = ray_origin + ray_dir * t_sample
                
                # Convert to voxel coordinates
                voxel_coord = ((point_3d - voxel_min) / (voxel_max - voxel_min) * resolution).astype(np.int32)
                
                # Double-check bounds
                if np.all(voxel_coord >= 0) and np.all(voxel_coord < resolution):
                    valid_coords.append(voxel_coord)
                    valid_feats.append(features[i])
        
        if len(valid_coords) > 0:
            coords = np.array(valid_coords)
            feats = np.array(valid_feats)
        else:
            coords = np.array([]).reshape(0, 3).astype(np.int32)
            feats = np.array([]).reshape(0, features.shape[1])
        
        if return_debug:
            debug_info = {
                'out_of_bounds': N - len(coords),
                'grid_shape': (grid_h, grid_w),
                'patch_size': patch_size,
                'camera_position': T,
                'total_patches': N,
                'valid_patches': len(coords),
                'mesh_bounds': (voxel_min, voxel_max)
            }
            return coords, feats, debug_info

        return coords, feats

    def extract_dino_features(self, renders_dir, mesh, sha256, sampled_camera_params):
        print("🔍 Extracting DINOv2 features for", sha256)

        # Check if features already exist for this SHA256
        feat_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
        feat_file = feat_dir / f"{sha256}.npz"
        if feat_file.exists():
            print(f"✅ Features already exist for SHA256: {sha256}, skipping extraction")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load DINOv2 reg4 (518px input size)
        model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True)
        model.eval().to(device)

        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        frame_paths = sorted(glob.glob(str(renders_dir / "frame_*.png")))
        if not frame_paths:
            print(f"⚠️ No rendered frames found in {renders_dir}")
            return

        patchtokens_all = []
        indices_all = []

        resolution = 64
        mesh_bounds = mesh.bounds
        print(f"Mesh bounds: min={mesh_bounds[0]}, max={mesh_bounds[1]}")

        # Verification: Track feature coverage
        voxel_hit_count = np.zeros((resolution, resolution, resolution), dtype=int)
        total_valid_projections = 0
        total_out_of_bounds = 0

        num_frames = min(100, len(frame_paths))
        random_paths = random.sample(frame_paths, num_frames)

        with tqdm(random_paths, desc="DINOv2 inference", leave=False) as pbar:
            for path in pbar:
                image = Image.open(path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model.forward_features(tensor)
                    feat = feat.squeeze(0).cpu().numpy()

                idx = int(Path(path).stem.split("_")[-1])
                
                if idx not in sampled_camera_params:
                    print(f"Warning: No camera parameters for frame {idx}, skipping")
                    continue
                    
                K_original = sampled_camera_params[idx]['K']
                c2w = sampled_camera_params[idx]['c2w']
                
                # CRITICAL: Create adjusted intrinsics for 518x518 image
                original_width = K_original[0, 2] * 2
                original_height = K_original[1, 2] * 2
                image_size = 518
                
                scale_x = image_size / original_width
                scale_y = image_size / original_height
                
                K_adjusted = K_original.copy()
                K_adjusted[0, 0] *= scale_x  # fx
                K_adjusted[1, 1] *= scale_y  # fy
                K_adjusted[0, 2] = image_size / 2  # cx
                K_adjusted[1, 2] = image_size / 2  # cy

                # Project features with ADJUSTED intrinsics
                coords, feat, debug_info = self.project_dino_features_to_voxels(
                    features=feat,
                    camera_intrinsics=torch.tensor(K_adjusted),  # Use adjusted K!
                    camera_extrinsics=torch.tensor(c2w),
                    image_size=518,
                    resolution=64,
                    return_debug=True,
                    mesh_bounds=mesh_bounds
                )

                # Rest of the function remains the same...
                total_valid_projections += len(coords)
                total_out_of_bounds += debug_info['out_of_bounds']
                
                for coord in coords:
                    voxel_hit_count[coord[0], coord[1], coord[2]] += 1

                patchtokens_all.append(feat)
                indices_all.append(coords)

        # Concatenate all results
        patchtokens = np.concatenate(patchtokens_all, axis=0)
        indices = np.concatenate(indices_all, axis=0)

        # Verification Report
        print()  # Add explicit newline to separate from progress bar
        print("\n📊 Ray-casting Verification Report:")
        print(f"Total patches processed: {len(frame_paths[:10]) * 1374}")
        print(f"Valid projections: {total_valid_projections}")
        print(f"Out of bounds projections: {total_out_of_bounds}")
        print(f"Projection success rate: {total_valid_projections / (total_valid_projections + total_out_of_bounds) * 100:.1f}%")
        
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
        
        # Visualize coverage heatmap
        self._save_coverage_visualization(voxel_hit_count, renders_dir / "dino_coverage_debug.png")

        # Save features using SHA256 - TRELLIS standard
        # feat_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feat_dir / f"{sha256}.npz",
            patchtokens=patchtokens.astype(np.float32),
            indices=indices.astype(np.int32)
        )

        print(f"✅ Saved DINOv2 features to {feat_dir / f'{sha256}.npz'}")

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
            
        # Debug visualization
        # fig, ax = plt.subplots(figsize=(10, 10))
        # # Load original resolution image
        # img = Image.open(renders_dir / "frame_00000.png")
        # ax.imshow(img)
        
        # # Calculate scaling factors
        # orig_width, orig_height = img.size
        # scale_x = orig_width / 518
        # scale_y = orig_height / 518
        
        # # Plot reprojected voxels (scaled to original image size)
        # if np.sum(in_bounds) > 0:
        #     scaled_points = points_img[in_bounds].copy()
        #     scaled_points[:, 0] *= scale_x
        #     scaled_points[:, 1] *= scale_y
        #     ax.scatter(scaled_points[:, 0], scaled_points[:, 1], 
        #             c='red', s=1, alpha=0.5, label='Voxel centers')
        
        # # Also plot the mesh center (scaled)
        # if p_cam[2] > 0:
        #     scaled_p_img = p_img.copy()
        #     scaled_p_img[0] *= scale_x
        #     scaled_p_img[1] *= scale_y
        #     ax.scatter(scaled_p_img[0], scaled_p_img[1], c='green', s=100, marker='o', label='Mesh center')
        
        # # Plot bounding box corners for reference
        # bounds = mesh.bounds
        # corners = []
        # for x in [bounds[0][0], bounds[1][0]]:
        #     for y in [bounds[0][1], bounds[1][1]]:
        #         for z in [bounds[0][2], bounds[1][2]]:
        #             corners.append([x, y, z, 1])
        # corners = np.array(corners)
        
        # corners_cam = (w2c @ corners.T).T[:, :3]
        # corners_valid = corners_cam[corners_cam[:, 2] > 0]
        # if len(corners_valid) > 0:
        #     corners_img = (K @ corners_valid.T).T
        #     corners_img = corners_img[:, :2] / corners_img[:, 2:3]
        #     ax.scatter(corners_img[:, 0], corners_img[:, 1], 
        #             c='blue', s=50, marker='x', label='BBox corners')
        
        # ax.set_xlim(0, 518)
        # ax.set_ylim(518, 0)  # Flip Y axis for image coordinates
        # ax.legend()
        # ax.set_title('Reprojection Debug - Camera 0')
        
        # plt.savefig(renders_dir / "reprojection_debug.png", bbox_inches='tight', dpi=150)
        # plt.close()
        # print(f"💾 Saved detailed debug to {renders_dir / 'reprojection_debug.png'}")
        
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
        """Load all frame data (images and camera parameters)"""
        frames_data = []

        # Find all frame json files
        json_files = sorted(car_dir.glob("frame_*.json"))

        # Possible image locations
        image_dirs = [
            car_dir,
            car_dir / "colmap_processed" / "images",
            car_dir / "images",
        ]

        for json_file in json_files:
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
                continue

            # Load camera parameters
            try:
                with open(json_file, 'r') as f:
                    camera_params = json.load(f)

                frames_data.append({
                    'image_path': image_path,
                    'camera_params': camera_params,
                    'frame_num': frame_num
                })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        return frames_data

    def calculate_aesthetic_score(self, mesh, renders_dir):
        """
        Calculate aesthetic score using LAION aesthetic predictor.
        Randomly samples frames and averages their scores.
        """
        try:
            import torch
            import open_clip
            from PIL import Image
            import random
            import glob

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
                print("Loading CLIP model for aesthetic scoring...")
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
                print("LAION aesthetic predictor loaded successfully")

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

            print(f"LAION aesthetic score: {final_score:.3f} (from {len(scores)} frames, "
                f"range: {min(scores):.3f}-{max(scores):.3f})")

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


# Main preprocessing script
def preprocess_3drealcar(source_dir, output_dir):
    processor = RealCar3DProcessor(source_dir, output_dir)
    source_path = Path(source_dir)

    # Collect all metadata
    all_metadata = []

    # Check if source_dir itself contains car data
    if any(source_path.glob("frame_*.json")):
        # Process the directory itself as a single car
        print(f"Processing single car in: {source_path}")
        try:
            metadata = processor.process_car(source_path)
            all_metadata.append(metadata)
            print(f"Successfully processed car: {source_path.name}")
        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Look for subdirectories containing car data
        print("Looking for car subdirectories...")
        car_dirs = []
        for d in source_path.iterdir():
            if d.is_dir() and any(d.glob("frame_*.json")):
                car_dirs.append(d)

        if not car_dirs:
            print(f"No car data found in {source_path}")
            return

        print(f"Found {len(car_dirs)} cars to process")

        for car_dir in tqdm(car_dirs, desc="Processing cars"):
            try:
                metadata = processor.process_car(car_dir)
                all_metadata.append(metadata)
                print(f"Successfully processed: {car_dir.name}")
            except Exception as e:
                print(f"Error processing {car_dir.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save metadata CSV in TRELLIS format
    if all_metadata:
        import pandas as pd
        df = pd.DataFrame(all_metadata)
        
        # Save with SHA256 as index (TRELLIS standard)
        df.set_index('sha256', inplace=True)
        df.to_csv(processor.output_dir / "metadata.csv")
        
        print(f"✅ Saved TRELLIS-compatible metadata for {len(all_metadata)} cars")


if __name__ == "__main__":
    # Configuration - easily adjustable
    SAMPLE_DATA_DIR = "./assets/3drealcar/HQ200"
    OUTPUT_DIR = "./assets/3drealcar/HQ200-processed-trellis"
    
    # Process all vehicles in the sample data directory
    preprocess_3drealcar(
        source_dir=SAMPLE_DATA_DIR,
        output_dir=OUTPUT_DIR
    )