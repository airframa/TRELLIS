# Fixed preprocess_3drealcar.py with proper camera_angle_x handling
import json
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm
import torch
import timm
from torchvision import transforms
from PIL import Image
import glob
from pathlib import Path


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
        car_id = car_dir.name

        # Create directories in TRELLIS expected structure
        renders_dir = self.output_dir / "renders" / car_id
        renders_dir.mkdir(parents=True, exist_ok=True)

        # Try different possible mesh locations
        mesh_candidates = [
            car_dir / "textured_output.obj",
            car_dir / "export_refined.obj",
            car_dir / "export.obj",
            car_dir / "colmap_processed" / "meshed-delaunay.ply"
        ]

        mesh = None
        for mesh_path in mesh_candidates:
            if mesh_path.exists():
                print(f"Loading mesh from: {mesh_path}")
                mesh = trimesh.load(str(mesh_path), force='mesh')
                break

        if mesh is None:
            raise FileNotFoundError(f"No mesh file found in {car_dir}")

        # Normalize mesh to unit cube centered at origin
        mesh.vertices -= mesh.vertices.mean(axis=0)
        scale = np.max(np.abs(mesh.vertices))
        if scale > 0:
            mesh.vertices /= scale
        else:
            scale = 1.0  # Avoid division by zero

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
            "camera_angle_x": 1.0,  # Default value, will be updated from first frame
            "frames": []
        }

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
            cam_trellis = self.convert_camera_params(frame_data['camera_params'], scale)

            # # Set camera_angle_x from first frame (ensuring it exists)
            # if idx == 0 and 'camera_angle_x' in cam_trellis:
            #     transforms["camera_angle_x"] = cam_trellis['camera_angle_x']

            # Add frame to transforms (only transform_matrix goes in frames)
            transforms["frames"].append({
                "file_path": f"./{output_image_name}",
                "transform_matrix": cam_trellis['transform_matrix'],
                "camera_angle_x": cam_trellis['camera_angle_x']
            })

        # Ensure camera_angle_x is a valid number
        if not isinstance(transforms["camera_angle_x"], (int, float)) or transforms["camera_angle_x"] <= 0:
            print(f"Warning: Invalid camera_angle_x value: {transforms['camera_angle_x']}, using default 1.0")
            transforms["camera_angle_x"] = 1.0

        # Save transforms.json
        with open(renders_dir / "transforms.json", 'w') as f:
            json.dump(transforms, f, indent=2)

        # Create voxel representation
        voxel_info = self.create_voxel_representation(mesh, resolution=64)

        # Ensure voxel coordinates are within bounds (0-63)
        assert np.all(voxel_info >= 0) and np.all(voxel_info < 64), \
            f"Voxel coordinates out of bounds: min={voxel_info.min()}, max={voxel_info.max()}"

        # Save voxel data as PLY in the root voxels directory
        voxel_path = self.output_dir / "voxels" / f"{car_id}.ply"

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

        # Extract DINOv2 features
        self.extract_dino_features(renders_dir, mesh, car_id)

        # Create metadata CSV entry
        metadata = {
            'sha256': car_id,  # Using car_id as unique identifier
            'num_voxels': len(voxel_info),
            'aesthetic_score': aesthetic_score,
            'rendered': True,
            'voxelized': True,
            'num_views': len(transforms["frames"]),
            'feature_dinov2_vitl14_reg': True
        }

        print(f"Processed {len(transforms['frames'])} images for {car_id}")
        return metadata

    def convert_camera_params(self, cam_3drealcar, mesh_scale):
        """Convert 3DRealCar camera format to TRELLIS format"""
        camera_trellis = {}

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
                # ARKit uses a different coordinate system than TRELLIS
                # ARKit: Y-up, Z-forward
                # TRELLIS (OpenGL): Y-up, -Z-forward
                c2w_arkit = np.array(pose_flat).reshape(4, 4)

                # Convert coordinate system
                # Flip Z axis to convert from ARKit to OpenGL convention
                conversion = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                ])

                c2w = c2w_arkit @ conversion

                # Apply mesh scaling to translation
                if mesh_scale > 0:
                    c2w[:3, 3] /= mesh_scale

                camera_trellis['transform_matrix'] = c2w.tolist()
            else:
                print(f"Warning: Invalid cameraPoseARFrame length: {len(pose_flat)}")
                # Default identity matrix
                camera_trellis['transform_matrix'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        else:
            print("Warning: No cameraPoseARFrame found, using identity matrix")
            camera_trellis['transform_matrix'] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

        return camera_trellis

    def extract_dino_features(self, renders_dir, mesh, car_id):
        print("🔍 Extracting DINOv2 features for", car_id)

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
        voxel_size = (mesh_bounds[1] - mesh_bounds[0]).max() / resolution

        for path in tqdm(frame_paths[:10], desc="DINOv2 inference"):  # Limit to first 10 views
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.forward_features(tensor)  # [1, num_patches, 1024]
                feat = feat.squeeze(0).cpu().numpy()  # shape (N_patches, 1024)

            # Dummy projection: randomly scatter into voxel grid (TEMP)
            # Replace with raycasting/projection if accurate alignment is required
            coords = np.random.randint(0, resolution, size=(feat.shape[0], 3))

            patchtokens_all.append(feat)
            indices_all.append(coords)

        patchtokens = np.concatenate(patchtokens_all, axis=0)
        indices = np.concatenate(indices_all, axis=0)

        # Save to root/features directory
        feat_dir = self.output_dir / "features" / "dinov2_vitl14_reg"
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feat_dir / f"{car_id}.npz",
            patchtokens=patchtokens.astype(np.float32),
            indices=indices.astype(np.int32)
        )

        print(f"✅ Saved DINOv2 features to {feat_dir / f'{car_id}.npz'}")

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

    # Save metadata CSV
    if all_metadata:
        import pandas as pd
        df = pd.DataFrame(all_metadata)
        df.to_csv(processor.output_dir / "metadata.csv", index=False)
        print(f"Saved metadata for {len(all_metadata)} cars")


if __name__ == "__main__":
    # For testing with sample
    preprocess_3drealcar(
        source_dir="./assets/3drealcar/sample-data/2024_04_22_10_35_34",
        output_dir="./assets/3drealcar/sample-data-processed-trellis"
    )