import torch
import os
import sys
import requests
from pathlib import Path
import trimesh # Used by samesh and useful for type hints
import copy
import json
import numpy as np
import subprocess # Added for running the worker script
from PIL import Image
from omegaconf import OmegaConf, DictConfig
from huggingface_hub import hf_hub_download
import folder_paths # ComfyUI paths

# Try importing pyrender, provide guidance if it fails
try:
    import pyrender
    # Set backend explicitly for headless potentially
    # os.environ['PYOPENGL_PLATFORM'] = 'egl' # Or 'osmesa' if EGL not available
except ImportError:
    print(" [93mWarning: pyrender not found. The SamMeshRenderer node will not work. \nPlease install it (pip install pyrender) and potentially required system libraries (like EGL or OSMesa for headless rendering). [0m")
    pyrender = None # Set to None so we can check later

# --- Setup & Checks ---
# Corrected path check: removed extra "samesh-main"
samesh_src_dir_to_check = os.path.normpath(os.path.join("samesh-main", "src")) # Normalized path ending

samesh_src_dir_found = None
for p in sys.path:
    try:
        # Normalize the path from sys.path before checking
        normalized_p = os.path.normpath(p)
        if normalized_p.endswith(samesh_src_dir_to_check):
            samesh_src_dir_found = p # Store the original path from sys.path
            print(f"Found samesh src dir in sys.path: {p}") # Debug message
            break
    except Exception as e:
         # Handle potential errors with paths in sys.path (e.g., None values)
         # print(f"Debug: Error checking sys.path entry '{p}': {e}")
         pass # Ignore problematic paths

# Path to the current custom node directory
comfyui_samesh_node_dir = os.path.dirname(os.path.realpath(__file__))
# Path to the worker script
WORKER_SCRIPT_PATH = os.path.join(comfyui_samesh_node_dir, "_run_segmentation_worker.py")

# Debug: Print the calculated base directory for the node
print(f"ComfyUI-Sam-Mesh Node Dir: {comfyui_samesh_node_dir}")

# Guard imports based on the FOUND path
# We still need these for type hints and potentially for reloading the mesh later
read_mesh = None
# segment_mesh_samesh_func = None # No longer needed directly in the node
SamModelMesh = None
remove_texture = None
colormap_faces_mesh = None
if samesh_src_dir_found: # Use the flag set in the loop
    try:
        # Only import what's needed by nodes *other* than Segmenter, or for loading results
        from samesh.data.loaders import read_mesh, remove_texture
        from samesh.models.sam_mesh import SamModelMesh, colormap_faces_mesh # Keep these imports if needed elsewhere or for type hints
    except ImportError as e:
        print(f"\033[31mError importing samesh functions (even though path seemed present): {e}\033[0m")
        # Fallback definitions might still be needed if other nodes use these
        read_mesh = None
        SamModelMesh = None
        remove_texture = None
        colormap_faces_mesh = None
else:
    print("\033[31mError: samesh source directory not found in sys.path after check. Check __init__.py path addition and submodule status.\033[0m")

# Fallback definitions if import fails
# if segment_mesh_samesh_func is None: # No longer calling this directly
#     def segment_mesh_samesh_func(*args, **kwargs):
#         raise ImportError("samesh.models.sam_mesh.segment_mesh could not be imported.")
if read_mesh is None:
    def read_mesh(*args, **kwargs):
        raise ImportError("samesh.data.loaders.read_mesh could not be imported.")
# ... (add similar fallbacks for others if needed)

# Default ComfyUI output/temp directories
output_dir = folder_paths.get_output_directory()
temp_dir = folder_paths.get_temp_directory()
sam_model_dir = os.path.join(folder_paths.models_dir, "sam") # Standard ComfyUI SAM dir
DEFAULT_OUTPUT_DIR = os.path.join(output_dir, "samesh")
DEFAULT_CACHE_DIR = os.path.join(temp_dir, "samesh_cache")

# Create SAM model directory if it doesn't exist
os.makedirs(sam_model_dir, exist_ok=True)

# --- SAM Model Definitions ---
SAM_MODELS = {
    "SAM2 Hiera Large": {
        "checkpoint_filename": "sam2_hiera_large.pt",
        "config_filename": "sam2_hiera_l.yaml",
        "repo_id": "facebook/sam2-hiera-large",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml"
    },
    "SAM2 Hiera Base+": {
        "checkpoint_filename": "sam2_hiera_base_plus.pt",
        "config_filename": "sam2_hiera_b+.yaml",
        "repo_id": "facebook/sam2-hiera-base-plus",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_b%2B.yaml" # Note: URL encoding for '+'
    },
    "SAM2 Hiera Small": {
        "checkpoint_filename": "sam2_hiera_small.pt",
        "config_filename": "sam2_hiera_s.yaml",
        "repo_id": "facebook/sam2-hiera-small",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_s.yaml"
    },
    "SAM2 Hiera Tiny": {
        "checkpoint_filename": "sam2_hiera_tiny.pt",
        "config_filename": "sam2_hiera_t.yaml",
        "repo_id": "facebook/sam2-hiera-tiny",
        "config_url": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_t.yaml"
    },
    # Add original SAM or other models here if needed/supported by samesh
}
SAM_MODEL_NAMES = list(SAM_MODELS.keys())

# --- SamModelDownloader Node ---
class SamModelDownloader:
    """
    Downloads a selected SAM2 Hiera model checkpoint and config required by SamMesh.
    Places them in the standard ComfyUI SAM model directory.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "model_name": (SAM_MODEL_NAMES, {"default": SAM_MODEL_NAMES[0], "tooltip_prompt": "Select the SAM2 Hiera model to download."}),
             }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("sam_checkpoint_path", "sam_model_config_path",)
    FUNCTION = "download_model"
    CATEGORY = "SamMesh/Utils"

    def download_file(self, url, save_path, model_name):
        try:
            print(f"SamModelDownloader ({model_name}): Downloading {os.path.basename(save_path)} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1MB

            from tqdm import tqdm
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
            print(f"SamModelDownloader ({model_name}): Download complete: {save_path}")
            return save_path
        except Exception as e:
            print(f"\033[31mError downloading {url} for {model_name}: {e}[0m")
            if os.path.exists(save_path): # Clean up partial download
                os.remove(save_path)
            raise # Re-raise the exception

    def download_model(self, model_name: str):
        if model_name not in SAM_MODELS:
            raise ValueError(f"Selected model '{model_name}' is not defined.")

        model_info = SAM_MODELS[model_name]
        checkpoint_filename = model_info["checkpoint_filename"]
        config_filename = model_info["config_filename"]
        repo_id = model_info["repo_id"]
        config_url = model_info["config_url"]

        checkpoint_path = os.path.join(sam_model_dir, checkpoint_filename)
        config_path = os.path.join(sam_model_dir, config_filename)

        # Download Checkpoint if missing
        if not os.path.exists(checkpoint_path):
            print(f"SamModelDownloader ({model_name}): Checkpoint {checkpoint_filename} not found. Downloading...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=checkpoint_filename,
                    local_dir=sam_model_dir,
                    local_dir_use_symlinks=False, # Good practice for ComfyUI
                    resume_download=True
                )
                print(f"SamModelDownloader ({model_name}): Checkpoint downloaded to {checkpoint_path}")
            except Exception as e:
                print(f"\033[31mError downloading checkpoint for {model_name} from Hugging Face: {e}[0m")
                raise
        else:
            print(f"SamModelDownloader ({model_name}): Checkpoint found: {checkpoint_path}")

        # Download Config YAML if missing
        if not os.path.exists(config_path):
             print(f"SamModelDownloader ({model_name}): Config {config_filename} not found. Downloading...")
             self.download_file(config_url, config_path, model_name)
        else:
             print(f"SamModelDownloader ({model_name}): Config found: {config_path}")

        # Verify files exist after attempting download
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Failed to download or locate checkpoint for {model_name}: {checkpoint_path}")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Failed to download or locate config for {model_name}: {config_path}")

        return (checkpoint_path, config_path,)

# --- SamMeshLoader Node --- (Modified to pass mesh_path along)
MESH_EXTENSIONS = ['.glb', '.gltf', '.obj', '.ply', '.stl', '.3mf'] # Common mesh extensions

class SamMeshLoader:
    """
    A node to load mesh data using samesh's loader.
    Provides a dropdown list of meshes found in the ComfyUI input directory.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        try:
            for f in os.listdir(input_dir):
                if os.path.isfile(os.path.join(input_dir, f)):
                    _, ext = os.path.splitext(f)
                    if ext.lower() in MESH_EXTENSIONS:
                        files.append(f)
        except Exception as e:
            print(f" [93mWarning [SamMeshLoader]: Could not list input directory {input_dir}: {e} [0m")

        return {
            "required": {
                "mesh": (sorted(files) if files else ["No mesh files found"], {"tooltip_prompt": "Select a mesh file from the ComfyUI input directory."}),
            },
            "optional": {
                "normalize_mesh": ("BOOLEAN", {"default": False, "tooltip_prompt": "Scales and translates the mesh to fit within a unit cube centered at the origin. Useful for standardizing mesh size for consistent SAM processing, but can alter original scale."}),
                "process_mesh": ("BOOLEAN", {"default": True, "tooltip_prompt": "Enables Trimesh's default processing (e.g., removing duplicate vertices, merging close vertices, fixing face winding). Can help clean up messy geometry but may alter vertex/UV data or topology. Disable if precise original structure is critical, especially with textures."}),
            }
        }

    RETURN_TYPES = ("SAM_MESH", "STRING",)
    RETURN_NAMES = ("sam_mesh", "loaded_mesh_path",)
    FUNCTION = "load_mesh"
    CATEGORY = "SamMesh"

    def load_mesh(self, mesh: str, normalize_mesh: bool = False, process_mesh: bool = True):
        if read_mesh is None:
            raise ImportError("samesh.data.loaders.read_mesh could not be imported.")

        if mesh == "No mesh files found":
             raise ValueError("No mesh files found in the ComfyUI input directory.")

        # Construct full path from filename
        mesh_full_path = folder_paths.get_annotated_filepath(mesh)

        if not mesh_full_path or not os.path.exists(mesh_full_path):
             raise FileNotFoundError(f"Mesh file not found or path is invalid: {mesh_full_path} (from filename: {mesh})")

        print(f"SamMeshLoader: Loading mesh from: {mesh_full_path}")
        mesh_file_path = Path(mesh_full_path)

        try:
            # Use the imported read_mesh function from samesh
            loaded_mesh = read_mesh(mesh_file_path, norm=normalize_mesh, process=process_mesh)

            if loaded_mesh is None:
                raise ValueError(f"samesh.data.loaders.read_mesh returned None for {mesh_full_path}. Is the file valid?")
            
            # --- Debug Prints for Material --- 
            print(f"SamMeshLoader Debug: Loaded mesh type: {type(loaded_mesh)}")
            if hasattr(loaded_mesh, 'visual'):
                print(f"SamMeshLoader Debug: Visual type: {type(loaded_mesh.visual)}")
                if hasattr(loaded_mesh.visual, 'material') and loaded_mesh.visual.material is not None:
                    print(f"SamMeshLoader Debug: Material type: {type(loaded_mesh.visual.material)}")
                    if isinstance(loaded_mesh.visual.material, trimesh.visual.material.PBRMaterial):
                        print(f"SamMeshLoader Debug: PBR baseColorTexture: {getattr(loaded_mesh.visual.material, 'baseColorTexture', 'Not found')}")
                        print(f"SamMeshLoader Debug: PBR metallicRoughnessTexture: {getattr(loaded_mesh.visual.material, 'metallicRoughnessTexture', 'Not found')}")
                    elif hasattr(loaded_mesh.visual.material, 'image_texture'): # For SimpleMaterial or older TextureVisuals
                        print(f"SamMeshLoader Debug: Material image_texture: {getattr(loaded_mesh.visual.material, 'image_texture', 'Not found')}")
                    else:
                        print("SamMeshLoader Debug: Material is of an unrecognized type for texture checking or has no direct image_texture.")
                else:
                    print("SamMeshLoader Debug: Mesh visual has no material or material is None.")
                if hasattr(loaded_mesh.visual, 'uv') and loaded_mesh.visual.uv is not None:
                    print(f"SamMeshLoader Debug: UVs are present. Count: {len(loaded_mesh.visual.uv)}")
                else:
                    print("SamMeshLoader Debug: Mesh visual has no UVs or UVs are None.")
            else:
                print("SamMeshLoader Debug: Loaded mesh has no visual attribute.")
            # --- End Debug Prints ---

            if not isinstance(loaded_mesh, trimesh.Trimesh):
                 print(f"Warning: read_mesh did not return a Trimesh object (got {type(loaded_mesh)}). Returning as is.")

            print(f"SamMeshLoader: Mesh loaded successfully. Vertices: {len(loaded_mesh.vertices)}, Faces: {len(loaded_mesh.faces)}")
            # Make sure the mesh path is absolute for the worker script
            absolute_mesh_path = os.path.abspath(mesh_full_path)
            return (loaded_mesh, absolute_mesh_path,)

        except Exception as e:
            print(f"\033[31mError loading mesh {mesh_full_path}: {e}\033[0m")
            # Re-raise the exception so ComfyUI knows the node failed
            raise

# --- SamMeshSegmenter Node --- (Modified)
class SamMeshSegmenter:
    """
    Segments a mesh using the SAMesh model. Loads the result into memory
    and outputs the mesh object and the path to the face-to-label JSON file.
    The intermediate segmented mesh file generated by the worker is deleted.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam_mesh": ("SAM_MESH",), # Input mesh object
                "mesh_path": ("STRING", {"forceInput": True}), # Original mesh path
                "sam_checkpoint_path": ("STRING", {"forceInput": True}), # From Downloader
                "sam_model_config_path": ("STRING", {"forceInput": True}),# From Downloader
                # --- User configurable parameters ---
                "output_directory": ("STRING", {"default": DEFAULT_OUTPUT_DIR, "tooltip_prompt": "Directory to save the face-to-label JSON file and intermediate files."}),
                "cache_directory": ("STRING", {"default": DEFAULT_CACHE_DIR, "tooltip_prompt": "Directory for samesh to cache rendering or computation results."}),
                "keep_texture": ("BOOLEAN", {"default": False, "tooltip_prompt": "If checked, the segmentation process will attempt to preserve the original mesh's texture and apply it to the segmented parts. UV mapping quality depends on the samesh library's capabilities."}),
            },
            "optional": {
                 "target_labels": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip_prompt": "Desired number of distinct segments. Set to -1 for the samesh library's default/automatic behavior. Higher numbers may result in finer-grained segmentation."}),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip_prompt": "Random seed for reproducible segmentation results."}),
            }
        }

    RETURN_TYPES = ("SAM_MESH", "STRING",)
    RETURN_NAMES = ("segmented_mesh", "face2label_path",)
    FUNCTION = "segment_mesh"
    CATEGORY = "SamMesh"

    def segment_mesh(self, sam_mesh: trimesh.Trimesh, mesh_path: str,
                     sam_checkpoint_path: str, sam_model_config_path: str,
                     output_directory: str, cache_directory: str,
                     keep_texture: bool, target_labels: int = -1,
                     seed: int = 0
                     ):

        # --- Input Validation ---
        if not isinstance(sam_mesh, trimesh.Trimesh):
            print(f"Warning: Input 'sam_mesh' is not a Trimesh object (got {type(sam_mesh)}). Attempting to proceed, but errors may occur.")
        if not mesh_path or not os.path.exists(mesh_path):
             mesh_path = os.path.abspath(mesh_path)
             if not os.path.exists(mesh_path):
                  raise FileNotFoundError(f"Original mesh file path is invalid or missing: {mesh_path}")
        else:
             mesh_path = os.path.abspath(mesh_path)

        if not sam_checkpoint_path or not os.path.exists(sam_checkpoint_path):
            raise FileNotFoundError(f"SAM checkpoint file not found: {sam_checkpoint_path}")
        if not sam_model_config_path or not os.path.exists(sam_model_config_path):
            raise FileNotFoundError(f"SAM model config file not found: {sam_model_config_path}")

        print(f"SamMeshSegmenter: Starting segmentation for: {mesh_path} via worker script.")

        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(cache_directory, exist_ok=True)

        # --- Prepare arguments for the worker script ---
        output_filename_prefix = "segmented"
        output_extension = "glb"
        visualize = False

        # Cap the seed to be within the valid range for numpy.random.seed (0 to 2**32 - 1)
        # 2**32 - 1 = 4294967295
        # Using modulo is a common way to bring a large seed into the valid range.
        capped_seed = seed % (2**32)

        cmd = [
            sys.executable,
            WORKER_SCRIPT_PATH,
            "--mesh_path", mesh_path,
            "--sam_checkpoint_path", sam_checkpoint_path,
            "--sam_model_config_path", sam_model_config_path,
            "--output_directory", output_directory,
            "--cache_directory", cache_directory,
            "--output_filename_prefix", output_filename_prefix,
            "--output_extension", output_extension,
            "--target_labels", str(target_labels),
        ]
        if keep_texture:
            cmd.append("--keep_texture")
        
        cmd.extend(["--seed", str(capped_seed)])

        # --- Prepare Environment for Subprocess ---
        env = os.environ.copy()
        try:
             comfy_base = folder_paths.base_path
             env['COMFYUI_BASE_PATH'] = comfy_base
        except AttributeError:
             print("Warning: Could not get ComfyUI base path via folder_paths.base_path. Path resolution inside worker might fail.")
             env['COMFYUI_BASE_PATH'] = ''

        # --- Execute Worker Script ---
        print(f"SamMeshSegmenter: Executing worker: {' '.join(cmd)}")
        try:
            samesh_worker_cwd = os.path.join(comfyui_samesh_node_dir, "samesh-main", "src")
            print(f"SamMeshSegmenter: Setting worker CWD to: {samesh_worker_cwd}")
            process = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, cwd=samesh_worker_cwd)
            print(f"SamMeshSegmenter: Worker stdout:\n{process.stdout}")
            if process.stderr:
                 print(f"SamMeshSegmenter: Worker stderr:\n{process.stderr}")

            # --- Parse result from worker stdout ---
            last_line = process.stdout.strip().splitlines()[-1]
            result_data = json.loads(last_line)
            final_output_mesh_path = result_data.get("output_mesh_path")
            final_output_json_path = result_data.get("face2label_path")

            if not final_output_mesh_path or "Error:" in final_output_mesh_path:
                 raise RuntimeError(f"Worker script failed to produce mesh output: {final_output_mesh_path}")
            if not final_output_json_path or "Error:" in final_output_json_path:
                 raise RuntimeError(f"Worker script failed to produce JSON output: {final_output_json_path}")

            print(f"SamMeshSegmenter: Worker finished successfully. Output JSON: {final_output_json_path}")
            print(f"SamMeshSegmenter: Loading result mesh from {final_output_mesh_path}...")

            if not os.path.exists(final_output_mesh_path):
                 raise FileNotFoundError(f"Segmented mesh file generated by worker not found: {final_output_mesh_path}")

            # --- Load the resulting mesh & DELETE intermediate file --- #
            segmented_colored_mesh = trimesh.load(final_output_mesh_path, force='mesh')
            if not isinstance(segmented_colored_mesh, trimesh.Trimesh):
                 raise TypeError(f"Loaded result mesh is not a Trimesh object (got {type(segmented_colored_mesh)}) from path: {final_output_mesh_path}")

            # print(f"SamMeshSegmenter: Result mesh loaded. Deleting intermediate file: {final_output_mesh_path}")
            # try:
            #     os.remove(final_output_mesh_path)
            #     print(f"SamMeshSegmenter: Intermediate file deleted.")
            # except OSError as e:
            #      print(f" [93mWarning: Could not delete intermediate segmented mesh file {final_output_mesh_path}: {e} [0m")

            return (segmented_colored_mesh, final_output_json_path,)

        except subprocess.CalledProcessError as e:
            print(f"\033[31mError: Worker script failed (return code {e.returncode}).\033[0m")
            print(f"\033[31mCommand: {' '.join(e.cmd)}\033[0m")
            print(f"\033[31mStderr:\n{e.stderr}\033[0m")
            print(f"\033[31mStdout:\n{e.stdout}\033[0m")
            raise RuntimeError(f"Segmentation worker script failed. Check logs for details.") from e
        except json.JSONDecodeError as e:
            print(f"\033[31mError: Could not parse JSON result from worker script stdout.\033[0m")
            print(f"\033[31mStdout:\n{process.stdout}\033[0m")
            raise RuntimeError(f"Failed to parse result from segmentation worker.") from e
        except Exception as e:
            print(f"\033[31mError during segmentation subprocess execution or result loading: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise

# --- SamMeshExporter Node ---
class SamMeshExporter:
    """
    Exports the segmented mesh as a single GLB file containing a Trimesh Scene.
    Each object in the scene corresponds to one segment, suitable for use with HoloPart.
    """
    RETURN_TYPES = ()
    FUNCTION = "export_parts"
    OUTPUT_NODE = True
    CATEGORY = "SamMesh/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_mesh": ("SAM_MESH",),
                "face2label_path": ("STRING", {"forceInput": True}),
                "output_filename": ("STRING", {"default": "holopart_input.glb", "tooltip_prompt": "Filename for the exported GLB scene (e.g., 'my_segmented_parts.glb')."}),
            }
        }

    def export_parts(self, segmented_mesh: trimesh.Trimesh, face2label_path: str, output_filename: str):
        if not os.path.exists(face2label_path):
            raise FileNotFoundError(f"Face-to-label JSON file not found: {face2label_path}")

        output_base_dir = folder_paths.get_output_directory()
        if not output_filename.lower().endswith('.glb'):
            output_filename += '.glb'
            
        final_output_path = os.path.join(output_base_dir, output_filename)

        print(f"SamMeshExporter: Preparing scene for HoloPart input: {final_output_path}")

        try:
            with open(face2label_path, 'r') as f:
                face2label = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load or parse face2label JSON from {face2label_path}: {e}")

        try:
            face_labels = {int(k): int(v) for k, v in face2label.items()}
            unique_labels = sorted(list(set(face_labels.values())))
            print(f"SamMeshExporter: Found {len(unique_labels)} unique segment labels.")
        except Exception as e:
             raise ValueError(f"Error processing face labels from JSON (ensure keys/values are integers): {e}")

        base_mesh = segmented_mesh
        mesh_parts = []

        # --- Check if the base mesh has texture information ---
        has_texture_info = False
        if (hasattr(base_mesh, 'visual') and 
            isinstance(base_mesh.visual, trimesh.visual.TextureVisuals) and 
            hasattr(base_mesh.visual, 'uv') and 
            base_mesh.visual.uv is not None and 
            hasattr(base_mesh.visual, 'material') and 
            base_mesh.visual.material is not None and
            len(base_mesh.visual.uv) == len(base_mesh.vertices)): # Ensure UVs match vertices
            has_texture_info = True
            print("SamMeshExporter: Base mesh has texture info (UVs and Material). Attempting to texture segments.")
        else:
            print("SamMeshExporter: Warning - Base mesh lacks complete texture info (UVs or Material). Segments will not be textured.")
        # ------------------------------------------------------

        for label in unique_labels:
            faces_for_label_mask = [idx for idx, lbl in face_labels.items() if lbl == label]

            if not faces_for_label_mask:
                 print(f"SamMeshExporter: Warning - No faces found for label {label}. Skipping.")
                 continue

            try:
                face_indices_for_label = np.array(faces_for_label_mask, dtype=np.int64)
                segment_faces = base_mesh.faces[face_indices_for_label]
                unique_vertex_indices = np.unique(segment_faces)
                segment_vertices = base_mesh.vertices[unique_vertex_indices]
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}
                new_segment_faces = np.array([
                    [vertex_map[v_idx] for v_idx in face]
                    for face in segment_faces
                ], dtype=np.int64)

                segment_mesh = trimesh.Trimesh(vertices=segment_vertices, faces=new_segment_faces, process=False)
                # --- BEGIN ADDED DEBUG ---
                print(f"SamMeshExporter Debug (Label {label}): segment_mesh created. Vertices: {segment_mesh.vertices.shape}, Faces: {segment_mesh.faces.shape}")
                # --- END ADDED DEBUG ---

                # --- Assign Texture/Material if available ---
                if has_texture_info:
                    try:
                        # Extract the relevant UV coordinates for the segment's vertices
                        segment_uvs = base_mesh.visual.uv[unique_vertex_indices]

                        # --- BEGIN ADDED DEBUG (from previous session, kept for context) ---
                        if isinstance(segment_uvs, np.ndarray) and segment_uvs.ndim == 2 and segment_uvs.shape[1] == 2 and segment_uvs.shape[0] > 0:
                            print(f"SamMeshExporter Debug (Label {label}): segment_uvs shape: {segment_uvs.shape}, " +
                                  f"min_uv: {np.min(segment_uvs, axis=0)}, max_uv: {np.max(segment_uvs, axis=0)}, " +
                                  f"num_segment_vertices_from_source: {len(segment_vertices)}") # Changed last part for clarity
                            if segment_uvs.shape[0] != len(segment_vertices): # len(segment_vertices) is source for segment_uvs count
                                print(f" [91mSamMeshExporter Error (Label {label}): UV source count {segment_uvs.shape[0]} != Original segment vertex count {len(segment_vertices)} [0m")
                        elif isinstance(segment_uvs, np.ndarray):
                            print(f" [93mSamMeshExporter Warning (Label {label}): segment_uvs has unexpected data: shape {segment_uvs.shape}, type {segment_uvs.dtype} [0m")
                        else:
                            print(f" [93mSamMeshExporter Warning (Label {label}): segment_uvs is not a numpy array or is empty. Type: {type(segment_uvs)} [0m")
                        # --- END ADDED DEBUG ---

                        # --- BEGIN NEW CRITICAL CHECK ---
                        if segment_mesh.vertices.shape[0] != segment_uvs.shape[0]:
                            print(f" [91mSamMeshExporter CRITICAL ERROR (Label {label}): Actual segment_mesh vertex count {segment_mesh.vertices.shape[0]} MISMATCHES segment_uvs count {segment_uvs.shape[0]} BEFORE TextureVisuals assignment! [0m")
                        else:
                            print(f"SamMeshExporter Debug (Label {label}): Actual segment_mesh vertex count {segment_mesh.vertices.shape[0]} matches segment_uvs count {segment_uvs.shape[0]} before assignment.")
                        # --- END NEW CRITICAL CHECK ---

                        original_texture_image = None
                        # Ensure the base material is PBRMaterial and has a baseColorTexture
                        if isinstance(base_mesh.visual.material, trimesh.visual.material.PBRMaterial) and \
                           hasattr(base_mesh.visual.material, 'baseColorTexture') and \
                           base_mesh.visual.material.baseColorTexture is not None:
                            original_texture_image = base_mesh.visual.material.baseColorTexture
                        # Fallback for non-PBR or differently structured materials that might have an image_texture
                        elif hasattr(base_mesh.visual.material, 'image_texture') and \
                             base_mesh.visual.material.image_texture is not None:
                            original_texture_image = base_mesh.visual.material.image_texture

                        if original_texture_image is not None:
                            # Create a new PBRMaterial for the segment
                            # You can extend this to copy other PBR properties if needed
                            new_material = trimesh.visual.material.PBRMaterial(
                                baseColorTexture=original_texture_image,
                                metallicFactor=getattr(base_mesh.visual.material, 'metallicFactor', 0.0),
                                roughnessFactor=getattr(base_mesh.visual.material, 'roughnessFactor', 0.5)
                                # Add other PBR properties from base_mesh.visual.material here if necessary
                            )
                            segment_mesh.visual = trimesh.visual.TextureVisuals(
                                uv=segment_uvs,
                                material=new_material 
                            )
                            # print(f"Segment {label} textured with new PBRMaterial.") # Optional debug
                        else:
                            print(f" [93mSamMeshExporter: Warning - Could not extract texture image from base_mesh material for label {label}. Segment will not be textured. [0m")
                            segment_mesh.visual = trimesh.visual.ColorVisuals() # Fallback to no texture

                    except Exception as tex_e:
                        print(f" [93mSamMeshExporter: Error applying texture to label {label}: {tex_e} [0m")
                        # Fallback: remove potentially incomplete visual info
                        segment_mesh.visual = trimesh.visual.ColorVisuals()
                else:
                     # Fallback for non-textured input or if base mesh lacked texture info
                     # Check if base_mesh has vertex colors instead (e.g., from a previous colored export)
                     if hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'vertex_colors') and len(base_mesh.visual.vertex_colors) == len(base_mesh.vertices):
                         segment_mesh.visual = trimesh.visual.ColorVisuals(
                             mesh=segment_mesh,
                             vertex_colors=base_mesh.visual.vertex_colors[unique_vertex_indices]
                         )
                # --------------------------------------------

                mesh_parts.append(segment_mesh)

            except Exception as e:
                print(f" [91mSamMeshExporter: Error processing label {label}: {e} [0m")
                import traceback
                traceback.print_exc()

        if not mesh_parts:
            print(f"SamMeshExporter: No mesh parts were generated. Skipping scene export.")
            return {}

        try:
             scene = trimesh.Scene(mesh_parts)
             print(f"SamMeshExporter: Exporting scene with {len(mesh_parts)} segments to {final_output_path}")
             scene.export(final_output_path, file_type='glb')
             print(f"SamMeshExporter: Finished exporting scene.")
        except Exception as e:
             print(f" [91mSamMeshExporter: Error exporting scene to {final_output_path}: {e} [0m")
             import traceback
             traceback.print_exc()
             raise

        return {}


# --- SamMeshRenderer Node ---
class SamMeshRenderer:
    """
    Renders 4 views (front, right, top, back) of the input mesh
    and combines them into a single 1024x1024 image grid.
    Requires pyrender and Pillow.
    """
    @classmethod
    def INPUT_TYPES(s):
        if pyrender is None:
             return {"required": {"error": ("STRING", {"default": "pyrender not installed. Node disabled."})}}
        return {
            "required": {
                "mesh": ("SAM_MESH",),
                "render_resolution": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64, "tooltip_prompt": "Resolution (width & height) for each of the 4 rendered views."}),
                "background_color": ("STRING", {"default": "[0.1, 0.1, 0.1, 1.0]", "tooltip_prompt": "Background RGBA color as a string list, e.g., '[R, G, B, A]'. Values 0-1 or 0-255."}),
            },
            "optional": {
                "face2label_path": ("STRING", {"default": "", "forceInput": True, "tooltip_prompt": "Optional path to face2label.json. If provided and 'force_segment_colors' is true, segments will be colored."}),
                "force_segment_colors": ("BOOLEAN", {"default": True, "tooltip_prompt": "If checked and 'face2label_path' is provided and valid, this node will render the mesh with distinct colors for each segment label, overriding any original texture or vertex colors. Useful for visualizing segments."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_views",)
    FUNCTION = "render_views"
    CATEGORY = "SamMesh/Utils"

    def _parse_color(self, color_str: str, default=(0.1, 0.1, 0.1, 1.0)):
        try:
            color = json.loads(color_str)
            if isinstance(color, list) and len(color) == 4 and all(isinstance(x, (int, float)) for x in color):
                if any(x > 1.0 for x in color):
                    return [max(0.0, min(255.0, x)) / 255.0 for x in color]
                else:
                    return [max(0.0, min(1.0, x)) for x in color]
            else:
                print(f" [93mWarning: Invalid background_color format '{color_str}'. Using default. [0m")
                return list(default)
        except Exception:
            print(f" [93mWarning: Could not parse background_color string '{color_str}'. Using default. [0m")
            return list(default)


    def render_views(self, mesh: trimesh.Trimesh, render_resolution: int, background_color: str, face2label_path: str = "", force_segment_colors: bool = True):
        if pyrender is None:
            raise ImportError("pyrender is required for SamMeshRenderer but was not found.")

        bg_color = self._parse_color(background_color)
        
        working_mesh = mesh # Start with the input mesh
        is_colored_copy = False # Flag to indicate if segment colors were applied

        if force_segment_colors and face2label_path and os.path.exists(face2label_path):
            print(f"SamMeshRenderer: Attempting to apply segment colors from {face2label_path}")
            try:
                mesh_copy = copy.deepcopy(mesh) # Work on a copy

                with open(face2label_path, 'r') as f:
                    face2label_data = json.load(f)
                
                # Ensure face indices in JSON are integers
                face_labels_map = {int(k): int(v) for k, v in face2label_data.items()}
                
                if not mesh_copy.faces.shape[0] == 0 : # Check if mesh has faces
                    unique_labels = sorted(list(set(val for val in face_labels_map.values() if val is not None))) # Filter out None labels if any
                    
                    # Generate a color for each unique label
                    # Ensure we have enough colors if many labels exist; trimesh.visual.random_color might recycle for large counts.
                    # Using a simple colormap approach for more distinct colors if many labels.
                    if not unique_labels: # No labels found in the map
                        print("SamMeshRenderer: Warning - No valid labels found in face2label_path. Using original mesh.")
                    else:
                        label_to_color_idx_map = {label: i for i, label in enumerate(unique_labels)}
                        num_unique_labels = len(unique_labels)
                        
                        # Use trimesh.visual.random_color() to generate distinct colors
                        # This is more compatible with older trimesh versions than interpolate_rgba
                        colors_for_labels = [trimesh.visual.random_color() for _ in range(num_unique_labels)]
                                                
                        default_face_color = np.array([128, 128, 128, 255], dtype=np.uint8) # Grey for faces not in map or unlabelled
                        
                        new_face_colors = np.tile(default_face_color, (len(mesh_copy.faces), 1))

                        for face_idx, label_val in face_labels_map.items():
                            if label_val is not None and face_idx < len(mesh_copy.faces):
                                color_idx = label_to_color_idx_map.get(label_val)
                                if color_idx is not None:
                                    new_face_colors[face_idx] = colors_for_labels[color_idx]
                                else:
                                    # This case should ideally not be hit if face_labels_map values are all in unique_labels
                                    print(f"SamMeshRenderer: Warning - Label {label_val} for face {face_idx} not in unique_labels map.")
                            elif face_idx >= len(mesh_copy.faces):
                                print(f"SamMeshRenderer: Warning - Face index {face_idx} from JSON is out of bounds for mesh face count {len(mesh_copy.faces)}.")


                        # Ensure the visual type is ColorVisuals to use face/vertex colors
                        if not isinstance(mesh_copy.visual, trimesh.visual.ColorVisuals):
                            mesh_copy.visual = trimesh.visual.ColorVisuals()
                        
                        mesh_copy.visual.face_colors = new_face_colors
                        working_mesh = mesh_copy # Use the colored copy
                        is_colored_copy = True # Set the flag
                        print(f"SamMeshRenderer: Applied segment colors to {len(face_labels_map)} mapped faces.")
                else:
                    print("SamMeshRenderer: Warning - Input mesh for segment coloring has no faces.")
            except Exception as e:
                print(f"SamMeshRenderer: Warning - Failed to apply segment colors: {e}. Rendering original mesh.")
                # working_mesh remains original 'mesh' in case of error

        # If segment colors were applied, render non-smooth (flat), otherwise smooth.
        # pyrender.Mesh.from_trimesh should handle interpreting working_mesh.visual
        # (vertex colors if is_colored_copy, or original texture/material otherwise)
        render_mesh = pyrender.Mesh.from_trimesh(working_mesh, smooth=not is_colored_copy)
        print(f"SamMeshRenderer: Created pyrender.Mesh. is_colored_copy={is_colored_copy}, smooth={not is_colored_copy}")
        
        # --- Pyrender Primitive Appearance Setup (Simplified) ---
        # Trust pyrender.Mesh.from_trimesh to set up colors/materials based on working_mesh.visual.
        # Add a fallback default material if the primitive ends up with no appearance.
        if render_mesh.primitives:
            primitive = render_mesh.primitives[0]
            if hasattr(primitive, 'color_0') and primitive.color_0 is not None:
                print(f"SamMeshRenderer: pyrender.Mesh primitive has vertex colors (color_0). is_colored_copy={is_colored_copy}")
            elif hasattr(primitive, 'material') and primitive.material is not None:
                print(f"SamMeshRenderer: pyrender.Mesh primitive has a material: {primitive.material.name}. is_colored_copy={is_colored_copy}")
            else:
                print("SamMeshRenderer: pyrender.Mesh.from_trimesh resulted in no colors or material. Applying default pyrender material.")
                primitive.material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[0.8, 0.8, 0.8, 1.0], # Default grey
                    metallicFactor=0.2,
                    roughnessFactor=0.6
                )
        else:
            print("SamMeshRenderer [Warning]: Mesh has no primitives after pyrender.Mesh.from_trimesh.")

        scene = pyrender.Scene(bg_color=bg_color[:3], ambient_light=[0.2, 0.2, 0.2]) # Slightly brighter ambient
        mesh_node = scene.add(render_mesh, pose=np.eye(4), name="mesh")

        bounds = mesh.bounds
        if bounds is None:
             print(" [93mWarning: Mesh bounds could not be determined. Using default camera distance. [0m")
             center = [0,0,0]
             distance = 2.0
        else:
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            distance = scale * 1.5

        aspect_ratio = 1.0
        yfov = np.pi / 4.0
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
        camera_node = scene.add(camera, pose=np.eye(4))

        def look_at(eye, target, up):
            forward = np.subtract(target, eye)
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                 if abs(forward[1]) < 0.99:
                     right = np.cross(forward, [0, 1, 0])
                 else:
                     right = np.cross(forward, [1, 0, 0])

            right = right / np.linalg.norm(right)
            new_up = np.cross(right, forward)
            new_up = new_up / np.linalg.norm(new_up)

            cam_to_world = np.eye(4)
            cam_to_world[0, :3] = right
            cam_to_world[1, :3] = new_up
            cam_to_world[2, :3] = -forward
            cam_to_world[:3, 3] = eye
            return cam_to_world


        z_up = [0, 0, 1]
        poses = {
             "front": look_at(center + [0, -distance * 1.2, distance * 0.4], center, z_up),
             "right": look_at(center + [distance * 1.2, 0, distance * 0.4], center, z_up),
             "top":   look_at(center + [0, 0, distance * 1.5], center, [0, 1, 0]),
             "back":  look_at(center + [0, distance * 1.2, distance * 0.4], center, z_up),
        }


        renderer = pyrender.OffscreenRenderer(render_resolution, render_resolution)
        rendered_images = {}
        
        # Store the directional light node to remove/update it
        light_node = None

        view_keys = ["front", "right", "top", "back"]
        for key in view_keys:
            try:
                # Remove previous directional light if it exists
                if light_node is not None and scene.has_node(light_node):
                    scene.remove_node(light_node)
                
                # Add new directional light aligned with the camera
                # The light's pose is the same as the camera's pose for that view
                directional_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5) # Increased intensity
                light_node = scene.add(directional_light, pose=poses[key], name=f"light_for_{key}")
                
                scene.set_pose(camera_node, poses[key])
                color, depth = renderer.render(scene)
                rendered_images[key] = Image.fromarray(color, 'RGB')
            except Exception as e:
                 print(f" [91mError rendering view {key}: {e} [0m")
                 import traceback
                 traceback.print_exc()
                 rendered_images[key] = Image.new('RGB', (render_resolution, render_resolution), (50, 50, 50))


        grid_size = 1024
        img_per_view = grid_size // 2
        final_image = Image.new('RGB', (grid_size, grid_size))

        img_front = rendered_images["front"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_right = rendered_images["right"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_top = rendered_images["top"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_back = rendered_images["back"].resize((img_per_view, img_per_view), Image.LANCZOS)

        final_image.paste(img_front, (0, 0))
        final_image.paste(img_right, (img_per_view, 0))
        final_image.paste(img_top, (0, img_per_view))
        final_image.paste(img_back, (img_per_view, img_per_view))

        renderer.delete()

        image_np = np.array(final_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (image_tensor,)


# Add more nodes as needed
# Mapping dictionary for node display names
NODE_CLASS_MAPPINGS = {
    "SamModelDownloader+SamMesh": SamModelDownloader,
    "SamMeshLoader+SamMesh": SamMeshLoader,
    "SamMeshSegmenter+SamMesh": SamMeshSegmenter,
    "SamMeshExporter+SamMesh": SamMeshExporter,
    "SamMeshRenderer+SamMesh": SamMeshRenderer,
}

# Mapping dictionary for node human-readable names
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamModelDownloader+SamMesh": "SAM Model Downloader (SamMesh)",
    "SamMeshLoader+SamMesh": "Load Mesh (SamMesh)",
    "SamMeshSegmenter+SamMesh": "Segment Mesh with SAM (SamMesh)",
    "SamMeshExporter+SamMesh": "Export Segmented Mesh for HoloPart (SamMesh)",
    "SamMeshRenderer+SamMesh": "Render Mesh Views (SamMesh)",
} 