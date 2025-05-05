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
                 "model_name": (SAM_MODEL_NAMES, {"default": SAM_MODEL_NAMES[0]}),
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
                # Use file_upload=True for ComfyUI file dialog
                # "mesh_path": ("STRING", {"default": "", "file_upload": True}),
                "mesh": (sorted(files) if files else ["No mesh files found"], ),
            },
            "optional": {
                "normalize_mesh": ("BOOLEAN", {"default": False}),
                "process_mesh": ("BOOLEAN", {"default": True}),
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
    Segments a mesh using the SAMesh model.
    Requires paths to SAM model files (use SamModelDownloader) and embeds default samesh configuration.
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
                "output_directory": ("STRING", {"default": DEFAULT_OUTPUT_DIR}),
                "cache_directory": ("STRING", {"default": DEFAULT_CACHE_DIR}),
                "output_filename_prefix": ("STRING", {"default": "segmented"}),
                "visualize": ("BOOLEAN", {"default": False}),
                "output_extension": (["glb", "obj", "ply"], {"default": "glb"}),
                "keep_texture": ("BOOLEAN", {"default": False}),
                # --- Samesh SAM Engine Config Overrides ---
                "sam_points_per_side": ("INT", {"default": 32, "min": 8, "max": 128, "step": 4}),
                "sam_pred_iou_thresh": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.01}),
                "sam_stability_score_thresh": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 0.99, "step": 0.01}),
                "sam_stability_score_offset": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                # --- Samesh Mesh Processing Config Overrides ---
                "samesh_min_area": ("INT", {"default": 1024, "min": 0, "max": 100000, "step": 64}),
                "samesh_connections_bin_resolution": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),
                "samesh_connections_bin_thresh_perc": ("FLOAT", {"default": 0.125, "min": 0.01, "max": 0.5, "step": 0.005}),
                "samesh_smoothing_thresh_perc_size": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.2, "step": 0.005}),
                "samesh_smoothing_thresh_perc_area": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.2, "step": 0.005}),
                "samesh_smoothing_iterations": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "samesh_repartition_lambda": ("INT", {"default": 6, "min": 0, "max": 20, "step": 1}),
                "samesh_repartition_iterations": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {
                 "target_labels": ("INT", {"default": -1, "min": -1, "max": 10000}), # -1 means disable
            }
        }

    RETURN_TYPES = ("SAM_MESH", "STRING", "STRING",)
    RETURN_NAMES = ("segmented_mesh", "output_mesh_path", "face2label_path",)
    FUNCTION = "segment_mesh"
    CATEGORY = "SamMesh"

    def segment_mesh(self, sam_mesh: trimesh.Trimesh, mesh_path: str,
                     sam_checkpoint_path: str, sam_model_config_path: str,
                     output_directory: str, cache_directory: str,
                     output_filename_prefix: str, visualize: bool, output_extension: str,
                     keep_texture: bool, target_labels: int = -1,
                     # Optional config overrides
                     sam_points_per_side: int = 32,
                     sam_pred_iou_thresh: float = 0.5,
                     sam_stability_score_thresh: float = 0.7,
                     sam_stability_score_offset: float = 1.0,
                     samesh_min_area: int = 1024,
                     samesh_connections_bin_resolution: int = 100,
                     samesh_connections_bin_thresh_perc: float = 0.125,
                     samesh_smoothing_thresh_perc_size: float = 0.025,
                     samesh_smoothing_thresh_perc_area: float = 0.025,
                     samesh_smoothing_iterations: int = 64,
                     samesh_repartition_lambda: int = 6,
                     samesh_repartition_iterations: int = 1,
                     ):

        # --- Input Validation ---
        if not isinstance(sam_mesh, trimesh.Trimesh):
            print(f"Warning: Input 'sam_mesh' is not a Trimesh object (got {type(sam_mesh)}). Attempting to proceed, but errors may occur.")
        # if segment_mesh_samesh_func is None or read_mesh is None or remove_texture is None or colormap_faces_mesh is None:
        #      raise ImportError("Required samesh functions are not available. Check imports and submodule.") # Less critical now as worker handles it
        if not mesh_path or not os.path.exists(mesh_path):
             # Ensure mesh_path is absolute before passing to worker
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
        cmd = [
            sys.executable, # Use the same python executable that runs ComfyUI
            WORKER_SCRIPT_PATH,
            "--mesh_path", mesh_path,
            "--sam_checkpoint_path", sam_checkpoint_path,
            "--sam_model_config_path", sam_model_config_path,
            "--output_directory", output_directory,
            "--cache_directory", cache_directory,
            "--output_filename_prefix", output_filename_prefix,
            "--output_extension", output_extension,
            "--target_labels", str(target_labels),
            # Config overrides
            "--sam_points_per_side", str(sam_points_per_side),
            "--sam_pred_iou_thresh", str(sam_pred_iou_thresh),
            "--sam_stability_score_thresh", str(sam_stability_score_thresh),
            "--sam_stability_score_offset", str(sam_stability_score_offset),
            "--samesh_min_area", str(samesh_min_area),
            "--samesh_connections_bin_resolution", str(samesh_connections_bin_resolution),
            "--samesh_connections_bin_thresh_perc", str(samesh_connections_bin_thresh_perc),
            "--samesh_smoothing_thresh_perc_size", str(samesh_smoothing_thresh_perc_size),
            "--samesh_smoothing_thresh_perc_area", str(samesh_smoothing_thresh_perc_area),
            "--samesh_smoothing_iterations", str(samesh_smoothing_iterations),
            "--samesh_repartition_lambda", str(samesh_repartition_lambda),
            "--samesh_repartition_iterations", str(samesh_repartition_iterations),
        ]
        if visualize:
            cmd.append("--visualize")
        if keep_texture:
            cmd.append("--keep_texture")

        # --- Prepare Environment for Subprocess ---
        # Pass ComfyUI base path to worker environment for dependency resolution if needed
        env = os.environ.copy()
        try:
             comfy_base = folder_paths.base_path
             env['COMFYUI_BASE_PATH'] = comfy_base
             # Add ComfyUI base and its parent to PYTHONPATH for the worker, just in case
             python_path = env.get('PYTHONPATH', '')
             paths_to_add = [comfy_base, os.path.dirname(comfy_base)]
             # Also add node's samesh paths explicitly if needed, though worker script does it too
             # paths_to_add.append(os.path.join(comfyui_samesh_node_dir, "samesh-main", "src"))
             # paths_to_add.append(os.path.join(comfyui_samesh_node_dir, "samesh-main", "third_party", "segment-anything-2"))
             env['PYTHONPATH'] = os.pathsep.join([p for p in paths_to_add if p] + [python_path])

        except AttributeError:
             print("Warning: Could not get ComfyUI base path via folder_paths.base_path. Path resolution inside worker might fail.")
             env['COMFYUI_BASE_PATH'] = '' # Send empty string

        # --- Execute Worker Script ---
        print(f"SamMeshSegmenter: Executing worker: {' '.join(cmd)}")
        # print(f"SamMeshSegmenter: Worker PYTHONPATH: {env.get('PYTHONPATH')}")
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            print(f"SamMeshSegmenter: Worker stdout:\n{process.stdout}")
            if process.stderr:
                 print(f"SamMeshSegmenter: Worker stderr:\n{process.stderr}")

            # --- Parse result from worker stdout ---
            # Worker script prints the JSON result as the last line of stdout
            last_line = process.stdout.strip().splitlines()[-1]
            result_data = json.loads(last_line)
            final_output_mesh_path = result_data.get("output_mesh_path")
            final_output_json_path = result_data.get("face2label_path")

            if not final_output_mesh_path or "Error:" in final_output_mesh_path:
                 raise RuntimeError(f"Worker script failed to produce mesh output: {final_output_mesh_path}")
            if not final_output_json_path or "Error:" in final_output_json_path:
                 raise RuntimeError(f"Worker script failed to produce JSON output: {final_output_json_path}")

            print(f"SamMeshSegmenter: Worker finished successfully. Output mesh: {final_output_mesh_path}, Output JSON: {final_output_json_path}")

            # --- Load the resulting mesh --- #
            # The node needs to return the trimesh object
            print(f"SamMeshSegmenter: Loading result mesh from {final_output_mesh_path}...")
            if read_mesh is None:
                raise ImportError("samesh.data.loaders.read_mesh function is not available to load the result.")
            # Use basic trimesh loading; samesh read_mesh might do extra processing we don't want here
            # segmented_colored_mesh = read_mesh(final_output_mesh_path, norm=False, process=False) 
            segmented_colored_mesh = trimesh.load(final_output_mesh_path, force='mesh') # Use standard trimesh loading
            if not isinstance(segmented_colored_mesh, trimesh.Trimesh):
                 raise TypeError(f"Loaded result mesh is not a Trimesh object (got {type(segmented_colored_mesh)}) from path: {final_output_mesh_path}")

            print(f"SamMeshSegmenter: Result mesh loaded.")

            return (segmented_colored_mesh, final_output_mesh_path, final_output_json_path)

        except subprocess.CalledProcessError as e:
            print(f"\033[31mError: Worker script failed (return code {e.returncode}).\033[0m")
            print(f"\033[31mCommand: {' '.join(e.cmd)}\033[0m")
            print(f"\033[31mStderr:\n{e.stderr}\033[0m")
            print(f"\033[31mStdout:\n{e.stdout}\033[0m")
            raise RuntimeError(f"Segmentation worker script failed. Check logs for details.") from e
        except json.JSONDecodeError as e:
            print(f"\033[31mError: Could not parse JSON result from worker script stdout.\033[0m")
            print(f"\033[31mStdout:\n{process.stdout}\033[0m") # Print stdout for debugging
            raise RuntimeError(f"Failed to parse result from segmentation worker.") from e
        except Exception as e:
            print(f"\033[31mError during segmentation subprocess execution or result loading: {e}\033[0m")
            import traceback
            traceback.print_exc()
            raise

# --- SamMeshExporter Node ---
class SamMeshExporter:
    """
    Exports the segmented mesh, splitting it into parts based on segment labels.
    Outputs individual mesh files into a specified subdirectory within the ComfyUI output folder.
    """
    RETURN_TYPES = () # No return value needed for exporter nodes
    FUNCTION = "export_parts"
    OUTPUT_NODE = True # Mark as output node
    CATEGORY = "SamMesh/Export"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmented_mesh": ("SAM_MESH",), # The mesh with vertex/face colors or metadata if samesh provides it
                "face2label_path": ("STRING", {"forceInput": True}), # Path to the JSON file mapping faces to labels
                "output_directory_name": ("STRING", {"default": "exported_segments"}), # Name for the output subfolder
                "output_format": (["obj", "glb", "ply"], {"default": "obj"}), # Format for exported parts
            }
        }

    def export_parts(self, segmented_mesh: trimesh.Trimesh, face2label_path: str, output_directory_name: str, output_format: str):
        if not os.path.exists(face2label_path):
            raise FileNotFoundError(f"Face-to-label JSON file not found: {face2label_path}")

        output_base_dir = folder_paths.get_output_directory()
        export_dir = os.path.join(output_base_dir, output_directory_name)
        os.makedirs(export_dir, exist_ok=True)
        print(f"SamMeshExporter: Exporting segments to: {export_dir}")

        try:
            with open(face2label_path, 'r') as f:
                face2label = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load or parse face2label JSON from {face2label_path}: {e}")

        # Ensure face indices in JSON are integers (sometimes they might be loaded as strings)
        try:
             # Convert keys (face indices) to integers and values (labels) to integers
            face_labels = {int(k): int(v) for k, v in face2label.items()}
            unique_labels = sorted(list(set(face_labels.values())))
            print(f"SamMeshExporter: Found {len(unique_labels)} unique segment labels.")
        except Exception as e:
             raise ValueError(f"Error processing face labels from JSON (ensure keys/values are integers): {e}")


        base_mesh = segmented_mesh # Use the input mesh directly

        for label in unique_labels:
            # Find faces belonging to the current label
            faces_for_label_mask = [idx for idx, lbl in face_labels.items() if lbl == label]

            if not faces_for_label_mask:
                 print(f"SamMeshExporter: Warning - No faces found for label {label}. Skipping.")
                 continue

            # Create a submesh for the current label
            # We need the original vertices and only the faces for this label
            try:
                # Trimesh face indexing can be tricky. Ensure mask is usable.
                # We need the *indices* of the faces in the original mesh that match the label
                face_indices_for_label = np.array(faces_for_label_mask, dtype=np.int64)

                # Select the faces from the base mesh
                segment_faces = base_mesh.faces[face_indices_for_label]

                # Create the new mesh. We need to re-index vertices.
                # `trimesh.Trimesh` constructor with only faces doesn't work directly like this.
                # We need to get the unique vertices used by these faces and the new face indices relative to those vertices.
                unique_vertex_indices = np.unique(segment_faces)
                segment_vertices = base_mesh.vertices[unique_vertex_indices]

                # Map old vertex indices (in base_mesh) to new indices (in segment_vertices)
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

                # Create new faces with updated vertex indices
                new_segment_faces = np.array([
                    [vertex_map[v_idx] for v_idx in face]
                    for face in segment_faces
                ], dtype=np.int64)

                # Create the final segment mesh
                segment_mesh = trimesh.Trimesh(vertices=segment_vertices, faces=new_segment_faces)

                # Check if the original mesh had vertex colors and copy them if possible
                if hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'vertex_colors') and len(base_mesh.visual.vertex_colors) == len(base_mesh.vertices):
                     segment_mesh.visual = trimesh.visual.ColorVisuals(
                         mesh=segment_mesh,
                         vertex_colors=base_mesh.visual.vertex_colors[unique_vertex_indices]
                     )
                # Check for face colors (less common but possible)
                # elif hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'face_colors') and len(base_mesh.visual.face_colors) == len(base_mesh.faces):
                #     segment_mesh.visual = trimesh.visual.ColorVisuals(
                #         mesh=segment_mesh,
                #         face_colors=base_mesh.visual.face_colors[face_indices_for_label] # Use the original face indices mask
                #      )


                output_filename = f"segment_{label}.{output_format}"
                output_path = os.path.join(export_dir, output_filename)

                # Export the submesh
                segment_mesh.export(output_path, file_type=output_format)
                print(f"SamMeshExporter: Exported {output_path}")

            except Exception as e:
                print(f"[91mSamMeshExporter: Error processing or exporting label {label}: {e}[0m")
                import traceback
                traceback.print_exc()
                # Continue to next label if one fails

        print(f"SamMeshExporter: Finished exporting segments.")
        return {} # Must return empty dict for output nodes


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
                "mesh": ("SAM_MESH",), # Input mesh object (can be segmented or original)
                "render_resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}), # Resolution per view
                "background_color": ("STRING", {"default": "[0.1, 0.1, 0.1, 1.0]"}), # RGBA as string list e.g. "[R, G, B, A]"
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
                # Normalize to 0-1 range if needed (e.g., if user enters 0-255)
                if any(x > 1.0 for x in color):
                    return [max(0.0, min(255.0, x)) / 255.0 for x in color]
                else:
                    return [max(0.0, min(1.0, x)) for x in color]
            else:
                print(f"[93mWarning: Invalid background_color format '{color_str}'. Using default.[0m")
                return list(default)
        except Exception:
            print(f"[93mWarning: Could not parse background_color string '{color_str}'. Using default.[0m")
            return list(default)


    def render_views(self, mesh: trimesh.Trimesh, render_resolution: int, background_color: str):
        if pyrender is None:
            raise ImportError("pyrender is required for SamMeshRenderer but was not found.")

        bg_color = self._parse_color(background_color)

        # --- Prepare Mesh for Pyrender ---
        # Ensure mesh has vertex colors if possible, otherwise use default material
        if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
             # Use vertex colors directly if they exist and seem valid
             # Ensure they are RGBA uint8
            colors = mesh.visual.vertex_colors
            if colors.shape[1] == 3: # If RGB, add alpha channel
                 colors = np.hstack((colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)))
            elif colors.dtype != np.uint8:
                 colors = (colors * 255).astype(np.uint8)

            render_mesh = pyrender.Mesh.from_trimesh(mesh, vertex_colors=colors, smooth=True)
            material = None # Material is derived from vertex colors
        else:
            # Use a default PBR material if no vertex colors
             render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
             material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.8, 0.8, 1.0], # Light grey
                metallicFactor=0.2,
                roughnessFactor=0.6
            )

        # --- Setup Scene ---
        scene = pyrender.Scene(bg_color=bg_color[:3], ambient_light=[0.1, 0.1, 0.3]) # Use RGB for bg_color
        node = scene.add(render_mesh, pose=np.eye(4), name="mesh")
        if material: # Add default material only if not using vertex colors
             node.mesh.primitives[0].material = material


        # --- Camera Setup ---
        # Calculate camera distance based on mesh bounds
        bounds = mesh.bounds
        if bounds is None:
             print("[93mWarning: Mesh bounds could not be determined. Using default camera distance.[0m")
             center = [0,0,0]
             distance = 2.0
        else:
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            distance = scale * 2.0 # Adjust multiplier as needed

        aspect_ratio = 1.0 # Square renders
        yfov = np.pi / 4.0 # Field of view ~45 degrees
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)

        # Define 4 camera poses (adjust as needed)
        # Pyrender uses OpenGL convention (+Y up, -Z forward)
        # Trimesh uses +Z up convention for scene transforms usually
        # We might need rotations to align views correctly depending on mesh orientation

        # Example: look_at function helper (common in graphics)
        def look_at(eye, target, up):
            forward = np.subtract(target, eye)
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            new_up = np.cross(right, forward) # Recompute up to be orthogonal
            new_up = new_up / np.linalg.norm(new_up)

            cam_to_world = np.eye(4)
            cam_to_world[0, :3] = right
            cam_to_world[1, :3] = new_up
            cam_to_world[2, :3] = -forward # OpenGL convention: -Z is forward
            cam_to_world[:3, 3] = eye
            return cam_to_world # This is world_from_cam, pyrender wants cam_from_world = inv(world_from_cam)
            # return np.linalg.inv(cam_to_world) # Return camera's pose matrix (cam_from_world)

        # Camera positions relative to centroid
        eye_front = center + np.array([0, -distance, 0]) # Y- is front in trimesh view usually
        eye_back = center + np.array([0, distance, 0])   # Y+ is back
        eye_right = center + np.array([distance, 0, 0])  # X+ is right
        eye_left = center + np.array([-distance, 0, 0]) # X- is left (Use this for "right" view in grid?)
        eye_top = center + np.array([0, 0, distance])     # Z+ is top
        eye_bottom = center + np.array([0, 0, -distance]) # Z- is bottom

        # Choose 4 views: Front, Right (from X+), Top (from Z+), Back (from Y+) ?
        # The `look_at` matrix needs to be inverted for pyrender `scene.add(camera, pose=...)`
        # Alternatively, pyrender camera pose is directly the world_from_cam transform.
        # Let's try direct poses, assuming standard OpenGL camera (-Z forward, +Y up)

        # Front View (Looking along -Z in world, camera at +Z world) - Confusing!
        # Let's define poses manually, easier to reason about object orientation
        # Camera pose is transform matrix placing camera in world space
        poses = {
            "front": np.array([ # Camera looking towards -Y
                [1, 0, 0, center[0]],
                [0, 1, 0, center[1] - distance], # Position along -Y
                [0, 0, 1, center[2]],
                [0, 0, 0, 1],
            ]),
            "right": np.array([ # Camera looking towards -X (from +X)
                [0, 0, -1, center[0] + distance], # Position along +X
                [0, 1, 0, center[1]],
                [1, 0, 0, center[2]],
                [0, 0, 0, 1],
            ]),
             "top": np.array([ # Camera looking towards -Z (from +Z)
                [1, 0, 0, center[0]],
                [0, 0, 1, center[1]],          # Rotation needed here
                [0, -1, 0, center[2] + distance], # Position along +Z
                [0, 0, 0, 1],
            ]),
            "back": np.array([ # Camera looking towards +Y (from -Y -> error, should be from +Y)
                 [ -1, 0, 0, center[0]],         # Rotate 180 deg around Z
                 [ 0, -1, 0, center[1] + distance],# Position along +Y
                 [ 0, 0, 1, center[2]],
                 [ 0, 0, 0, 1],
             ]),

            # Let's recalculate using look_at and inverting, might be clearer
            # Assuming default UP is (0, 1, 0) initially, which is often NOT Z for meshes
            # Let's assume mesh UP is Z+ for calculation, then adjust if needed
             "front_pose": np.linalg.inv(look_at(center + [0, -distance, 0], center, [0, 0, 1])), # Look from -Y, Z up
             "right_pose": np.linalg.inv(look_at(center + [distance, 0, 0], center, [0, 0, 1])), # Look from +X, Z up
             "top_pose": np.linalg.inv(look_at(center + [0, 0, distance], center, [0, 1, 0])),    # Look from +Z, Y up (standard top-down)
             "back_pose": np.linalg.inv(look_at(center + [0, distance, 0], center, [0, 0, 1])),   # Look from +Y, Z up
        }

        # --- Render ---
        renderer = pyrender.OffscreenRenderer(render_resolution, render_resolution)
        rendered_images = {}

        view_keys = ["front_pose", "right_pose", "top_pose", "back_pose"]
        for key in view_keys:
            try:
                scene.set_pose(camera_node, poses[key])
                color, depth = renderer.render(scene) # Color is uint8 numpy array (H, W, 3)
                rendered_images[key] = Image.fromarray(color, 'RGB')
            except Exception as e:
                 print(f"[91mError rendering view {key}: {e}[0m")
                 # Create a blank image as placeholder
                 rendered_images[key] = Image.new('RGB', (render_resolution, render_resolution), (50, 50, 50))


        # --- Combine Images ---
        grid_size = 1024
        img_per_view = grid_size // 2 # 512
        final_image = Image.new('RGB', (grid_size, grid_size))

        # Resize rendered images if they don't match img_per_view (e.g., if render_resolution was different)
        img_front = rendered_images["front_pose"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_right = rendered_images["right_pose"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_top = rendered_images["top_pose"].resize((img_per_view, img_per_view), Image.LANCZOS)
        img_back = rendered_images["back_pose"].resize((img_per_view, img_per_view), Image.LANCZOS)

        # Paste into grid:
        # Top-Left: Front | Top-Right: Right
        # Bot-Left: Top   | Bot-Right: Back
        final_image.paste(img_front, (0, 0))
        final_image.paste(img_right, (img_per_view, 0))
        final_image.paste(img_top, (0, img_per_view))
        final_image.paste(img_back, (img_per_view, img_per_view))

        renderer.delete() # Release GPU resources

        # --- Convert to ComfyUI IMAGE format (Tensor) ---
        image_np = np.array(final_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,] # Add batch dimension

        return (image_tensor,)


# Add more nodes as needed 