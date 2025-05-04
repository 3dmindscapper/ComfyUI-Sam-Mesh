import sys
import os
import inspect

# Add the samesh source directory to the Python path
# This allows importing modules from samesh like 'from samesh.data.loaders import read_mesh'
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
samesh_src_dir = os.path.join(current_dir, "samesh-main", "src")
# Also add the segment-anything-2 directory itself, as samesh likely imports 'sam2' directly
samesh_third_party_dir = os.path.join(current_dir, "samesh-main", "third_party", "segment-anything-2")

if samesh_src_dir not in sys.path:
    sys.path.insert(0, samesh_src_dir)
    print(f"ComfyUI-Sam-Mesh: Added {samesh_src_dir} to sys.path")

# Add the third_party sam2 directory if it's not already there
if samesh_third_party_dir not in sys.path:
     sys.path.insert(0, samesh_third_party_dir) # Add it with high priority
     print(f"ComfyUI-Sam-Mesh: Added {samesh_third_party_dir} to sys.path for sam2 import")

# Check for submodule presence (optional but helpful)
if samesh_src_dir and not os.path.exists(os.path.join(samesh_third_party_dir, "sam2")):
     print(f"\033[93mWarning: Potential Git submodule issue. Directory {samesh_third_party_dir} seems incomplete or missing.\033[0m")
     print(f"\033[93mIf you encounter import errors related to 'sam2', try running 'git submodule update --init --recursive' in the ComfyUI-Sam-Mesh/samesh-main directory.\033[0m")

from .nodes import SamMeshLoader, SamMeshSegmenter, SamModelDownloader, SamMeshExporter, SamMeshRenderer

# Mapping node class names to node display names
NODE_CLASS_MAPPINGS = {
    "SamMeshLoader": SamMeshLoader,
    "SamMeshSegmenter": SamMeshSegmenter,
    "SamModelDownloader": SamModelDownloader,
    "SamMeshExporter": SamMeshExporter,
    "SamMeshRenderer": SamMeshRenderer,
}

# Mapping node class names to human-readable names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamMeshLoader": "Load SamMesh",
    "SamMeshSegmenter": "Segment SamMesh",
    "SamModelDownloader": "(Down)load SAM Model (SamMesh)",
    "SamMeshExporter": "Export SamMesh Segments",
    "SamMeshRenderer": "Render SamMesh Views",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("\033[34mComfyUI-Sam-Mesh: \033[92mLoaded\033[0m") 

# Tell ComfyUI where to look for web files (JS)
WEB_DIRECTORY = "js" 