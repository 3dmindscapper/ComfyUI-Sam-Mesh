# ComfyUI-Sam-Mesh

This project integrates the Segment Any Mesh (SAMesh) library into ComfyUI, providing a set of custom nodes for performing zero-shot 3D mesh part segmentation using SAM2 (Segment Anything Model 2). 
It allows users to load meshes, segment them into parts, and visualize or export the results directly within a ComfyUI workflow.

# Example Images
![Image](https://github.com/user-attachments/assets/5b7dcf1f-659e-4133-8b1a-71328a82d926)

![Image](https://github.com/user-attachments/assets/283fe77e-c0ef-4fcf-85f3-15f1c9b45054)

## Features

Segment 3D meshes into distinct parts using the power of SAM2.
Download required SAM2 Hiera models directly within ComfyUI.
Load 3D meshes from common file formats.
Options to preprocess meshes (normalization, Trimesh processing).
Attempt to preserve original textures on segmented parts.
Export segmented meshes as a single GLB file, with each segment as a separate object (suitable for HoloPart or other 3D applications).
Render multi-view previews (front, right, top, back) of meshes.
Option to visualize segments with distinct colors in the rendered views.

## Installation

Clone this Repository:
Navigate to your ComfyUI/custom_nodes/ directory and clone this repository:
Apply to README.md
Run
(The samesh library and its SAM2 dependency are included within the samesh-main directory of this project.)

2. Install Dependencies

Key dependencies include:
torch and torchvision (usually already present in a ComfyUI environment)
trimesh
pyrender
omegaconf
huggingface-hub
requests
Pillow (PIL)
numpy
tqdm
Other packages required by samesh and SAM2.
Ensure your ComfyUI Python environment is active if you are using virtual environments.

3. Pyrender System Dependencies (for Renderer Node):

The Render Mesh Views node uses pyrender. For headless rendering (common in ComfyUI setups, especially on servers), pyrender might require additional system libraries.
On Linux, you might need to install EGL or OSMesa. For example:
Apply to README.md
Run
Consult the pyrender installation guide for more details specific to your operating system if you encounter issues with the renderer node (e.g., "Failed to create context" errors).
Note from samesh README: If you encounter pyrender issues related to ctypes, try installing PyOpenGL==3.1.7.

4. Restart ComfyUI: After installation, restart ComfyUI to ensure the custom nodes are loaded.

## Nodes Overview

Here's a brief overview of the nodes provided:

1. SAM Model Downloader (SamMesh)
Purpose: Downloads SAM2 Hiera model checkpoints and their corresponding configuration YAML files. These are required by the segmentation node.
Inputs:
model_name: (Dropdown) Select the SAM2 Hiera model to download (e.g., Large, Base+, Small, Tiny).
Outputs:
sam_checkpoint_path: Path to the downloaded SAM model checkpoint file.
sam_model_config_path: Path to the downloaded SAM model configuration file.

2. Load Mesh (SamMesh)
Purpose: Loads 3D mesh files from your ComfyUI/input/ directory.
Inputs:
mesh: (Dropdown) Select a mesh file from the ComfyUI input directory.
normalize_mesh: (Boolean, default: False)
Tooltip: Scales and translates the mesh to fit within a unit cube centered at the origin. Useful for standardizing mesh size for consistent SAM processing, but can alter original scale.
process_mesh: (Boolean, default: True)
Tooltip: Enables Trimesh's default processing (e.g., removing duplicate vertices, merging close vertices, fixing face winding). Can help clean up messy geometry but may alter vertex/UV data or topology. Disable if precise original structure is critical, especially with textures.
Outputs:
sam_mesh: The loaded mesh as a Trimesh object.
loaded_mesh_path: The absolute file path of the loaded mesh.

3. Segment Mesh with SAM (SamMesh)
Purpose: Performs the core 3D mesh segmentation using the SAMesh algorithm and a selected SAM model.
Inputs:
sam_mesh: The Trimesh object from the Load Mesh node.
mesh_path: The file path of the original mesh (output from Load Mesh).
sam_checkpoint_path: Path to the SAM checkpoint (from SAM Model Downloader).
sam_model_config_path: Path to the SAM model config (from SAM Model Downloader).
output_directory: Directory to save intermediate segmentation files.
cache_directory: Directory for samesh to cache rendering or computation results.
keep_texture: (Boolean, default: False)
Tooltip: If checked, the segmentation process will attempt to preserve the original mesh's texture and apply it to the segmented parts. UV mapping quality depends on the samesh library's capabilities.
target_labels: (Integer, default: -1)
Tooltip: Desired number of distinct segments. Set to -1 for the samesh library's default/automatic behavior. Higher numbers may result in finer-grained segmentation.
seed: (Integer, default: 0) Random seed for reproducible segmentation results.
Outputs:
segmented_mesh: The segmented mesh as a Trimesh object. If keep_texture was successful, this mesh will be textured; otherwise, it's typically colored by segment ID by the samesh library.
face2label_path: Path to a JSON file mapping each face index of the original mesh to a segment label.

4. Export Segmented Mesh for HoloPart (SamMesh)
Purpose: Exports the segmented mesh as a single GLB file. Each segment is saved as a distinct object within the GLB scene, which can be useful for applications like HoloPart or for importing into 3D modeling software.
Inputs:
segmented_mesh: The Trimesh object from the Segment Mesh node.
face2label_path: The path to the face-to-label JSON file (from Segment Mesh).
output_filename: The desired name for the exported GLB file (e.g., my_segmented_model.glb). It will be saved in your ComfyUI/output/ directory.
Outputs: (This is an output node, so it saves a file but doesn't pass on data via connections).

5. Render Mesh Views (SamMesh)
Purpose: Renders four standard views (front, right, top, back) of an input mesh and combines them into a single image.
Inputs:
mesh: The Trimesh object to render.
render_resolution: (Integer, default: 1024) The resolution (width & height) for each of the 4 views. The final image will be render_resolution*2 x render_resolution*2.
background_color: (String, e.g., [0.1, 0.1, 0.1, 1.0]) RGBA color for the background.
face2label_path: (String, optional) Path to a face2label.json file.
force_segment_colors: (Boolean, default: True)
Tooltip: If checked and face2label_path is provided and valid, this node will render the mesh with distinct colors for each segment label, overriding any original texture or vertex colors. Useful for visualizing segments.

## Outputs:
rendered_views: An IMAGE output suitable for previewing or further processing in ComfyUI.
Example Workflow
(You can add a screenshot of your nodes connected in a typical workflow here.)
A basic workflow would be:
SAM Model Downloader -> Segment Mesh with SAM (checkpoint and config paths)
Load Mesh -> Segment Mesh with SAM (sam_mesh and mesh_path)
Segment Mesh with SAM -> Export Segmented Mesh (segmented_mesh and face2label_path)
Segment Mesh with SAM (segmented_mesh) AND Segment Mesh with SAM (face2label_path) -> Render Mesh Views (to visualize segments if force_segment_colors is on).

## Troubleshooting
Submodule Not Found / samesh Import Errors: Ensure you have correctly initialized the Git submodule by running git submodule update --init --recursive from the ComfyUI-Sam-Mesh directory.
Render Mesh Views Node Fails or Shows Black Images:
This node requires pyrender. Ensure it's installed correctly via requirements.txt.
You may need to install system libraries for headless OpenGL rendering (like EGL or OSMesa on Linux). See the pyrender documentation.
Texture Issues with keep_texture: The success of texture preservation depends on the samesh library's capabilities and the input mesh's UV layout. Complex UVs or meshes heavily altered by segmentation might not retain textures perfectly.

## License

# MIT License

# Copyright (c) 2025 VAST-AI-Research and contributors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

##Acknowledgements
This project is a wrapper around the powerful Segment Any Mesh (SAMesh) library. All credit for the core segmentation algorithm and its implementation goes to the original SAMesh authors: George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang.
Please refer to the SAMesh GitHub repository and their research paper for more details on the underlying technology.
