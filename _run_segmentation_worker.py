import sys
import os
import json
import argparse
from pathlib import Path
import traceback

# Helper to add paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
samesh_base_dir = os.path.join(script_dir, "samesh-main")
samesh_src_dir = os.path.join(samesh_base_dir, "src")
sam2_dir = os.path.join(samesh_base_dir, "third_party", "segment-anything-2")

def add_to_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)

# Add samesh base, src and sam2 to path *before* importing them
add_to_path(samesh_base_dir) # Add the base directory
add_to_path(samesh_src_dir)
add_to_path(sam2_dir)

# Add ComfyUI base path if provided (for folder_paths resolution within samesh)
# This might be needed for implicit dependencies or path lookups within samesh
comfyui_base_path = os.environ.get("COMFYUI_BASE_PATH")
if comfyui_base_path:
    print(f"[SamMesh Worker Init] Adding COMFYUI_BASE_PATH: {comfyui_base_path}")
    add_to_path(comfyui_base_path)
    # Also try adding the parent of base_path, as sometimes imports expect that structure
    comfyui_parent_path = os.path.dirname(comfyui_base_path)
    print(f"[SamMesh Worker Init] Adding parent of COMFYUI_BASE_PATH: {comfyui_parent_path}")
    add_to_path(comfyui_parent_path)
else:
     print("[SamMesh Worker Init] COMFYUI_BASE_PATH not found in environment.")


# Now import samesh components
try:
    from omegaconf import OmegaConf
    from samesh.models.sam_mesh import segment_mesh as segment_mesh_samesh_func
    print(f"Worker: Successfully imported samesh modules.")
    # print(f"Worker sys.path: {sys.path}") # Debug path
except ImportError as e:
    print(f"Worker Error: Failed to import samesh components. Check paths and dependencies.", file=sys.stderr)
    print(f"Worker sys.path: {sys.path}", file=sys.stderr)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Worker Error: Unexpected error during import.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run SamMesh Segmentation")
    parser.add_argument("--mesh_path", required=True, help="Path to the input mesh file.")
    parser.add_argument("--sam_checkpoint_path", required=True, help="Path to the SAM checkpoint.")
    parser.add_argument("--sam_model_config_path", required=True, help="Path to the SAM model config YAML.")
    parser.add_argument("--output_directory", required=True, help="Directory to save outputs.")
    parser.add_argument("--cache_directory", required=True, help="Directory for caching.")
    parser.add_argument("--output_filename_prefix", required=True, help="Prefix for output filenames.")
    parser.add_argument("--visualize", action='store_true', help="Enable visualization.")
    parser.add_argument("--output_extension", required=True, choices=['glb', 'obj', 'ply'], help="Output mesh format.")
    parser.add_argument("--keep_texture", action='store_true', help="Keep original texture.")
    parser.add_argument("--target_labels", type=int, default=-1, help="Target number of labels (-1 to disable).")

    # Config overrides (make them optional, provide defaults matching the node if needed)
    # Defaults should ideally match those in the SamMeshSegmenter node's function signature
    parser.add_argument("--sam_points_per_side", type=int, default=32)
    parser.add_argument("--sam_pred_iou_thresh", type=float, default=0.5)
    parser.add_argument("--sam_stability_score_thresh", type=float, default=0.7)
    parser.add_argument("--sam_stability_score_offset", type=float, default=1.0)
    parser.add_argument("--samesh_min_area", type=int, default=1024)
    parser.add_argument("--samesh_connections_bin_resolution", type=int, default=100)
    parser.add_argument("--samesh_connections_bin_thresh_perc", type=float, default=0.125)
    parser.add_argument("--samesh_smoothing_thresh_perc_size", type=float, default=0.025)
    parser.add_argument("--samesh_smoothing_thresh_perc_area", type=float, default=0.025)
    parser.add_argument("--samesh_smoothing_iterations", type=int, default=64)
    parser.add_argument("--samesh_repartition_lambda", type=int, default=6)
    parser.add_argument("--samesh_repartition_iterations", type=int, default=1)

    args = parser.parse_args()

    print(f"Worker: Received arguments: {args}")

    mesh_file_path = Path(args.mesh_path)
    filename_stem = mesh_file_path.stem

    # --- Create configuration directly (similar to the node) ---
    print(f"Worker: Creating samesh config object.")
    try:
        config = OmegaConf.create({
            "cache": args.cache_directory,
            "cache_overwrite": False,
            "output": args.output_directory,
            "sam": {
                "sam": {
                    "checkpoint": "placeholder.pt", # Will be overridden
                    "model_config": "placeholder.yaml", # Will be overridden
                    "auto": True, # Matches original node logic implicitly used by SamModelMesh
                    "ground": False,
                    "engine_config": {
                        "points_per_side": args.sam_points_per_side,
                        "crop_n_layers": 0,
                        "pred_iou_thresh": args.sam_pred_iou_thresh,
                        "stability_score_thresh": args.sam_stability_score_thresh,
                        "stability_score_offset": args.sam_stability_score_offset,
                    }
                }
            },
            "sam_mesh": {
                "use_modes": ['sdf', 'norms'], # Keep default
                "min_area": args.samesh_min_area,
                "connections_bin_resolution": args.samesh_connections_bin_resolution,
                "connections_bin_threshold_percentage": args.samesh_connections_bin_thresh_perc,
                "smoothing_threshold_percentage_size": args.samesh_smoothing_thresh_perc_size,
                "smoothing_threshold_percentage_area": args.samesh_smoothing_thresh_perc_area,
                "smoothing_iterations": args.samesh_smoothing_iterations,
                "repartition_cost": 1, # Keep default
                "repartition_lambda": args.samesh_repartition_lambda,
                "repartition_iterations": args.samesh_repartition_iterations,
            },
            # Added default renderer config as it's required by SamModelMesh init
            "renderer": {
                "target_dim": [1024, 1024],
                "camera_generation_method": "icosahedron",
                "renderer_args": {"interpolate_norms": True},
                "sampling_args": {"radius": 2},
                "lighting_args": {}
            }
        })

        # --- Override SAM model paths in config ---
        print(f"Worker: Setting SAM model paths in config.")
        config.sam.sam.checkpoint = args.sam_checkpoint_path

        # Determine the correct config filename from the checkpoint path
        config_filename = os.path.basename(args.sam_model_config_path)
        config.sam.sam.model_config = config_filename
        print(f"Worker: Setting model_config in samesh config to: {config_filename}")

        # Handle target_labels
        target_labels_arg = None if args.target_labels < 0 else args.target_labels

        print(f"Worker: Config prepared:\n{OmegaConf.to_yaml(config)}") # Debug config

        # --- Call the samesh segmentation function ---
        print(f"Worker: Calling samesh.models.sam_mesh.segment_mesh...")
        # The function saves the mesh and returns the trimesh object. We don't need the object here.
        # We just need the side effect of the files being created.
        _ = segment_mesh_samesh_func(
            filename=mesh_file_path,
            config=config,
            visualize=args.visualize,
            extension=args.output_extension,
            target_labels=target_labels_arg,
            texture=args.keep_texture
        )
        print(f"Worker: samesh function completed.")

        # --- Determine output paths (same logic as in the node) ---
        output_mesh_filename = f"{filename_stem}_{args.output_filename_prefix}.{args.output_extension}"
        output_json_filename = f"{filename_stem}_{args.output_filename_prefix}_face2label.json"
        final_output_mesh_path = os.path.join(args.output_directory, output_mesh_filename)
        final_output_json_path = os.path.join(args.output_directory, output_json_filename)

        default_saved_mesh_path = os.path.join(args.output_directory, filename_stem, f"{filename_stem}_segmented.{args.output_extension}")
        default_saved_json_path = os.path.join(args.output_directory, filename_stem, f"{filename_stem}_face2label.json")
        output_subdir = os.path.join(args.output_directory, filename_stem)

        # --- Rename/move output files ---
        rename_success = True
        if os.path.exists(default_saved_mesh_path):
            try:
                # --- Add check and removal of destination file ---
                if os.path.exists(final_output_mesh_path):
                    print(f"Worker: Destination mesh {final_output_mesh_path} exists. Removing before rename.")
                    os.remove(final_output_mesh_path)
                # -------------------------------------------------
                os.rename(default_saved_mesh_path, final_output_mesh_path)
                print(f"Worker: Renamed mesh to {final_output_mesh_path}")
            except Exception as e:
                 print(f"Worker Error: Failed to rename mesh {default_saved_mesh_path} to {final_output_mesh_path}: {e}", file=sys.stderr)
                 final_output_mesh_path = f"Error: Failed to rename mesh - {e}"
                 rename_success = False
        else:
             print(f"Worker Warning: Expected output mesh not found at {default_saved_mesh_path}", file=sys.stderr)
             final_output_mesh_path = f"Error: Output mesh not found at {default_saved_mesh_path}"
             rename_success = False

        if os.path.exists(default_saved_json_path):
            try:
                 # --- Add check and removal of destination file ---
                 if os.path.exists(final_output_json_path):
                     print(f"Worker: Destination json {final_output_json_path} exists. Removing before rename.")
                     os.remove(final_output_json_path)
                 # -------------------------------------------------
                 os.rename(default_saved_json_path, final_output_json_path)
                 print(f"Worker: Renamed json to {final_output_json_path}")
            except Exception as e:
                 print(f"Worker Error: Failed to rename JSON {default_saved_json_path} to {final_output_json_path}: {e}", file=sys.stderr)
                 final_output_json_path = f"Error: Failed to rename JSON - {e}"
                 rename_success = False
        else:
             print(f"Worker Warning: Expected output json not found at {default_saved_json_path}", file=sys.stderr)
             final_output_json_path = f"Error: Output JSON not found at {default_saved_json_path}"
             rename_success = False

        # Clean up empty subdir
        try:
            if os.path.exists(output_subdir) and not os.listdir(output_subdir):
                os.rmdir(output_subdir)
        except OSError as e:
            print(f"Worker Warning: Could not remove empty samesh output directory {output_subdir}: {e}", file=sys.stderr)


        if not rename_success:
             print("Worker Error: One or more output files could not be found or renamed.", file=sys.stderr)
             # Outputting paths even on error might help debugging
             # sys.exit(1) # Decide if failure to rename should be a fatal error

        # --- Output paths as JSON to stdout ---
        result = {
            "output_mesh_path": final_output_mesh_path,
            "face2label_path": final_output_json_path
        }
        # Print status message *before* the JSON
        print("Worker: Segmentation script finished successfully.") 
        # Print JSON result as the very last line
        print(json.dumps(result))

    except ImportError:
         # Already handled above, but catch again just in case
         print("Worker Error: Exiting due to import failure.", file=sys.stderr)
         sys.exit(1)
    except FileNotFoundError as e:
         print(f"Worker Error: File not found - {e}", file=sys.stderr)
         traceback.print_exc(file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"Worker Error: An unexpected error occurred during segmentation.", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 