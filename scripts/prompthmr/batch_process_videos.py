#!/usr/bin/env python3
"""
Batch process videos to extract SMPL poses in camera & world space.

Usage:
    # Single video
    python scripts/batch_process_videos.py --input data/examples/dance_1.mp4 --output results

    # Directory of videos
    python scripts/batch_process_videos.py --input /path/to/videos/ --output results

    # Static camera mode (faster)
    python scripts/batch_process_videos.py --input data/examples/ --output results --static_camera
"""

import os
import sys

# Must set paths before other imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# PromptHMR root — set the PROMPTHMR_ROOT environment variable to your checkout
# ============================================================================
PROMPTHMR_ROOT = os.environ.get("PROMPTHMR_ROOT", os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "PromptHMR"))
sys.path.insert(0, PROMPTHMR_ROOT)
sys.path.insert(0, os.path.join(PROMPTHMR_ROOT, "scripts"))
sys.path.insert(0, _SCRIPT_DIR)  # for descriptor_utils living alongside this script
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================================
# Set cache directories (like demo_phmr.py)
# Override by setting the PROMPTHMR_CACHE_DIR environment variable.
# ============================================================================
CACHE_DIR = os.environ.get("PROMPTHMR_CACHE_DIR", os.path.expanduser("~/.cache/prompthmr"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TMPDIR"] = f"{CACHE_DIR}/tmp"
os.environ["TEMP"] = f"{CACHE_DIR}/tmp"
os.environ["TMP"] = f"{CACHE_DIR}/tmp"
os.environ["XDG_CACHE_HOME"] = f"{CACHE_DIR}/xdg_cache"
os.environ["HF_HOME"] = f"{CACHE_DIR}/huggingface"
os.environ["TORCH_HOME"] = f"{CACHE_DIR}/torch"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/huggingface/transformers"
os.environ["YOLO_CONFIG_DIR"] = f"{CACHE_DIR}/ultralytics"
for d in ["tmp", "xdg_cache", "huggingface", "torch", "ultralytics"]:
    os.makedirs(f"{CACHE_DIR}/{d}", exist_ok=True)

# ============================================================================
# Imports (after cache setup)
# ============================================================================
import numpy as np
import argparse
import joblib
from glob import glob

from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from pipeline import Pipeline
from pipeline.utils import load_video_frames


# ============================================================================
# Pose Export Functions
# ============================================================================

def save_poses_npy(results, output_dir):
    """
    Save per-person poses in camera and world frames as NPY files.
    
    NPY structure (dict):
        - frames: (N,) frame indices where person appears
        - pose: (N, 165) axis-angle pose parameters
        - rotmat: (N, 55, 3, 3) rotation matrices (camera only)
        - shape: (N, 10) betas
        - trans: (N, 3) translation
    """
    cam_dir = os.path.join(output_dir, 'poses', 'camera')
    world_dir = os.path.join(output_dir, 'poses', 'world')
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(world_dir, exist_ok=True)
    
    for pid, person in results['people'].items():
        frames = person['frames']
        
        # Camera frame
        smplx_cam = person['smplx_cam']
        cam_data = {
            'frames': np.array(frames),
            'pose': np.array(smplx_cam['pose']),          # (N, 165) axis-angle
            'rotmat': np.array(smplx_cam['rotmat']),      # (N, 55, 3, 3)
            'shape': np.array(smplx_cam['shape']),        # (N, 10)
            'trans': np.array(smplx_cam['trans']),        # (N, 3)
        }
        np.save(os.path.join(cam_dir, f'person_{pid}.npy'), cam_data)
        
        # World frame
        smplx_world = person['smplx_world']
        world_data = {
            'frames': np.array(frames),
            'pose': np.array(smplx_world['pose']),        # (N, 165) axis-angle
            'shape': np.array(smplx_world['shape']),      # (N, 10)
            'trans': np.array(smplx_world['trans']),      # (N, 3)
        }
        np.save(os.path.join(world_dir, f'person_{pid}.npy'), world_data)
    
    # Save camera parameters (per-frame extrinsics)
    cam_params = {
        # Camera frame (SLAM output)
        'pred_cam_R': np.array(results['camera']['pred_cam_R']),      # (T, 3, 3)
        'pred_cam_T': np.array(results['camera']['pred_cam_T']),      # (T, 3)
        'img_focal': results['camera']['img_focal'],
        'img_center': np.array(results['camera']['img_center']),
        # World frame (gravity-aligned)
        'Rwc': np.array(results['camera_world']['Rwc']),              # (T, 3, 3)
        'Twc': np.array(results['camera_world']['Twc']),              # (T, 3)
        'Rcw': np.array(results['camera_world']['Rcw']),              # (T, 3, 3)
        'Tcw': np.array(results['camera_world']['Tcw']),              # (T, 3)
    }
    np.save(os.path.join(output_dir, 'poses', 'camera_params.npy'), cam_params)
    
    print(f"  -> Saved poses to {output_dir}/poses/")
    return cam_dir, world_dir


# ============================================================================
# Main Processing
# ============================================================================

def process_video(video_path, output_base_dir, static_camera=False, force_reprocess=False,
                  add_descriptor=False, cache_dir=None):
    """
    Process a single video and save poses.
    
    Args:
        video_path: path to input video
        output_base_dir: base output directory
        static_camera: assume static camera (no SLAM)
        force_reprocess: reprocess even if output exists
        add_descriptor: if True, run VLM to add clothing descriptor per person (cached under cache_dir)
        cache_dir: directory for VLM cache (required if add_descriptor=True)
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    
    # Check if already processed
    poses_exist = os.path.exists(os.path.join(output_dir, 'poses', 'camera_params.npy'))
    
    if poses_exist and not force_reprocess:
        print(f"Skipping {video_name} (already processed, use --force to reprocess)")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Run pipeline
    pipeline = Pipeline(static_cam=static_camera)
    results = pipeline(video_path, output_dir, save_only_essential=False)
    
    # Optional: add visual descriptor (clothing) per person via VLM (cached under cache_dir)
    if add_descriptor and cache_dir:
        print("Adding person descriptors (VLM)...")
        try:
            images, _, _ = load_video_frames(
                video_path,
                output_folder=output_dir,
                max_height=896,
                max_fps=60,
            )
            from descriptor_utils import add_person_descriptors, save_descriptors
            results = add_person_descriptors(
                results, images, cache_dir=cache_dir, output_dir=output_dir
            )
            save_descriptors(results, output_dir)
            joblib.dump(results, os.path.join(output_dir, "results.pkl"))
        except Exception as e:
            print(f"  Warning: descriptor step failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save poses as NPY
    print("Saving poses...")
    save_poses_npy(results, output_dir)
    
    print(f"Done: {video_name}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Batch process videos for SMPL pose extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single video
  python scripts/batch_process_videos.py --input data/examples/dance_1.mp4 --output results

  # Process directory of videos
  python scripts/batch_process_videos.py --input /path/to/videos/ --output results

  # Static camera (faster, no SLAM)
  python scripts/batch_process_videos.py --input data/examples/ --output results --static_camera

Output Structure:
  output_dir/
  ├── video_name/
  │   ├── poses/
  │   │   ├── camera/
  │   │   │   └── person_<id>.npy    # pose, rotmat, shape, trans in camera space
  │   │   ├── world/
  │   │   │   └── person_<id>.npy    # pose, shape, trans in world space
  │   │   ├── descriptors.json       # (if --descriptor) person_id -> clothing description
  │   │   └── camera_params.npy     # per-frame camera extrinsics
  │   └── results.pkl                # full pipeline results (includes descriptor per person if --descriptor)
        """
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or directory containing videos')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    
    # Processing options
    parser.add_argument('--static_camera', action='store_true',
                        help='Assume static camera (skips SLAM, faster)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocess even if output exists')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID (default: 0)')
    
    # File options
    parser.add_argument('--extensions', type=str, default='mp4,avi,mov,mkv',
                        help='Video file extensions to process (default: mp4,avi,mov,mkv)')
    parser.add_argument('--descriptor', action='store_true',
                        help='Add a visual descriptor (clothing) per person via small VLM; '
                             'VLM weights are cached under CACHE_DIR')

    args = parser.parse_args()
    
    # Set GPU (override the default set at top)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load SMPLX model (for potential future use, but pipeline handles it internally)
    print("Loading SMPLX model...")
    _ = SMPLX_Layer(SMPLX_PATH).cuda()
    
    # Gather videos
    if os.path.isfile(args.input):
        video_paths = [args.input]
    else:
        extensions = args.extensions.split(',')
        video_paths = []
        for ext in extensions:
            video_paths.extend(glob(os.path.join(args.input, f'*.{ext}')))
            video_paths.extend(glob(os.path.join(args.input, f'*.{ext.upper()}')))
        video_paths = sorted(set(video_paths))
    
    if len(video_paths) == 0:
        print(f"No videos found in {args.input}")
        return
    
    print(f"Found {len(video_paths)} video(s) to process")
    for vp in video_paths:
        print(f"  - {os.path.basename(vp)}")
    
    # Process each video
    successful = 0
    failed = []
    
    for video_path in video_paths:
        try:
            result = process_video(
                video_path,
                args.output,
                static_camera=args.static_camera,
                force_reprocess=args.force,
                add_descriptor=args.descriptor,
                cache_dir=CACHE_DIR,
            )
            if result is not None:
                successful += 1
        except Exception as e:
            print(f"ERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((video_path, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(video_paths)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for vp, err in failed:
            print(f"  - {os.path.basename(vp)}: {err}")


if __name__ == '__main__':
    main()
