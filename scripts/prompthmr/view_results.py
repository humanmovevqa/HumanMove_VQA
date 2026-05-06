#!/usr/bin/env python3
"""
View saved results in Viser 3D viewer.

Usage:
    python scripts/view_results.py --results_dir results/video_name --video /path/to/video.mp4

    # With subsampling for faster loading
    python scripts/view_results.py --results_dir results/video_name --video /path/to/video.mp4 --subsample 4

    # Save poses in Motion-X 322 format (Y-up coords); each person_<id>_motionx.npz has 'motionx' + 'descriptor'
    python scripts/view_results.py --results_dir results/video_name --video /path/to/video.mp4 --save_motionx

    # Save poses with display transform (trans: -X, Z, Y - matches render_video); same .npz layout
    python scripts/view_results.py --results_dir results/video_name --video /path/to/video.mp4 --save_motionx_display

    # Render skeleton video with checkerboard floor
    python scripts/view_results.py --results_dir results/video_name --video /path/to/video.mp4 --render_video
"""

import json
import os
import sys

# ============================================================================
# PromptHMR root — set the PROMPTHMR_ROOT environment variable to your checkout
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTHMR_ROOT = os.environ.get("PROMPTHMR_ROOT", os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "PromptHMR"))
sys.path.insert(0, PROMPTHMR_ROOT)
sys.path.insert(0, os.path.join(PROMPTHMR_ROOT, "scripts"))
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
import torch
import numpy as np
import time
import joblib
import tyro
from typing import Optional

from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline


# ============================================================================
# Video Rendering (Y-up with Viser-style floor)
# ============================================================================
def render_skeleton_video(
    world4d,
    smplx_model,
    output_path: str,
    fps: int = 30,
    static_camera: bool = True,
):
    """
    Render skeleton video with Y-up coordinate system and Viser-style floor.
    
    Input coordinate system (Y-up):
    - X: right
    - Y: up (vertical)
    - Z: forward/depth
    
    Matplotlib 3D uses Z as vertical, so we swap Y↔Z for plotting:
    - plot_x = input_x
    - plot_y = input_z (depth)
    - plot_z = input_y (height, rendered as vertical)
    
    Args:
        world4d: dict of frame_idx -> {'pose', 'shape', 'trans', 'track_id'}
        smplx_model: SMPLX model for computing joints/vertices
        output_path: Path to save video (.mp4)
        fps: Frames per second
        static_camera: If True, use fixed camera bounds for full trajectory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    import imageio
    import io
    
    # SMPLX skeleton connections with side labels
    SKELETON_BODY = [
        # Spine (center)
        (0, 3, 'center'), (3, 6, 'center'), (6, 9, 'center'), (9, 12, 'center'), (12, 15, 'center'),
        # Left leg
        (0, 1, 'left'), (1, 4, 'left'), (4, 7, 'left'), (7, 10, 'left'),
        # Right leg
        (0, 2, 'right'), (2, 5, 'right'), (5, 8, 'right'), (8, 11, 'right'),
        # Left arm
        (9, 13, 'left'), (13, 16, 'left'), (16, 18, 'left'), (18, 20, 'left'),
        # Right arm
        (9, 14, 'right'), (14, 17, 'right'), (17, 19, 'right'), (19, 21, 'right'),
    ]
    
    SIDE_COLORS = {
        'left': '#2E86AB',   # Blue
        'right': '#E94F37',  # Red
        'center': '#333333', # Black/dark gray
    }
    
    print(f"Rendering Y-up skeleton video to {output_path}...")
    
    # Collect all joint positions and vertices per frame
    all_joints = []
    all_verts = []
    frame_indices = sorted(world4d.keys())
    
    for frame_idx in frame_indices:
        frame_data = world4d[frame_idx]
        if len(frame_data['track_id']) == 0:
            all_joints.append(None)
            all_verts.append(None)
            continue
        
        # Get joints and vertices from SMPLX
        rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        output = smplx_model(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=frame_data['shape'].cuda(),
            transl=frame_data['trans'].cuda()
        )
        joints = output.joints[:, :22].cpu().numpy()  # (num_people, 22, 3)
        verts = output.vertices.cpu().numpy()          # (num_people, V, 3)
        all_joints.append(joints)
        all_verts.append(verts)
    
    # Compute floor from vertices (like Viser)
    all_vert_points = []
    for verts in all_verts:
        if verts is not None:
            all_vert_points.append(verts.reshape(-1, 3))
    all_vert_points = np.concatenate(all_vert_points, axis=0)
    
    # Input coords: X, Y (up), Z
    # Floor at Y=0 (data is already floor-fitted)
    floor_y = 0.0
    
    # Compute extents in input coordinates
    x_min, x_max = all_vert_points[:, 0].min(), all_vert_points[:, 0].max()
    y_min, y_max = all_vert_points[:, 1].min(), all_vert_points[:, 1].max()  # height
    z_min, z_max = all_vert_points[:, 2].min(), all_vert_points[:, 2].max()  # depth
    
    # Scale and center (like Viser)
    sx = x_max - x_min
    sz = z_max - z_min
    floor_scale = max(sx, sz) * 1.5
    
    cx = (x_max + x_min) / 2
    cz = (z_max + z_min) / 2
    
    # Compute view bounds (in PLOT coordinates after Y↔Z swap)
    # plot_x = input_x, plot_y = input_z, plot_z = input_y
    padding = 0.3
    x_range = sx * (1 + padding)
    z_range = sz * (1 + padding)
    y_range = (y_max - y_min) * (1 + padding)
    max_range = max(x_range, z_range, y_range) / 2
    
    # Fixed view limits for static camera (in plot coordinates)
    # Note: X is negated, so limits are flipped
    if static_camera:
        view_margin = max_range * 0.15
        # plot_x = -input_x (negated, so flip the limits)
        fixed_xlim = (-cx - max_range - view_margin, -cx + max_range + view_margin)
        # plot_y = input_z (depth)
        fixed_ylim = (cz - max_range - view_margin, cz + max_range + view_margin)
        # plot_z = input_y (height) - floor at 0
        fixed_zlim = (floor_y - 0.1, y_max + view_margin)
    
    frames = []
    num_frames = len(frame_indices)
    
    for frame_num in range(num_frames):
        fig = plt.figure(figsize=(6, 6), facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='white')
        
        joints_list = all_joints[frame_num]
        
        # Draw floor plane (X-Z plane at Y=0 in input coords)
        # In plot coords: X-Y plane at Z=0, with X negated
        checker_size = floor_scale / 8
        for i in range(8):
            for j in range(8):
                # Input coords: x0, floor_y=0, z0
                x0 = cx - floor_scale/2 + i * checker_size
                z0 = cz - floor_scale/2 + j * checker_size
                color = '#CCCCCC' if (i + j) % 2 == 0 else '#AAAAAA'
                # Plot coords: [plot_x, plot_y, plot_z] = [-input_x, input_z, input_y]
                floor_verts = [
                    [-x0, z0, floor_y],
                    [-(x0 + checker_size), z0, floor_y],
                    [-(x0 + checker_size), z0 + checker_size, floor_y],
                    [-x0, z0 + checker_size, floor_y]
                ]
                floor_tile = Poly3DCollection([floor_verts], alpha=0.6, facecolor=color, edgecolor='#999999', linewidth=0.3)
                ax.add_collection3d(floor_tile)
        
        if joints_list is not None:
            num_people = joints_list.shape[0]
            for person_idx in range(num_people):
                joints = joints_list[person_idx]  # (22, 3) in input coords [x, y, z]
                
                # Swap Y and Z for matplotlib (Z is vertical in matplotlib)
                # Also negate X to match MotionScript left-right convention
                # plot_x = -input_x, plot_y = input_z, plot_z = input_y
                joints_plot = np.zeros_like(joints)
                joints_plot[:, 0] = -joints[:, 0]  # plot_x = -input_x (flip left-right)
                joints_plot[:, 1] = joints[:, 2]   # plot_y = input_z (depth)
                joints_plot[:, 2] = joints[:, 1]   # plot_z = input_y (height)
                
                # Draw bones with side-specific colors
                for i, j, side in SKELETON_BODY:
                    if i < len(joints_plot) and j < len(joints_plot):
                        color = SIDE_COLORS.get(side, '#333333')
                        ax.plot([joints_plot[i, 0], joints_plot[j, 0]],
                                [joints_plot[i, 1], joints_plot[j, 1]],
                                [joints_plot[i, 2], joints_plot[j, 2]],
                                c=color, linewidth=4, alpha=0.9, solid_capstyle='round')
                
                # Draw joints
                ax.scatter(joints_plot[:, 0], joints_plot[:, 1], joints_plot[:, 2], 
                           c='#555555', s=50, alpha=1.0, edgecolors='white', linewidths=1)
        
        # Set view limits (in plot coordinates, X is negated)
        if static_camera:
            ax.set_xlim3d(fixed_xlim)
            ax.set_ylim3d(fixed_ylim)
            ax.set_zlim3d(fixed_zlim)
            ax.autoscale(False)
        else:
            margin = max_range * 0.1
            ax.set_xlim(-cx - max_range - margin, -cx + max_range + margin)
            ax.set_ylim(cz - max_range - margin, cz + max_range + margin)
            ax.set_zlim(floor_y - 0.1, y_max + margin)
        
        # Clean look - remove axes
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        
        # Isometric view angle
        # elev=20 looks down from above, azim=45 rotates for nice 3/4 view
        ax.view_init(elev=20, azim=45)
        ax.dist = 7.0 if static_camera else 5.5
        
        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                    pad_inches=0, facecolor='white', edgecolor='none')
        buf.seek(0)
        img_frame = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        frames.append(img_frame)
        
        if (frame_num + 1) % 50 == 0:
            print(f"  Rendered {frame_num + 1}/{num_frames} frames...")
    
    # Compute consistent crop bounds across ALL frames
    global_crop = None
    for img_frame in frames:
        img_array = np.array(img_frame)
        non_white = np.where(np.any(img_array < 250, axis=2))
        if len(non_white[0]) > 0:
            y_min_f, y_max_f = non_white[0].min(), non_white[0].max()
            x_min_f, x_max_f = non_white[1].min(), non_white[1].max()
            if global_crop is None:
                global_crop = [x_min_f, y_min_f, x_max_f, y_max_f]
            else:
                global_crop[0] = min(global_crop[0], x_min_f)
                global_crop[1] = min(global_crop[1], y_min_f)
                global_crop[2] = max(global_crop[2], x_max_f)
                global_crop[3] = max(global_crop[3], y_max_f)
    
    # Apply consistent crop to all frames
    if global_crop:
        pad = 15
        x_min_crop = max(0, global_crop[0] - pad)
        y_min_crop = max(0, global_crop[1] - pad)
        x_max_crop = global_crop[2] + pad
        y_max_crop = global_crop[3] + pad
        frames = [f.crop((x_min_crop, y_min_crop, x_max_crop, y_max_crop)) for f in frames]
    
    # Save as MP4
    if frames:
        mp4_path = output_path.rsplit('.', 1)[0] + '.mp4'
        frame_arrays = [np.array(f) for f in frames]
        imageio.mimsave(mp4_path, frame_arrays, fps=fps)
        print(f"Saved: {mp4_path}")
        print(f"  Frames: {num_frames}")
        print(f"  FPS: {fps}")


# ============================================================================
# Motion-X Conversion
# ============================================================================
def convert_world4d_to_motionx(world4d, results):
    """
    Convert world4d poses to Motion-X 322 format.
    
    The world4d data already has proper transforms applied:
    - Axis flip (Y-up)
    - Floor fitting (Y=0)
    
    Motion-X 322 format:
        [0:3]     root_orient (1×3)
        [3:66]    pose_body (21×3)
        [66:156]  pose_hand (30×3)
        [156:159] pose_jaw (1×3)
        [159:209] face_expr (50)      <- zeros
        [209:309] face_shape (100)    <- zeros
        [309:312] trans (3)
        [312:322] betas (10)
    
    Args:
        world4d: dict of frame_idx -> {'pose', 'shape', 'trans', 'track_id'}
        results: original results dict (for getting full frame info per person)
        
    Returns:
        dict: person_id -> (N, 322) Motion-X array
    """
    # Collect poses per track_id across all frames
    person_data = {}
    
    for frame_idx in sorted(world4d.keys()):
        frame_data = world4d[frame_idx]
        if len(frame_data['track_id']) == 0:
            continue
            
        track_ids = frame_data['track_id'].numpy()
        poses = frame_data['pose'].numpy()      # (num_people, 55, 3)
        shapes = frame_data['shape'].numpy()    # (num_people, 10)
        trans = frame_data['trans'].numpy()     # (num_people, 3)
        
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid not in person_data:
                person_data[tid] = {'frames': [], 'pose': [], 'shape': [], 'trans': []}
            
            person_data[tid]['frames'].append(frame_idx)
            person_data[tid]['pose'].append(poses[i].reshape(-1))  # (165,)
            person_data[tid]['shape'].append(shapes[i])            # (10,)
            person_data[tid]['trans'].append(trans[i])             # (3,)
    
    # Convert to Motion-X 322 format per person
    motionx_data = {}
    
    for tid, data in person_data.items():
        N = len(data['frames'])
        pose = np.array(data['pose'])       # (N, 165)
        shape = np.array(data['shape'])     # (N, 10)
        trans = np.array(data['trans'])     # (N, 3)
        frames = np.array(data['frames'])   # (N,)
        
        # Initialize Motion-X array
        motionx = np.zeros((N, 322), dtype=np.float32)
        
        # Map PromptHMR (165) to Motion-X (322)
        # PromptHMR: [global_orient(3), body_pose(63), jaw(3), leye(3), reye(3), lhand(45), rhand(45)]
        # Motion-X:  [root_orient(3), pose_body(63), pose_hand(90), pose_jaw(3), face_expr(50), face_shape(100), trans(3), betas(10)]
        
        motionx[:, 0:3] = pose[:, 0:3]        # root_orient <- global_orient
        motionx[:, 3:66] = pose[:, 3:66]      # pose_body <- body_pose
        motionx[:, 66:156] = pose[:, 75:165]  # pose_hand <- left_hand + right_hand
        motionx[:, 156:159] = pose[:, 66:69]  # pose_jaw <- jaw_pose
        # [159:209] face_expr stays zeros
        # [209:309] face_shape stays zeros
        motionx[:, 309:312] = trans           # trans
        motionx[:, 312:322] = shape           # betas
        
        motionx_data[tid] = {
            'motionx': motionx,
            'frames': frames,
        }
    
    return motionx_data


def convert_world4d_to_motionx_display(world4d, results):
    """
    Convert world4d poses to Motion-X 322 format with display transform.
    
    Applies the same coordinate transform as render_video:
    - trans_x = -input_x (flip left-right)
    - trans_y = input_z (depth becomes Y)  
    - trans_z = input_y (height becomes Z)
    
    Note: Only translation is transformed. Pose parameters remain in original
    Y-up coordinates (would need rotation transform for full conversion).
    
    Motion-X 322 format:
        [0:3]     root_orient (1×3)
        [3:66]    pose_body (21×3)
        [66:156]  pose_hand (30×3)
        [156:159] pose_jaw (1×3)
        [159:209] face_expr (50)      <- zeros
        [209:309] face_shape (100)    <- zeros
        [309:312] trans (3)           <- TRANSFORMED
        [312:322] betas (10)
    
    Args:
        world4d: dict of frame_idx -> {'pose', 'shape', 'trans', 'track_id'}
        results: original results dict
        
    Returns:
        dict: person_id -> (N, 322) Motion-X array with transformed translation
    """
    # Collect poses per track_id across all frames
    person_data = {}
    
    for frame_idx in sorted(world4d.keys()):
        frame_data = world4d[frame_idx]
        if len(frame_data['track_id']) == 0:
            continue
            
        track_ids = frame_data['track_id'].numpy()
        poses = frame_data['pose'].numpy()      # (num_people, 55, 3)
        shapes = frame_data['shape'].numpy()    # (num_people, 10)
        trans = frame_data['trans'].numpy()     # (num_people, 3)
        
        for i, tid in enumerate(track_ids):
            tid = int(tid)
            if tid not in person_data:
                person_data[tid] = {'frames': [], 'pose': [], 'shape': [], 'trans': []}
            
            person_data[tid]['frames'].append(frame_idx)
            person_data[tid]['pose'].append(poses[i].reshape(-1))  # (165,)
            person_data[tid]['shape'].append(shapes[i])            # (10,)
            person_data[tid]['trans'].append(trans[i])             # (3,)
    
    # Convert to Motion-X 322 format per person
    motionx_data = {}
    
    for tid, data in person_data.items():
        N = len(data['frames'])
        pose = np.array(data['pose'])       # (N, 165)
        shape = np.array(data['shape'])     # (N, 10)
        trans = np.array(data['trans'])     # (N, 3)
        frames = np.array(data['frames'])   # (N,)
        
        # Apply display transform to translation
        # Same as render_video: x' = -x, y' = z, z' = y
        trans_display = np.zeros_like(trans)
        trans_display[:, 0] = -trans[:, 0]  # X negated
        trans_display[:, 1] = trans[:, 2]   # Z -> Y
        trans_display[:, 2] = trans[:, 1]   # Y -> Z
        
        # Initialize Motion-X array
        motionx = np.zeros((N, 322), dtype=np.float32)
        
        # Map PromptHMR (165) to Motion-X (322)
        motionx[:, 0:3] = pose[:, 0:3]        # root_orient <- global_orient
        motionx[:, 3:66] = pose[:, 3:66]      # pose_body <- body_pose
        motionx[:, 66:156] = pose[:, 75:165]  # pose_hand <- left_hand + right_hand
        motionx[:, 156:159] = pose[:, 66:69]  # pose_jaw <- jaw_pose
        # [159:209] face_expr stays zeros
        # [209:309] face_shape stays zeros
        motionx[:, 309:312] = trans_display   # trans (TRANSFORMED)
        motionx[:, 312:322] = shape           # betas
        
        motionx_data[tid] = {
            'motionx': motionx,
            'frames': frames,
        }
    
    return motionx_data


def get_descriptors_for_persons(results, results_dir, person_ids):
    """
    Get clothing/appearance descriptor for each person id (0-based track id as in motionx output).

    Tries results['people'][pid]['descriptor'] (pid = tid or tid+1), then
    results_dir/poses/descriptors.json as fallback.

    Returns:
        dict: str(tid) -> descriptor text (e.g. "0" -> "a person wearing ...")
    """
    out = {}
    people = results.get("people") or {}
    # Load fallback from poses/descriptors.json if present
    fallback = {}
    poses_descriptors = os.path.join(results_dir, "poses", "descriptors.json")
    if os.path.isfile(poses_descriptors):
        try:
            with open(poses_descriptors, encoding="utf-8") as f:
                fallback = json.load(f)
        except Exception:
            pass
    for tid in person_ids:
        text = "unknown"
        for pid in (tid, tid + 1):
            if pid in people:
                text = people[pid].get("descriptor", text)
                if text != "unknown":
                    break
        if text == "unknown" and fallback:
            text = fallback.get(str(tid), fallback.get(str(tid + 1), "unknown"))
        out[str(tid)] = text
    return out


def main(
    results_dir: str,
    video: str,
    subsample: int = 1,
    total: int = 1500,
    save_motionx: bool = False,
    save_motionx_display: bool = False,
    motionx_output: Optional[str] = None,
    render_video: bool = False,
    video_output: Optional[str] = None,
    video_fps: int = 30,
):
    """
    View saved results in Viser 3D viewer.
    
    Args:
        results_dir: Path to results directory (e.g., results/video_name)
        video: Path to original video file
        subsample: Subsample frames for faster loading (default: 1 = all frames)
        total: Maximum number of frames to visualize (default: 1500)
        save_motionx: If True, save poses in Motion-X 322 format (Y-up coords) and exit
        save_motionx_display: If True, save poses with display transform (trans: -X, Z, Y) and exit
        motionx_output: Output directory for Motion-X files (default: results_dir/motionx/)
        render_video: If True, render skeleton video with checkerboard floor and exit
        video_output: Output path for rendered video (default: results_dir/skeleton.mp4)
        video_fps: FPS for rendered video (default: 30)
    """
    # Check results exist
    results_path = os.path.join(results_dir, 'results.pkl')
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return
    
    print(f"Loading results from {results_path}")
    results = joblib.load(results_path)
    
    # Load SMPLX model
    print("Loading SMPLX model...")
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    
    # Load images
    print("Loading video frames...")
    pipeline = Pipeline()
    pipeline.results = results
    images, _ = pipeline.load_frames(video, results_dir)
    pipeline.images = images
    
    # Create world4d for visualization
    print("Creating world4d data...")
    world4d = pipeline.create_world4d(step=subsample, total=total)
    world4d = {i: world4d[k] for i, k in enumerate(world4d)}
    
    # Save Motion-X format if requested (Y-up coords)
    if save_motionx:
        print("Converting to Motion-X 322 format (Y-up coords)...")
        motionx_data = convert_world4d_to_motionx(world4d, results)
        
        # Determine output directory
        output_dir = motionx_output if motionx_output else os.path.join(results_dir, 'motionx')
        os.makedirs(output_dir, exist_ok=True)
        
        # Descriptors (clothing/appearance) per person for inclusion in same file
        descriptors = get_descriptors_for_persons(results, results_dir, list(motionx_data.keys()))
        # Save each person's pose + descriptor in one .npz (keys: 'motionx', 'descriptor')
        for tid, data in motionx_data.items():
            output_path = os.path.join(output_dir, f'person_{tid}_motionx.npz')
            desc = descriptors.get(str(tid), "unknown")
            np.savez_compressed(
                output_path,
                motionx=data['motionx'],
                descriptor=np.array(desc),  # stored as 0-d array, load with .item()
            )
            print(f"  Saved person {tid}: {output_path}")
            print(f"    Shape: {data['motionx'].shape}")
            print(f"    Frames: {data['frames'][0]} - {data['frames'][-1]} ({len(data['frames'])} total)")
            print(f"    Descriptor: {desc[:80]}{'...' if len(desc) > 80 else ''}")
        
        print(f"\nMotion-X files saved to: {output_dir}")
        print("  Each .npz contains 'motionx' (N,322) and 'descriptor' (str). Load: np.load(path, allow_pickle=True)")
        print("\nTo generate caption, run:")
        print(f"  python caption_single_motionx.py -m {output_dir}/person_0_motionx.npz -o output_dir/")
    
    # Save Motion-X format with display transform if requested
    if save_motionx_display:
        print("Converting to Motion-X 322 format (display transform: -X, Z, Y)...")
        motionx_data = convert_world4d_to_motionx_display(world4d, results)
        
        # Determine output directory
        output_dir = motionx_output if motionx_output else os.path.join(results_dir, 'motionx_display')
        os.makedirs(output_dir, exist_ok=True)
        
        # Descriptors per person for inclusion in same file
        descriptors = get_descriptors_for_persons(results, results_dir, list(motionx_data.keys()))
        # Save each person's pose + descriptor in one .npz (keys: 'motionx', 'descriptor')
        for tid, data in motionx_data.items():
            output_path = os.path.join(output_dir, f'person_{tid}_motionx_display.npz')
            desc = descriptors.get(str(tid), "unknown")
            np.savez_compressed(
                output_path,
                motionx=data['motionx'],
                descriptor=np.array(desc),
            )
            print(f"  Saved person {tid}: {output_path}")
            print(f"    Shape: {data['motionx'].shape}")
            print(f"    Frames: {data['frames'][0]} - {data['frames'][-1]} ({len(data['frames'])} total)")
            print(f"    Trans transform: x'=-x, y'=z, z'=y (matches render_video)")
            print(f"    Descriptor: {desc[:80]}{'...' if len(desc) > 80 else ''}")
        
        print(f"\nMotion-X files (display coords) saved to: {output_dir}")
        print("  Each .npz contains 'motionx' (N,322) and 'descriptor' (str). Load: np.load(path, allow_pickle=True)")
    
    # Render skeleton video if requested
    if render_video:
        output_path = video_output if video_output else os.path.join(results_dir, 'skeleton.mp4')
        # Ensure the directory for the output mp4 exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        render_skeleton_video(
            world4d=world4d,
            smplx_model=smplx,
            output_path=output_path,
            fps=video_fps // subsample,
            static_camera=True,
        )
    
    if save_motionx or save_motionx_display or render_video:
        return
    
    # Subsample images to match
    images = images[:total][::subsample]
    
    # Compute vertices for visualization
    print("Computing mesh vertices...")
    all_verts = []
    for k in world4d:
        world3d = world4d[k]
        if len(world3d['track_id']) == 0:  # no people
            continue
        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        verts = smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=world3d['shape'].cuda(),
            transl=world3d['trans'].cuda()
        ).vertices.cpu().numpy()
        
        world3d['vertices'] = verts
        all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))
    
    if len(all_verts) == 0:
        print("Error: No people found in results")
        return
    
    all_verts = torch.cat(all_verts)
    [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)
    
    # Launch Viser
    print("Starting Viser server...")
    server, gui = viser_vis_world4d(
        images,
        world4d,
        smplx.faces,
        floor=[gv, gf],
        init_fps=30 / subsample
    )
    
    url = f'https://localhost:{server.get_port()}'
    print(f'\n{"="*60}')
    print(f'Viser URL: {url}')
    print(f'{"="*60}')
    print('For longer videos, it may take a few seconds for the webpage to load.')
    print('Press Ctrl+C to stop the server.\n')
    
    gui_playing, gui_timestep, gui_framerate, num_frames = gui
    while True:
        # Update the timestep if we're playing
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == '__main__':
    tyro.cli(main)
