#!/usr/bin/env python3
"""
Visualize SMPL meshes in camera space and save images/videos.

Usage:
    # Single frame
    python scripts/visualize_gloss.py --results_dir results/video_name --frame 0 --output output.png
    
    # Single frame with background
    python scripts/visualize_gloss.py --results_dir results/video_name --frame 0 --video video.mp4 --output output.png
    
    # Full video overlay (saves frames + video)
    python scripts/visualize_gloss.py --results_dir results/video_name --video video.mp4 --render_video --output_dir output_frames/
    
    # Side-by-side comparison (original | overlay)
    python scripts/visualize_gloss.py --results_dir results/video_name --video video.mp4 --render_video --side_by_side --output_dir output_frames/
"""

import os
import sys

# ============================================================================
# PromptHMR root — set the PROMPTHMR_ROOT environment variable to your checkout
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTHMR_ROOT = os.environ.get("PROMPTHMR_ROOT", os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "PromptHMR"))
sys.path.insert(0, PROMPTHMR_ROOT)
sys.path.insert(0, os.path.join(PROMPTHMR_ROOT, "scripts"))

import argparse
import numpy as np
import torch
import joblib
import cv2
from tqdm import tqdm

from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.vis.renderer import Renderer


def load_results(results_dir):
    """Load results.pkl from results directory."""
    results_path = os.path.join(results_dir, 'results.pkl')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    return joblib.load(results_path)


def get_smplx_params_at_frame(results, person_id, frame_idx):
    """
    Get SMPLX parameters for a person at a specific frame (camera space).
    
    Returns:
        dict with pose, shape, trans or None if person not in frame
    """
    person = results['people'].get(person_id)
    if person is None:
        return None
    
    frames = np.array(person['frames'])
    matches = np.where(frames == frame_idx)[0]
    if len(matches) == 0:
        return None
    
    # Find index in person's frame list
    idx = matches[0]
    
    smplx_cam = person['smplx_cam']
    return {
        'pose': torch.tensor(smplx_cam['pose'][idx]).unsqueeze(0),      # (1, 165)
        'rotmat': torch.tensor(smplx_cam['rotmat'][idx]).unsqueeze(0),  # (1, 55, 3, 3)
        'shape': torch.tensor(smplx_cam['shape'][idx]).unsqueeze(0),    # (1, 10)
        'trans': torch.tensor(smplx_cam['trans'][idx]).unsqueeze(0),    # (1, 3)
    }


def compute_vertices(smplx_model, params):
    """Compute SMPLX vertices from parameters."""
    with torch.no_grad():
        output = smplx_model(
            global_orient=params['rotmat'][:, :1].cuda(),
            body_pose=params['rotmat'][:, 1:22].cuda(),
            betas=params['shape'].cuda(),
            transl=params['trans'].cuda()
        )
    return output.vertices.cpu().numpy()[0]  # (10475, 3)


def render_and_save(vertices_list, faces, output_path, focal_length=1000, 
                    img_width=1920, img_height=1080, background=None):
    """
    Render meshes using PyTorch3D and save to file.
    
    Args:
        vertices_list: list of (V, 3) numpy arrays or torch tensors
        faces: (F, 3) numpy array or torch tensor
        output_path: path to save rendered image
        focal_length: camera focal length
        img_width, img_height: image dimensions
        background: optional background image (H, W, 3) RGB numpy array
    """
    # Create renderer
    renderer = Renderer(
        width=img_width,
        height=img_height,
        focal_length=focal_length,
        device='cuda',
        bin_size=0
    )
    
    # Convert vertices to torch tensor
    verts_tensors = []
    for verts in vertices_list:
        if isinstance(verts, np.ndarray):
            verts = torch.from_numpy(verts).float()
        verts_tensors.append(verts.cuda())
    
    # Stack all vertices: (num_people, V, 3)
    verts_batch = torch.stack(verts_tensors, dim=0)
    
    # Create background if not provided
    if background is None:
        background = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # white background
    
    # Render
    img_out = renderer.render_meshes(
        verts_list=verts_batch,
        faces=faces,
        background=background,
        default_color=False  # different colors per person
    )
    
    # Save (convert RGB to BGR for cv2)
    cv2.imwrite(output_path, img_out[:, :, ::-1])
    print(f"Saved render to: {output_path}")


def load_video_frame(video_path, frame_idx):
    """Load a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame[:, :, ::-1]  # BGR to RGB


def get_video_info(video_path):
    """Get video info: fps, frame count, width, height."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_count, width, height


def render_video(results, smplx_model, video_path, output_dir, person_ids=None, 
                 save_frames=True, side_by_side=False):
    """
    Render SMPL overlay for entire video.
    
    Args:
        results: loaded results.pkl
        smplx_model: SMPLX model
        video_path: input video path
        output_dir: output directory for frames and video
        person_ids: list of person IDs to render (None = all)
        save_frames: whether to save individual frames
        side_by_side: if True, output is [original | overlay] side by side
    
    Returns:
        output_video_path: path to rendered video
    """
    # Get camera parameters
    focal = results['camera']['img_focal']
    img_center = results['camera']['img_center']
    img_width = int(img_center[0] * 2)
    img_height = int(img_center[1] * 2)
    
    # Get video info
    fps, frame_count, vid_width, vid_height = get_video_info(video_path)
    print(f"Video: {frame_count} frames at {fps:.1f} fps, {vid_width}x{vid_height}")
    
    # Create output directories: output_dir/video_name/
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_basename)
    os.makedirs(video_output_dir, exist_ok=True)
    frames_dir = os.path.join(video_output_dir, 'frames')
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    
    # Get all person IDs if not specified
    if person_ids is None:
        person_ids = list(results['people'].keys())
    
    # Create renderer (reuse for all frames)
    renderer = Renderer(
        width=img_width,
        height=img_height,
        focal_length=focal,
        device='cuda',
        bin_size=0
    )
    faces = smplx_model.faces
    
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    
    # Create video writer
    if side_by_side:
        video_name = 'sidebyside.mp4'
    else:
        video_name = 'overlay.mp4'
    output_video_path = os.path.join(video_output_dir, video_name)
    
    # H.264 requires even dimensions
    frame_width = img_width if img_width % 2 == 0 else img_width + 1
    frame_height = img_height if img_height % 2 == 0 else img_height + 1
    
    # Double width for side-by-side
    out_width = frame_width * 2 if side_by_side else frame_width
    out_height = frame_height
    
    # Use imageio for better codec support
    import imageio
    writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264', 
                                 pixelformat='yuv420p', macro_block_size=1)
    print(f"Output video resolution: {out_width}x{out_height}" + (" (side-by-side)" if side_by_side else ""))
    
    # Process each frame
    for frame_idx in tqdm(range(frame_count), desc="Rendering frames"):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Background
        background = frame_bgr[:, :, ::-1]  # BGR to RGB
        
        # Get vertices for all people in this frame
        vertices_list = []
        for pid in person_ids:
            params = get_smplx_params_at_frame(results, pid, frame_idx)
            if params is None:
                continue
            verts = compute_vertices(smplx_model, params)
            vertices_list.append(verts)
        
        # Render frame
        if len(vertices_list) > 0:
            # Convert vertices to torch
            verts_tensors = [torch.from_numpy(v).float().cuda() for v in vertices_list]
            verts_batch = torch.stack(verts_tensors, dim=0)
            
            # Render overlay
            img_out = renderer.render_meshes(
                verts_list=verts_batch,
                faces=faces,
                background=background,
                default_color=False
            )
        else:
            # No people in frame, just use background
            img_out = background
        
        # Convert to BGR for saving
        img_overlay_bgr = img_out[:, :, ::-1] if isinstance(img_out, np.ndarray) else img_out
        img_overlay_bgr = img_overlay_bgr.astype(np.uint8)
        
        # Resize overlay if needed (H.264 requires even dimensions)
        if img_overlay_bgr.shape[1] != frame_width or img_overlay_bgr.shape[0] != frame_height:
            img_overlay_bgr = cv2.resize(img_overlay_bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        
        # Create final frame (side-by-side or overlay only)
        if side_by_side:
            # Resize original frame to match
            original_bgr = frame_bgr
            if original_bgr.shape[1] != frame_width or original_bgr.shape[0] != frame_height:
                original_bgr = cv2.resize(original_bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            # Concatenate: [original | overlay]
            final_bgr = np.concatenate([original_bgr, img_overlay_bgr], axis=1)
        else:
            final_bgr = img_overlay_bgr
        
        # Save full-res frame
        if save_frames:
            frame_path = os.path.join(frames_dir, f'{frame_idx:06d}.png')
            cv2.imwrite(frame_path, final_bgr)
        
        # Write to video (imageio expects RGB)
        img_rgb = final_bgr[:, :, ::-1]
        writer.append_data(img_rgb)
    
    cap.release()
    writer.close()
    
    print(f"Saved video to: {output_video_path}")
    if save_frames:
        print(f"Saved {frame_count} frames to: {frames_dir}")
    
    return output_video_path


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL meshes in camera space')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to results directory containing results.pkl')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to visualize (default: 0)')
    parser.add_argument('--person_id', type=int, default=None,
                        help='Specific person ID to visualize (default: all)')
    parser.add_argument('--output', type=str, default='render.png',
                        help='Output image path for single frame (default: render.png)')
    parser.add_argument('--video', type=str, default=None,
                        help='Video path for background overlay')
    parser.add_argument('--render_video', action='store_true',
                        help='Render full video (requires --video)')
    parser.add_argument('--output_dir', type=str, default='output_render',
                        help='Output directory for video rendering (default: output_render)')
    parser.add_argument('--no_save_frames', action='store_true',
                        help='Skip saving individual frames (only save video)')
    parser.add_argument('--side_by_side', action='store_true',
                        help='Create side-by-side comparison (original | overlay)')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    
    # Load SMPLX model
    print("Loading SMPLX model...")
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()
    
    # Get person IDs to visualize
    if args.person_id is not None:
        person_ids = [args.person_id]
    else:
        person_ids = list(results['people'].keys())
    
    # Full video rendering mode
    if args.render_video:
        if args.video is None:
            print("Error: --render_video requires --video")
            return
        
        print(f"Rendering full video overlay...")
        render_video(
            results=results,
            smplx_model=smplx,
            video_path=args.video,
            output_dir=args.output_dir,
            person_ids=person_ids,
            save_frames=not args.no_save_frames,
            side_by_side=args.side_by_side
        )
        print("Done!")
        return
    
    # Single frame mode
    # Get camera parameters
    focal = results['camera']['img_focal']
    img_center = results['camera']['img_center']
    img_width = int(img_center[0] * 2)
    img_height = int(img_center[1] * 2)
    print(f"Image size: {img_width}x{img_height}, focal: {focal:.1f}")
    
    # Load background if video provided
    background = None
    if args.video:
        print(f"Loading frame {args.frame} from {args.video}")
        background = load_video_frame(args.video, args.frame)
        if background is None:
            print(f"Warning: Could not load frame {args.frame} from video")
    
    print(f"Visualizing frame {args.frame}, persons: {person_ids}")
    
    # Compute vertices for each person
    vertices_list = []
    for pid in person_ids:
        params = get_smplx_params_at_frame(results, pid, args.frame)
        if params is None:
            print(f"  Person {pid} not in frame {args.frame}, skipping")
            continue
        
        verts = compute_vertices(smplx, params)
        vertices_list.append(verts)
        print(f"  Person {pid}: vertices shape {verts.shape}")
    
    if len(vertices_list) == 0:
        print("No people found in specified frame!")
        return
    
    # Render and save
    faces = smplx.faces
    render_and_save(
        vertices_list, 
        faces, 
        args.output,
        focal_length=focal,
        img_width=img_width,
        img_height=img_height,
        background=background
    )
    print("Done!")


if __name__ == '__main__':
    main()
