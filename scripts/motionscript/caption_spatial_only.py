#!/usr/bin/env python3
"""
Generate motioncode captions for spatial trackers only (translation + rotation).

Only tracks:
- displacement_x/y/z (from translation)
- rotation_pitch/roll/yaw (from root orientation)

Excludes all joint-based motioncodes (angular, proximity, spatial_relation).

Usage:
    python caption_spatial_only.py --motion_path /path/to/motion.npy
    python caption_spatial_only.py --motion_path /path/to/motion.npz --output_dir ./output

    Supports .npy (raw 322 array) or .npz with 'motionx' and optional 'descriptor'.
    If descriptor is present in the file, it is written to the output JSON as "descriptor"; else "".
"""

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Set EGL for GPU-accelerated headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ============================================================================
# MotionScript root — set MOTIONSCRIPT_ROOT to your MotionScript/src checkout.
# PoseScript root  — set TEXT2POSE_PATH to your posescript/src checkout.
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIONSCRIPT_ROOT = os.environ.get(
    "MOTIONSCRIPT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "MotionScript", "src"),
)
sys.path.insert(0, _SCRIPT_DIR)      # for scripts living alongside this file
sys.path.insert(0, MOTIONSCRIPT_ROOT)

TEXT2POSE_PATH = os.environ.get("TEXT2POSE_PATH", os.path.expanduser("~/posescript/src"))
sys.path.insert(0, TEXT2POSE_PATH)

import numpy as np
import torch

# MotionScript imports
import captioning as captioning_py
import text2pose.utils as utils

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEFAULT_OUTPUT_DIR = os.path.join(SRC_DIR, "out_temp", "spatial_captions")

# =============================================================================
# Rotation Utilities
# =============================================================================

def rotvec_to_euler(rotvec):
    """Convert rotation vector to euler angles."""
    return utils.rotvec_to_eulerangles(rotvec)

# =============================================================================
# Data Loading
# =============================================================================

def load_motionx_npy(npy_path: str):
    """
    Load Motion-X SMPL-X 322 format from .npy or .npz file.
    
    .npz may contain 'motionx' (N, 322) and optionally 'descriptor' (str, from PromptHMR view_results).
    
    Returns:
        Tuple of (translation, root_orient, descriptor). descriptor is "" if not present.
    """
    loaded = np.load(npy_path, allow_pickle=True)
    if npy_path.lower().endswith('.npz'):
        data = torch.tensor(loaded['motionx']).float().to(DEVICE)
        descriptor = ""
        if 'descriptor' in loaded:
            d = loaded['descriptor']
            descriptor = d.item() if d.ndim == 0 else str(d)
            if descriptor and not isinstance(descriptor, str):
                descriptor = str(descriptor)
    else:
        data = torch.tensor(loaded).float().to(DEVICE)
        descriptor = ""
    
    # Parse SMPL-X 322 format
    root_orient = data[:, :3]           # (N, 3) - root orientation (axis-angle)
    trans = data[:, 309:312]            # (N, 3) - translation
    
    return trans, root_orient, descriptor

# =============================================================================
# Coordinate Tensor Building (Spatial Only)
# =============================================================================

def build_spatial_coords(trans, root_orient, num_frames, normalize_first_frame=False):
    """
    Build coordinate tensor with only spatial data (no joints).
    
    For spatial motioncodes, we need:
    - position_x/y/z posecodes (from translation)
    - orientation_pitch/roll/yaw posecodes (from root_orient)
    
    The coordinate tensor structure:
    - Joint positions: zeros (52 joints × 3) - not used for spatial motioncodes
    - Root orientation: euler angles (1 × 3)
    - Translation: XYZ position (1 × 3)
    
    Args:
        trans: Translation tensor (N, 3)
        root_orient: Root orientation tensor (N, 3) - axis-angle
        num_frames: Number of frames
        normalize_first_frame: If True, normalize all values relative to first frame
    
    Returns:
        coords: (N, 54, 3) tensor compatible with MotionScript
    """
    # Convert root orientation to euler angles
    rad2deg = lambda theta_rad: 180.0 * theta_rad / math.pi
    root_euler_orient = torch.zeros_like(root_orient).to(DEVICE)
    
    for frame in range(root_euler_orient.shape[0]):
        theta_x, theta_y, theta_z = rotvec_to_euler(root_orient[frame, :].unsqueeze(0))
        root_euler_orient[frame, :] = torch.cat([
            torch.unsqueeze(rad2deg(theta_x), 0),
            torch.unsqueeze(rad2deg(theta_y), 0),
            torch.unsqueeze(rad2deg(theta_z), 0)
        ]).squeeze()
    
    # Normalize angles helper function
    def normalize_angles(angles):
        angles = angles % 360
        angles[angles > 180] -= 360
        return angles
    
    # Apply first frame normalization if requested
    if normalize_first_frame:
        # Normalize all rotations relative to first frame
        for axis in range(3):  # pitch (0), roll (1), yaw (2)
            angle_diff = root_euler_orient[0, axis]
            root_euler_orient[:, axis] = normalize_angles(root_euler_orient[:, axis] - angle_diff)
        
        # Normalize all translations relative to first frame
        trans = trans - trans[0:1, :]  # Subtract first frame's position
    else:
        # Only normalize yaw (original behavior)
        angle_diff = root_euler_orient[0, 2]
        root_euler_orient[:, 2] = normalize_angles(root_euler_orient[:, 2] - angle_diff)
    
    # Build coordinate tensor
    # MotionScript expects: (frames, 52 joints + 2 virtual joints, 3)
    # Virtual joints: orientation (index 52), translation (index 53)
    
    # Create zero tensor for all 52 joints (not used for spatial motioncodes)
    joint_coords = torch.zeros(num_frames, 52, 3).to(DEVICE)
    
    # Add root orientation as virtual joint (index 52)
    coords = torch.cat([joint_coords, root_euler_orient.unsqueeze(1)], dim=1)
    
    # Add translation as virtual joint (index 53)
    coords = torch.cat([coords, trans.unsqueeze(1)], dim=1)
    
    # Final shape: (N, 54, 3)
    coords = coords.view(coords.shape[0], -1, 3)
    
    return coords

# =============================================================================
# Caption Generation (Spatial Only)
# =============================================================================

def generate_spatial_caption(
    motion_path: str,
    motion_id: str = None,
    start_frame: int = None,
    end_frame: int = None,
    verbose: bool = True,
    quant: bool = False,
    normalize_first_frame: bool = False
):
    """
    Generate caption for spatial motioncodes only.
    
    Args:
        motion_path: Path to .npy motion file
        motion_id: Optional ID for the motion (defaults to filename)
        start_frame: Start frame (default: 0)
        end_frame: End frame (default: all frames)
        verbose: Print progress info
        quant: Store quantitative motioncode values
        normalize_first_frame: If True, normalize pitch/roll/yaw and disp_x/y/z relative to first frame
        
    Returns:
        Dictionary with caption results.
    """
    if motion_id is None:
        motion_id = os.path.splitext(os.path.basename(motion_path))[0]
    
    if verbose:
        print(f"Processing: {motion_id}")
        print(f"Loading: {motion_path}")
    
    # Load motion data (and optional descriptor from .npz)
    trans, root_orient, descriptor = load_motionx_npy(motion_path)
    
    fps = 30.0  # Motion-X is typically 30 FPS
    num_frames = trans.shape[0]
    
    if verbose:
        print(f"Loaded {num_frames} frames ({num_frames/fps:.2f} seconds)")
    
    # Set frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = num_frames
    
    # Validate frame range
    start_frame = max(0, start_frame)
    end_frame = min(num_frames, end_frame)
    
    if end_frame - start_frame < 10:
        return {
            'motion_id': motion_id,
            'error': 'Motion too short (< 10 frames)',
            'descriptor': descriptor if descriptor else '',
            'description': '',
            'num_frames': num_frames
        }
    
    # Extract frame range
    trans = trans[start_frame:end_frame]
    root_orient = root_orient[start_frame:end_frame]
    num_frames_segment = trans.shape[0]
    
    # Build coordinate tensor (spatial only)
    coords = build_spatial_coords(trans, root_orient, num_frames_segment, 
                                   normalize_first_frame=normalize_first_frame)
    
    if verbose:
        print(f"Coordinate tensor shape: {coords.shape}")
        print("Generating spatial caption (displacement + rotation only)...")
    
    # Generate caption using MotionScript
    # Filter to only spatial motioncodes (displacement + rotation)
    save_dir = os.path.join(DEFAULT_OUTPUT_DIR, "generated_captions", "tmp")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Call captioning.main() but we need to filter to spatial motioncodes only
        # The eligibility filtering in captioning.py (line 481-499) uses set_of_accepted_mkinds
        # We'll temporarily patch that list to only include spatial motioncodes
        
        import captioning
        
        # Monkey-patch the eligibility filtering by wrapping main()
        # We'll intercept the eligibility adjustment step
        original_main = captioning.main
        
        def spatial_filtered_main(coords, save_dir, **kwargs):
            """Only compute spatial posecodes (global translation and rotation) from the start."""
            # Define allowed types
            allowed_posecode_types = ['position_x', 'position_y', 'position_z',
                                     'orientation_pitch', 'orientation_roll', 'orientation_yaw']
            allowed_motioncode_types = ['displacement_x', 'displacement_y', 'displacement_z',
                                       'rotation_pitch', 'rotation_roll', 'rotation_yaw']
            
            # Get quant flag from kwargs
            quant_flag = kwargs.get('quant', False)
            
            # Patch prepare_posecode_queries to only prepare spatial posecode queries
            original_prepare_posecode_queries = captioning.prepare_posecode_queries
            def patched_prepare_posecode_queries():
                """Only prepare queries for spatial posecode types."""
                from captioning_data import ALL_ELEMENTARY_POSECODES
                
                # Filter ALL_ELEMENTARY_POSECODES to only spatial types
                filtered_elementary_posecodes = {k: v for k, v in ALL_ELEMENTARY_POSECODES.items() 
                                                if k in allowed_posecode_types}
                
                # Temporarily replace ALL_ELEMENTARY_POSECODES
                original_all_elementary = dict(ALL_ELEMENTARY_POSECODES)
                ALL_ELEMENTARY_POSECODES.clear()
                ALL_ELEMENTARY_POSECODES.update(filtered_elementary_posecodes)
                
                try:
                    # Call original function - it will only process spatial posecodes
                    result = original_prepare_posecode_queries()
                finally:
                    # Restore original
                    ALL_ELEMENTARY_POSECODES.clear()
                    ALL_ELEMENTARY_POSECODES.update(original_all_elementary)
                
                return result
            
            # Patch prepare_super_posecode_queries to return empty (super-posecodes depend on non-spatial posecodes)
            original_prepare_super_posecode_queries = captioning.prepare_super_posecode_queries
            def patched_prepare_super_posecode_queries(p_queries):
                """Return empty dict - super-posecodes not needed for spatial-only tracking."""
                return {}
            
            # Patch infer_posecodes to only compute spatial posecodes
            original_infer_posecodes = captioning.infer_posecodes
            def patched_infer_posecodes(coords, p_queries, sp_queries, request='Default', verbose=True):
                """Only compute spatial posecodes."""
                from posecodes import POSECODE_OPERATORS
                import posecodes
                
                # Filter p_queries to only spatial types
                filtered_p_queries = {k: v for k, v in p_queries.items() 
                                     if k in allowed_posecode_types}
                
                # Filter POSECODE_OPERATORS to only spatial types
                original_operators = dict(POSECODE_OPERATORS)
                filtered_operators = {k: v for k, v in POSECODE_OPERATORS.items() 
                                    if k in allowed_posecode_types}
                posecodes.POSECODE_OPERATORS = filtered_operators
                if hasattr(captioning, 'POSECODE_OPERATORS'):
                    captioning.POSECODE_OPERATORS = filtered_operators
                
                try:
                    # Call original - it will only iterate over spatial posecode types
                    p_interpretations, p_eligibility = original_infer_posecodes(
                        coords, filtered_p_queries, sp_queries, request, verbose
                    )
                finally:
                    # Restore original
                    posecodes.POSECODE_OPERATORS = original_operators
                    if hasattr(captioning, 'POSECODE_OPERATORS'):
                        captioning.POSECODE_OPERATORS = original_operators
                
                return p_interpretations, p_eligibility
            
            original_infer_motioncodes = captioning.infer_motioncodes
            
            def patched_infer_motioncodes(coords, p_interpretations, p_queries, sp_queries, m_queries,
                                         request='Default', verbose=True, quant=False):
                """Filter to spatial-only and override MOTIONCODE_OPERATORS.items() to only return spatial types."""
                # Filter p_interpretations to only spatial posecode types
                filtered_p_interpretations = {k: v for k, v in p_interpretations.items() 
                                            if k in allowed_posecode_types}
                
                # Filter p_queries to only spatial posecode types
                filtered_p_queries = {k: v for k, v in p_queries.items() 
                                     if k in allowed_posecode_types}
                
                # Filter m_queries to only spatial motioncode types
                filtered_m_queries = {k: v for k, v in m_queries.items() 
                                     if k in allowed_motioncode_types}
                
                # Use quant flag from outer scope (captured in closure)
                quant_flag_to_use = quant_flag if 'quant_flag' in locals() or 'quant_flag' in globals() else quant
                
                # Override MOTIONCODE_OPERATORS.items() to only return spatial motioncode types
                from posecodes import MOTIONCODE_OPERATORS
                import posecodes
                
                # Save original
                original_operators = dict(MOTIONCODE_OPERATORS)
                
                # Filter to only spatial motioncode types
                filtered_operators = {k: v for k, v in MOTIONCODE_OPERATORS.items() 
                                    if k in allowed_motioncode_types}
                posecodes.MOTIONCODE_OPERATORS = filtered_operators
                # Also update in captioning module (it imports via *)
                if hasattr(captioning, 'MOTIONCODE_OPERATORS'):
                    captioning.MOTIONCODE_OPERATORS = filtered_operators
                
                try:
                    # Now call original - it will only iterate over spatial motioncode types
                    m_interpretations, m_eligibility = original_infer_motioncodes(
                        coords, filtered_p_interpretations, filtered_p_queries, sp_queries, filtered_m_queries,
                        request, verbose, quant=quant_flag_to_use
                    )
                finally:
                    # Restore original
                    posecodes.MOTIONCODE_OPERATORS = original_operators
                    if hasattr(captioning, 'MOTIONCODE_OPERATORS'):
                        captioning.MOTIONCODE_OPERATORS = original_operators
                
                return m_interpretations, m_eligibility
            
            # Also need to patch the eligibility adjustment section that hardcodes 'proximity'
            # This happens in main() at lines 501-505 - it tries to access m_eligibility['proximity']
            # We'll wrap infer_motioncodes to add an empty 'proximity' entry if it doesn't exist
            def patched_infer_motioncodes_with_proximity(*args, **kwargs):
                m_interpretations, m_eligibility = patched_infer_motioncodes(*args, **kwargs)
                
                # Add empty 'proximity' entry if it doesn't exist (for eligibility adjustment code)
                if 'proximity' not in m_eligibility:
                    # Create empty structure matching the expected format
                    # Based on the code, m_eligibility[m_kind] is a list of lists of tuples
                    # We'll create an empty one
                    m_eligibility['proximity'] = []
                
                return m_interpretations, m_eligibility
            
            # Override functions
            captioning.prepare_posecode_queries = patched_prepare_posecode_queries
            captioning.prepare_super_posecode_queries = patched_prepare_super_posecode_queries
            captioning.infer_posecodes = patched_infer_posecodes
            captioning.infer_motioncodes = patched_infer_motioncodes_with_proximity
            
            try:
                result = original_main(coords, save_dir, **kwargs)
            finally:
                # Restore originals
                captioning.prepare_posecode_queries = original_prepare_posecode_queries
                captioning.prepare_super_posecode_queries = original_prepare_super_posecode_queries
                captioning.infer_posecodes = original_infer_posecodes
                captioning.infer_motioncodes = original_infer_motioncodes
            
            return result
        
        # Call with spatial filter
        (binning_detail, 
         motioncodes_vis, 
         description_non_agg, 
         description,
         quant_values) = spatial_filtered_main(
            coords,
            save_dir=save_dir,
            babel_info=False,
            simplified_captions=False,
            apply_transrel_ripple_effect=False,
            apply_stat_ripple_effect=False,
            random_skip=False,
            motion_tracking=True,
            verbose=verbose,
            ablations=[],
            quant=quant
        )
    except Exception as e:
        return {
            'motion_id': motion_id,
            'error': str(e),
            'descriptor': descriptor if descriptor else '',
            'description': '',
            'num_frames': num_frames
        }
    
    # Clean up descriptions
    description_text = " ".join([x for x in description if x != '']).strip()
    description_non_agg_text = " ".join([x for x in description_non_agg if x != '']).strip()
    
    start_time = float(start_frame) / fps
    end_time = float(end_frame) / fps
    
    result = {
        'motion_id': motion_id,
        'motion_path': motion_path,
        'descriptor': descriptor if descriptor else '',
        'description': description_text,
        'description_non_aggregated': description_non_agg_text,
        'binning_detail': binning_detail,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'start_time': start_time,
        'end_time': end_time,
        'num_frames': num_frames,
        'fps': fps,
        'duration': end_time - start_time,
        'motioncode_types': ['displacement_x', 'displacement_y', 'displacement_z',
                            'rotation_pitch', 'rotation_roll', 'rotation_yaw']
    }
    
    # Add category definitions (bin thresholds) globally
    from captioning_data import MOTIONCODE_OPERATORS_VALUES
    category_definitions = {}
    spatial_motioncodes = ['displacement_x', 'displacement_y', 'displacement_z',
                          'rotation_pitch', 'rotation_roll', 'rotation_yaw']
    
    for m_kind in spatial_motioncodes:
        if m_kind in MOTIONCODE_OPERATORS_VALUES:
            m_ops = MOTIONCODE_OPERATORS_VALUES[m_kind]
            
            # Build spatial category bins with threshold ranges
            spatial_categories = []
            category_names = m_ops.get('category_names', [])
            category_thresholds = m_ops.get('category_thresholds', [])
            
            # Create bin definitions: each category corresponds to a range
            for i, cat_name in enumerate(category_names):
                if i == 0:
                    # First category: from -infinity to first threshold
                    bin_def = {
                        'category': cat_name,
                        'range': f'< {category_thresholds[0]}' if category_thresholds else 'all values'
                    }
                elif i == len(category_names) - 1:
                    # Last category: from last threshold to +infinity
                    bin_def = {
                        'category': cat_name,
                        'range': f'> {category_thresholds[-1]}'
                    }
                else:
                    # Middle categories: between thresholds
                    bin_def = {
                        'category': cat_name,
                        'range': f'{category_thresholds[i-1]} to {category_thresholds[i]}'
                    }
                spatial_categories.append(bin_def)
            
            # Build temporal category bins with threshold ranges
            temporal_categories = []
            velocity_names = m_ops.get('category_names_velocity', [])
            velocity_thresholds = m_ops.get('category_thresholds_velocity', [])
            
            for i, vel_name in enumerate(velocity_names):
                if i == 0:
                    bin_def = {
                        'category': vel_name,
                        'range': f'≤ {velocity_thresholds[0]}' if velocity_thresholds else 'all values'
                    }
                elif i == len(velocity_names) - 1:
                    bin_def = {
                        'category': vel_name,
                        'range': f'> {velocity_thresholds[-1]}'
                    }
                else:
                    bin_def = {
                        'category': vel_name,
                        'range': f'{velocity_thresholds[i-1]} < |velocity| ≤ {velocity_thresholds[i]}'
                    }
                temporal_categories.append(bin_def)
            
            category_definitions[m_kind] = {
                'spatial_categories': {
                    'bins': spatial_categories,
                    'category_names': category_names,
                    'category_thresholds': category_thresholds,
                    'units': 'units (1 unit = 10cm)' if 'displacement' in m_kind else 'degrees',
                    'description': 'Spatial categories based on intensity (magnitude of motion). Values are compared against thresholds.'
                },
                'temporal_categories': {
                    'bins': temporal_categories,
                    'category_names': velocity_names,
                    'category_thresholds': velocity_thresholds,
                    'units': 'units/frame (at 30fps: multiply by 30 for units/second)',
                    'description': 'Temporal categories based on velocity (rate of change). Absolute value of velocity is compared against thresholds.'
                }
            }
    
    result['category_definitions'] = category_definitions
    
    # Add quantitative values if available
    if quant and quant_values:
        result['quantitative_values'] = quant_values
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Motion ID: {motion_id}")
        print(f"Duration: {result['duration']:.2f}s ({end_frame - start_frame} frames)")
        print(f"{'='*60}")
        print(f"\nSpatial Caption:\n{description_text}")
        print(f"{'='*60}\n")
    
    return result

def save_result(result: dict, output_dir: str):
    """Save caption result to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    motion_id = result['motion_id']
    
    # Ensure binning_detail is a string (in case it's None or other type)
    if 'binning_detail' in result and result['binning_detail'] is not None:
        binning_detail = str(result['binning_detail'])
    else:
        binning_detail = ""
    
    # Update result with string version
    result['binning_detail'] = binning_detail
    
    # Save JSON with all info
    json_path = os.path.join(output_dir, f"{motion_id}_spatial_caption.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save plain text caption
    txt_path = os.path.join(output_dir, f"{motion_id}_spatial_caption.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(result.get('description', ''))
    
    # Save detailed report (similar to caption_single_motionx.py)
    report_path = os.path.join(output_dir, f"{motion_id}_spatial_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Motion ID: {motion_id}\n")
        f.write(f"Source: {result.get('motion_path', 'N/A')}\n")
        f.write(f"Descriptor: {result.get('descriptor', '') or '(none)'}\n")
        f.write(f"Duration: {result.get('duration', 0):.2f}s\n")
        f.write(f"Frames: {result.get('start_frame', 0)} - {result.get('end_frame', 0)}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CAPTION (Aggregated):\n")
        f.write(f"{'='*60}\n")
        f.write(f"{result.get('description', '')}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"CAPTION (Non-Aggregated):\n")
        f.write(f"{'='*60}\n")
        f.write(f"{result.get('description_non_aggregated', '')}\n")
        if binning_detail:
            f.write(f"\n{'='*60}\n")
            f.write(f"BINNING DETAIL:\n")
            f.write(f"{'='*60}\n")
            f.write(f"{binning_detail}\n")
    
    print(f"Results saved to: {output_dir}")
    print(f"  - {os.path.basename(json_path)}")
    print(f"  - {os.path.basename(txt_path)}")
    print(f"  - {os.path.basename(report_path)}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate spatial motioncode captions (translation + rotation only)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--motion_path', '-m',
        type=str,
        required=True,
        help='Path to Motion-X .npy or .npz file (SMPL-X 322; .npz may include descriptor)'
    )
    
    parser.add_argument(
        '--motion_id', '-i',
        type=str,
        default=None,
        help='Motion ID (default: filename without extension)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--start_frame', '-s',
        type=int,
        default=None,
        help='Start frame (default: 0)'
    )
    
    parser.add_argument(
        '--end_frame', '-e',
        type=int,
        default=None,
        help='End frame (default: all frames)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--quant',
        action='store_true',
        help='Store quantitative values (raw measurements) for motioncodes before categorization'
    )
    
    parser.add_argument(
        '--normalize-first-frame',
        dest='normalize_first_frame',
        action='store_true',
        help='Normalize pitch/roll/yaw and displacement x/y/z relative to first frame'
    )
    
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results to files'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.motion_path):
        print(f"Error: Motion file not found: {args.motion_path}")
        sys.exit(1)
    
    # Generate caption
    result = generate_spatial_caption(
        motion_path=args.motion_path,
        motion_id=args.motion_id,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        verbose=not args.quiet,
        quant=args.quant,
        normalize_first_frame=args.normalize_first_frame
    )
    
    # Check for errors
    if result.get('error'):
        print(f"Error: {result['error']}")
        sys.exit(1)
    
    # Save results
    if not args.no_save:
        save_result(result, args.output_dir)
    
    # Print caption to stdout
    if args.quiet:
        print(result['description'])

if __name__ == '__main__':
    main()

