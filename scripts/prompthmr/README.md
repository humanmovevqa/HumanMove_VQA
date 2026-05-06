# PromptHMR scripts

Scripts for running [PromptHMR](https://github.com/yufu-wang/PromptHMR) on raw videos to extract world-space SMPL-X tracks, per-person clothing descriptors, Motion-X pose exports, and mesh visualizations.

---

## Setup

Install PromptHMR and download model weights following the upstream README, then set:

```bash
export PROMPTHMR_ROOT=/path/to/PromptHMR          # required
export PROMPTHMR_CACHE_DIR=/path/to/cache          # optional; defaults to ~/.cache/prompthmr
```

Activate the PromptHMR conda environment before running any script here.

---

## Scripts

### `batch_process_videos.py`

Runs the full PromptHMR pipeline (detection → SLAM → pose estimation) on a video or directory of videos. Saves per-person SMPL-X parameters in camera and world space.

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Video file or directory |
| `--output` | `results` | Root output directory |
| `--static_camera` | off | Assume static camera, skip SLAM (faster) |
| `--descriptor` | off | Run BLIP-2 to add a clothing description per person |
| `--force` | off | Reprocess even if output already exists |
| `--gpu` | `0` | CUDA device ID |
| `--extensions` | `mp4,avi,mov,mkv` | File extensions to glob over |

**Usage**

```bash
# Single video
python scripts/prompthmr/batch_process_videos.py \
    --input /path/to/video.mp4 \
    --output /path/to/results/

# Directory of videos with clothing descriptor
python scripts/prompthmr/batch_process_videos.py \
    --input /path/to/videos/ \
    --output /path/to/results/ \
    --descriptor
```

**Output layout**

```
results/
└── <video_name>/
    ├── results.pkl                 # full pipeline dict (people, camera, camera_world)
    └── poses/
        ├── camera/
        │   └── person_<id>.npy    # dict: frames(N,), pose(N,165), rotmat(N,55,3,3), shape(N,10), trans(N,3)
        ├── world/
        │   └── person_<id>.npy    # dict: frames(N,), pose(N,165), shape(N,10), trans(N,3) — Y-up, floor-fitted
        ├── camera_params.npy      # pred_cam_R/T, Rwc/Twc/Rcw/Tcw, img_focal, img_center
        ├── descriptors.json       # (if --descriptor) { "person_id": "clothing text", ... }
        └── descriptor_crops/      # (if --descriptor) cropped person images fed to BLIP-2
```

**Internal flow**

1. `Pipeline(static_cam=...)` loads SLAM, detector (SAM2 + ViTPose), and the PromptHMR video model.
2. Per frame: detect persons → estimate SMPL-X in camera space → lift to world space via SLAM.
3. `save_poses_npy` writes the `.npy` files above.
4. If `--descriptor`: loads `descriptor_utils.py`, picks up to 3 keyframes per person, crops with the tracked bbox, and runs BLIP-2 (`blip2-opt-2.7b`) on each crop; results are merged into one clothing phrase.

---

### `view_results.py`

Loads a `results.pkl` and provides three modes: interactive Viser viewer, Motion-X 322 export, or skeleton video rendering.

**Motion-X 322 export (main use)**

```bash
python scripts/prompthmr/view_results.py \
    --results_dir /path/to/results/<video_name>/ \
    --video /path/to/video.mp4 \
    --save_motionx \
    --motionx_output /path/to/motionx_poses/<video_name>/
```

Each person is written as `person_<id>_motionx.npz` containing:
- `motionx`: `(N, 322)` float32 — Motion-X 322 format (root_orient, pose_body, pose_hand, pose_jaw, zeros for face, trans, betas)
- `descriptor`: 0-d string array — clothing description (load with `.item()`)

**Motion-X 322 layout mapping from PromptHMR 165-dim pose:**

| Slice | Content | PromptHMR source |
|---|---|---|
| `[0:3]` | root_orient | `pose[0:3]` |
| `[3:66]` | pose_body (21 joints) | `pose[3:66]` |
| `[66:156]` | pose_hand (30 joints) | `pose[75:165]` |
| `[156:159]` | pose_jaw | `pose[66:69]` |
| `[159:309]` | face_expr + face_shape | zeros |
| `[309:312]` | trans | world translation |
| `[312:322]` | betas | shape (10) |

**Skeleton video**

```bash
python scripts/prompthmr/view_results.py \
    --results_dir /path/to/results/<video_name>/ \
    --video /path/to/video.mp4 \
    --render_video \
    --video_output /path/to/out/world.mp4
```

Renders a matplotlib 3-D skeleton with checkerboard floor (Y-up, isometric view). Coordinate swap for plotting: `plot_x = -input_x`, `plot_y = input_z`, `plot_z = input_y`.

**Viser viewer**

```bash
python scripts/prompthmr/view_results.py \
    --results_dir /path/to/results/<video_name>/ \
    --video /path/to/video.mp4 \
    --subsample 4    # show every 4th frame for faster loading
```

Opens an interactive 3-D viewer; open the printed URL in a browser.

---

### `visualize_gloss.py`

Renders SMPL-X meshes in **camera space** over original video frames using PyTorch3D.

**Full video overlay**

```bash
python scripts/prompthmr/visualize_gloss.py \
    --results_dir /path/to/results/<video_name>/ \
    --video /path/to/video.mp4 \
    --render_video \
    --output_dir /path/to/output/

# Side-by-side (original | mesh overlay)
python scripts/prompthmr/visualize_gloss.py \
    --results_dir /path/to/results/<video_name>/ \
    --video /path/to/video.mp4 \
    --render_video --side_by_side \
    --output_dir /path/to/output/
```

Output: `<output_dir>/<video_name>/overlay.mp4` (or `sidebyside.mp4`) and optionally per-frame PNGs in `frames/`.

**Single frame**

```bash
python scripts/prompthmr/visualize_gloss.py \
    --results_dir /path/to/results/<video_name>/ \
    --frame 42 \
    --video /path/to/video.mp4 \
    --output render_42.png
```

---

### `descriptor_utils.py`

Utility module imported by `batch_process_videos.py` (`--descriptor`). Not run directly.

**Key functions**

- `load_descriptor_vlm(cache_dir)` — downloads and loads BLIP-2 (`Salesforce/blip2-opt-2.7b`) with `float16` + `device_map=auto`. Weights cached under `cache_dir/huggingface/transformers/`.
- `add_person_descriptors(results, images, cache_dir, output_dir)` — for each tracked person: picks up to 3 keyframes (largest valid bboxes, temporally spread), crops with 10% padding, queries BLIP-2 with a clothing-focused prompt, merges descriptions with `_merge_clothing_descriptions`.
- `save_descriptors(results, output_dir)` — writes `poses/descriptors.json`.

**Prompt used for BLIP-2:**
> "Describe only the person's clothing and visible accessories. Mention garments, colors, patterns, materials, shoes, hats, glasses, bags, or jewelry if visible. Do not mention the person, pose, action, body position, camera view, background, skateboard, or scene. Return one short clothing-only phrase, not a sentence."

Post-processing strips pose/action stop-phrases and deduplicates clothing clauses.
