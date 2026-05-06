# HumanMove_VQA

Data generation pipeline for human motion video QA. We generate training and evaluation data from EMDB, EgoBody, and RICH datasets using [PromptHMR](https://github.com/yufu-wang/PromptHMR) for 3D pose estimation and MotionScript for spatial motion captioning and question generation.

Fine-tuning uses LLamaFactory with Qwen3-VL 8B.

---

## Pipeline overview

```
Raw videos
    │
    ▼
[Step 1]  batch_process_videos.py        ← PromptHMR: detect, SLAM, SMPL-X estimation
    │      + --descriptor                 ← BLIP-2: clothing description per person
    │
    ▼  results.pkl  +  poses/*.npy
    │
[Step 2]  view_results.py                ← export Motion-X 322 .npz (pose + descriptor)
          visualize_gloss.py             ← render camera-space mesh overlay (QC)
    │
    ▼  person_<id>_motionx.npz  (N×322 pose  +  clothing string)
    │
[Step 3]  caption_spatial_only.py        ← MotionScript: spatial caption JSON per track
    │        (displacement X/Y/Z  +  rotation pitch/roll/yaw)
    │
    ▼  *_spatial_caption.json
    │
    ├─[Step 4a] generate_spatial_question_v5.py  ←  MC questions (7 categories per video)
    │            aggregate_by_category.py         ←  merge + balance across all videos
    │
    │            → SpatialQA_<dataset>/           (evaluation benchmark)
    │
    └─[Step 4b] prepare_sft_data.py               ←  train/test JSONL for LLamaFactory
                convert_cot_to_observe_reason_answer.py  ←  <reasoning><answer> format

                → results/sft/<dataset>/          (SFT training data)
```

---

## Repository layout

```
scripts/
├── prompthmr/          PromptHMR scripts (pose extraction, Motion-X export, visualization)
│   ├── README.md
│   ├── batch_process_videos.py
│   ├── view_results.py
│   ├── visualize_gloss.py
│   └── descriptor_utils.py
│
├── motionscript/       MotionScript scripts (captioning, QA generation, SFT data)
│   ├── README.md
│   ├── caption_spatial_only.py
│   ├── generate_spatial_question_v5.py
│   ├── aggregate_by_category.py
│   ├── prepare_sft_data.py
│   └── convert_cot_to_observe_reason_answer.py
│
└── llamafactory/       LLamaFactory config for Qwen3-VL 8B SFT
    ├── README.md
    └── sft_answer.yaml
```

### LLamaFactory SFT

`scripts/llamafactory/sft_answer.yaml` is the [hiyouga/LLamaFactory](https://github.com/hiyouga/LlamaFactory) training config for fine-tuning **Qwen3-VL 8B** with LoRA on the answer-only JSONL produced by `prepare_sft_data.py`. Set `dataset_dir` (pointing to your `dataset_info.json`) and `output_dir`, then:

```bash
llamafactory-cli train scripts/llamafactory/sft_answer.yaml
```

See `scripts/llamafactory/README.md` for dataset registration, key config options, and multi-GPU notes.

See each subfolder's `README.md` for script-level details, flags, and output formats.

---

## Environment setup

### PromptHMR

```bash
git clone https://github.com/yufu-wang/PromptHMR
cd PromptHMR
scripts/install.sh --pt_version=2.6 --world-video=true
bash scripts/fetch_smplx.sh
bash scripts/fetch_data.sh
```

```bash
export PROMPTHMR_ROOT=/path/to/PromptHMR
export PROMPTHMR_CACHE_DIR=/path/to/cache   # optional; defaults to ~/.cache/prompthmr
```

### MotionScript

Clone [pjyazdian/MotionScript](https://github.com/pjyazdian/MotionScript) and follow its setup instructions:

```bash
git clone https://github.com/pjyazdian/MotionScript
cd MotionScript
pip install -r requirements.txt
python setup.py develop
```

```bash
export MOTIONSCRIPT_ROOT=/path/to/MotionScript/src
export TEXT2POSE_PATH=~/posescript/src       # required by caption_spatial_only.py
```

---

## Example: RICH dataset

### Step 1 — World tracks + clothing descriptors

```bash
python scripts/prompthmr/batch_process_videos.py \
    --input /path/to/rich_videos/ \
    --output /path/to/results/rich_clothing/ \
    --descriptor
```

### Step 2 — Motion-X export + visualizations

```bash
vid_name=<video_stem>

python scripts/prompthmr/view_results.py \
    --results_dir /path/to/results/rich_clothing/$vid_name/ \
    --video /path/to/rich_videos/$vid_name.mp4 \
    --save_motionx \
    --motionx_output /path/to/motionx_poses/$vid_name/ \
    --render_video \
    --video_output /path/to/videos/$vid_name/world.mp4

python scripts/prompthmr/visualize_gloss.py \
    --results_dir /path/to/results/rich_clothing/$vid_name/ \
    --video /path/to/rich_videos/$vid_name.mp4 \
    --render_video \
    --output_dir /path/to/videos/
```

### Step 3 — Spatial caption per track

```bash
python scripts/motionscript/caption_spatial_only.py \
    --motion_path /path/to/motionx_poses/$vid_name/person_0_motionx.npz \
    --output_dir results/spatial/rich/$vid_name/ \
    --quant --normalize-first-frame
```

### Step 4a — Evaluation QA

```bash
python scripts/motionscript/generate_spatial_question_v5.py \
    --input results/spatial/rich/$vid_name/person_0_motionx_spatial_caption.json \
    --video_path /path/to/rich_videos/$vid_name.mp4 \
    --output_dir results/questions/rich/$vid_name/ \
    --questions_per_axis 10 --seed 42

python scripts/motionscript/aggregate_by_category.py \
    results/questions/rich/ \
    results/SpatialQA_RICH/
```

### Step 4b — SFT training data

```bash
python scripts/motionscript/prepare_sft_data.py \
    --input_dir results/spatial/rich/ \
    --output_dir results/sft/rich_v1/ \
    --video_base_dir /path/to/rich_videos/ \
    --cot

python scripts/motionscript/convert_cot_to_observe_reason_answer.py \
    --input results/sft/rich_v1/train_cot.jsonl \
    --output results/sft/rich_v1/train_reason_answer_short.jsonl \
    --reasoning-answer-only
```
