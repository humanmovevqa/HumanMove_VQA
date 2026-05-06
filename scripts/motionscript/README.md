# MotionScript scripts

Scripts for converting Motion-X `.npz` pose tracks into spatial motion captions, evaluation QA sets, and SFT training data for Qwen3-VL fine-tuning. Built on top of [pjyazdian/MotionScript](https://github.com/pjyazdian/MotionScript).

---

## Setup

Clone and install [MotionScript](https://github.com/pjyazdian/MotionScript):

```bash
git clone https://github.com/pjyazdian/MotionScript
cd MotionScript
pip install -r requirements.txt
python setup.py develop
```

Then set environment variables:

```bash
export MOTIONSCRIPT_ROOT=/path/to/MotionScript/src   # required for caption_spatial_only.py and prepare_sft_data.py
export TEXT2POSE_PATH=~/posescript/src               # required for caption_spatial_only.py (text2pose / PoseScript)
```

`generate_spatial_question_v5.py`, `aggregate_by_category.py`, and `convert_cot_to_observe_reason_answer.py` are self-contained (stdlib only) and need no extra path setup.

---

## Scripts

### `caption_spatial_only.py`

Converts a Motion-X `.npz` file into a spatial motion caption JSON. Tracks only **translation** (displacement X/Y/Z) and **root rotation** (pitch/roll/yaw) — no joint-level motioncodes.

**Usage**

```bash
python scripts/motionscript/caption_spatial_only.py \
    --motion_path /path/to/motionx_poses/<vid>/person_0_motionx.npz \
    --output_dir results/spatial/<dataset>/<vid>/ \
    --quant \
    --normalize-first-frame
```

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--motion_path` | required | `.npy` (raw 322) or `.npz` (`motionx` + optional `descriptor`) |
| `--output_dir` | `out_temp/spatial_captions` | Output directory |
| `--quant` | off | Store raw quantitative values before binning |
| `--normalize-first-frame` | off | Normalize all displacements and rotations relative to frame 0 |
| `--start_frame` / `--end_frame` | all frames | Optional frame range |

**Output files** (in `output_dir/`)

- `<motion_id>_spatial_caption.json` — full result dict:
  - `descriptor`: clothing text from `.npz` (or `""`)
  - `description`: aggregated spatial caption (natural language)
  - `description_non_aggregated`: pre-aggregation caption
  - `binning_detail`: per-motioncode bin assignments
  - `quantitative_values`: raw measurements (if `--quant`)
  - `category_definitions`: bin threshold ranges for all 6 spatial motioncodes
  - `motioncode_types`: `[displacement_x/y/z, rotation_pitch/roll/yaw]`
- `<motion_id>_spatial_caption.txt` — caption text only
- `<motion_id>_spatial_report.txt` — human-readable report

**Internal flow**

1. Loads Motion-X 322: extracts `trans = data[:, 309:312]` and `root_orient = data[:, :3]`.
2. Converts root orientation axis-angle → Euler angles (degrees), optionally normalised to frame 0.
3. Builds a `(N, 54, 3)` coordinate tensor: 52 zero joint slots + root orientation (index 52) + translation (index 53).
4. Monkey-patches MotionScript's `captioning.main` to filter posecode/motioncode operators to the 6 spatial types only.
5. Returns MotionScript caption output + category bin definitions.

---

### `generate_spatial_question_v5.py`

Generates template-based multiple-choice QA pairs from a `*_spatial_caption.json`. Produces 7 question categories per video.

**Usage**

```bash
python scripts/motionscript/generate_spatial_question_v5.py \
    --input results/spatial/<dataset>/<vid>/person_0_motionx_spatial_caption.json \
    --video_path /path/to/videos/<vid>.mp4 \
    --output_dir results/questions/<dataset>/<vid>/ \
    --questions_per_axis 10 \
    --seed 42
```

**Output**: seven `qa_<category>.json` files, one per axis:

| Category | Question type |
|---|---|
| `numerical` | "How many times did X move Y?" (count-based) |
| `comparative` | "Did the person move more left or right?" (2-way or 3-way MC) |
| `dominant` | "Which direction dominated overall?" (3-way or 4-way MC) |
| `temporal` | "When did the first significant X movement occur?" (4-way MC) |
| `ordering` | "Which movement happened first, X or Y?" (2-way MC) |
| `trajectory_affordance` | "In which quarter was the person furthest from start?" (4-way MC) |
| `existence` | "Did the person move significantly to the left?" (yes/no) |

Each `qa_<category>.json` contains `video_path`, `qa_pairs` (list of `{question, options, answer, evidence}`).

---

### `aggregate_by_category.py`

Merges all per-video `qa_<category>.json` files from one or more input directories and writes one output file per category with balanced correct-option distribution.

**Usage**

```bash
python scripts/motionscript/aggregate_by_category.py \
    results/questions/<dataset>/ \
    results/SpatialQA_<dataset>/

# Merge multiple datasets
python scripts/motionscript/aggregate_by_category.py \
    results/questions/<dataset_a>/ \
    results/questions/<dataset_b>/ \
    results/SpatialQA_combined/
```

**Answer-key balancing**: for each category the correct letter is round-robined across entries (A–D for most, A–C for `comparative`/`dominant`, A–B for `existence`) by swapping option texts. Pass `--no-balance` to skip.

**Output**: `<output_dir>/<category>/<category>.json` — one JSONL-like file per category with fields `id`, `video_path`, `question` (with options embedded), `answer`.

---

### `prepare_sft_data.py`

Samples QA pairs from all `*_spatial_caption.json` files, splits by video into train/test, balances answer-key distribution, and writes LLamaFactory-compatible JSONL.

**Usage**

```bash
python scripts/motionscript/prepare_sft_data.py \
    --input_dir results/spatial/<dataset>/ \
    --output_dir results/sft/<dataset>_v1/ \
    --video_base_dir /path/to/videos/ \
    --cot
```

**Key flags**

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | required | One or more dirs with `*_spatial_caption.json` (searched recursively) |
| `--output_dir` | required | Output directory |
| `--video_base_dir` | `None` | Override video path in output JSONL (`<base>/<vid>.mp4`) |
| `--samples_per_category` | `10` | Max questions to sample per QA category per video |
| `--cot` | off | Also write `*_cot.jsonl` with raw `evidence` dict per entry |
| `--descriptor` | off | Personalize questions with the clothing descriptor from the JSON |
| `--no-balance` | off | Skip answer-key balancing |

**Output files**

- `train_answer_only.jsonl` / `test_answer_only.jsonl` — each line: `{video, task, conversations: [{user}, {assistant: "Answer: X"}]}`
- `train_cot.jsonl` / `test_cot.jsonl` (with `--cot`) — same + `evidence` key with raw QA evidence dict

Split is by video (not by frame): 90% train / 10% test by default.

---

### `convert_cot_to_observe_reason_answer.py`

Post-processes `*_cot.jsonl` entries: replaces the `"Answer: X"` assistant response with a structured `<observe> <reasoning> <answer>` format derived from the `evidence` dict.

**Usage**

```bash
# Full <observe><reasoning><answer> format
python scripts/motionscript/convert_cot_to_observe_reason_answer.py \
    --input results/sft/<dataset>_v1/train_cot.jsonl \
    --output results/sft/<dataset>_v1/train_ora.jsonl

# Reasoning + answer only (no <observe> block)
python scripts/motionscript/convert_cot_to_observe_reason_answer.py \
    --input results/sft/<dataset>_v1/train_cot.jsonl \
    --output results/sft/<dataset>_v1/train_reason_answer_short.jsonl \
    --reasoning-answer-only
```

**Output format** (per assistant turn):

```
<observe>
[{"code": "displacement_x", "intensity": 2.3, ...}, ...]
</observe>
<reasoning>left=2.3; right=0.8; choice=left; option=A</reasoning>
<answer>A</answer>
```

With `--reasoning-answer-only`: just `<reasoning>...</reasoning>\n<answer>A</answer>`.

**Reasoning format per task type:**

| Task | Reasoning content |
|---|---|
| `numerical` | `value=N; choice=<label>; option=X` |
| `comparative` | `left=X; right=Y; choice=<dominant>; option=X` |
| `dominant` | all 4-way totals + `choice=<dominant>; option=X` |
| `ordering` | `anchor=<event_short>; next_event=<event_short>; choice=<bucket>; option=X` |
| `trajectory_affordance` | per-quarter metric + `choice=<quarter>; option=X` |
| `existence` | `present/absent; option=X` |
