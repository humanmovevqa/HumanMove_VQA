import json
import os
import random
import glob
from collections import defaultdict
from pathlib import Path
import sys
import argparse
from typing import Sequence, Union
import time

# ============================================================================
# MotionScript root — set MOTIONSCRIPT_ROOT to your MotionScript/src checkout
# (needed for cot_formatting and other MotionScript utilities).
# ============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIONSCRIPT_ROOT = os.environ.get(
    "MOTIONSCRIPT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(_SCRIPT_DIR)), "MotionScript", "src"),
)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, MOTIONSCRIPT_ROOT)

from aggregate_by_category import _balance_mc_distribution, BALANCE_EXCLUDED

import generate_spatial_question_v5 as generator
from cot_formatting import rephrase_batch, load_rewriter_pipeline


def _qa_with_task(qa_list, task_name: str):
    return [{**qa, "task": task_name} for qa in qa_list]


def _apply_mc_balance_to_rows(rows: list[dict]) -> None:
    """
    Per task, rebalance correct-option letters like aggregate_by_category:
    existence A/B (~50/50), comparative & dominant A/B/C (~33% each),
    all other tasks A–D (~25% each). Rows whose options do not match the
    expected key set for that task are left unchanged.
    """
    by_task: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_task[row["qa"]["task"]].append(idx)
    for task, idxs in by_task.items():
        if task in BALANCE_EXCLUDED:
            continue
        # Sort idxs by the same key _balance_mc_distribution uses internally so
        # balanced[j] maps back to the correct rows[i] after the call.
        sorted_idxs = sorted(
            idxs,
            key=lambda i: (rows[i]["video_path"], rows[i]["qa"]["question"]),
        )
        entries = [
            {
                "video_path": rows[i]["video_path"],
                "question_stem": rows[i]["qa"]["question"],
                "options": rows[i]["qa"]["options"],
                "answer_letter": rows[i]["qa"]["answer"],
            }
            for i in sorted_idxs
        ]
        balanced = _balance_mc_distribution(task, entries)
        for j, i in enumerate(sorted_idxs):
            rows[i]["qa"]["options"] = balanced[j]["options"]
            rows[i]["qa"]["answer"] = balanced[j]["answer_letter"]

# We use the local HuggingFace transformers library for rewriting.
# This avoids external API dependencies but requires GPU memory.


def _normalize_input_dirs(raw: Union[str, Sequence[str], None]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def create_dataset(
    input_dirs: Union[str, Sequence[str]],
    output_dir: str,
    train_ratio: float = 0.9,
    samples_per_category: int = 10,
    rewrite_prob: float = 0.0,
    model_path: str = None,
    rewrite_batch_size: int = 1,
    video_base_dir: str = None,
    cot: bool = False,
    descriptor: bool = False,
    no_balance: bool = False,
):
    """
    Generates SFT data for QwenVL.
    input_dirs: single path or sequence of root directories (each searched recursively for *_spatial_caption.json).
    Always writes {split}_answer_only.jsonl (assistant responds with "Answer: X").
    Each entry includes "task" (numerical, comparative, dominant, temporal, ordering,
    trajectory_affordance, existence).
    After collecting all QAs for the split, correct answers are rebalanced per task by
    swapping option texts (same logic as aggregate_by_category): comparative/dominant
    use A–C (~33% each), existence A/B (~50/50), other tasks A–D (~25% each).
    Pass no_balance=True to skip this step and keep the as-generated option order.
    If cot=True, also writes {split}_cot.jsonl: same structure but each entry has an extra key "evidence" with the QA evidence dict.
    """
    
    # Initialize rewriter if needed
    rewriter = None
    if rewrite_prob > 0 and model_path:
        rewriter = load_rewriter_pipeline(model_path)
        if rewriter is None:
            return

    # 1. Find all spatial caption JSONs
    dirs = _normalize_input_dirs(input_dirs)
    if not dirs:
        print("No input directories given.")
        return
    json_files = []
    for d in dirs:
        json_files.extend(glob.glob(os.path.join(d, "**", "*_spatial_caption.json")))
    # Stable dedupe (same file reachable from overlapping roots)
    json_files = list(dict.fromkeys(json_files))
    if not json_files:
        print(f"No *_spatial_caption.json files found under: {', '.join(dirs)}")
        return

    # 2. Split by video (parent dir of each JSON), so all JSONs from same video go to same split
    def video_id_from_path(p):
        return os.path.basename(os.path.dirname(p))

    video_to_files = {}
    for p in json_files:
        vid = video_id_from_path(p)
        video_to_files.setdefault(vid, []).append(p)

    unique_videos = list(video_to_files.keys())
    random.shuffle(unique_videos)
    split_idx = int(len(unique_videos) * train_ratio)
    train_videos = set(unique_videos[:split_idx])
    test_videos = set(unique_videos[split_idx:])

    train_files = [f for f in json_files if video_id_from_path(f) in train_videos]
    test_files = [f for f in json_files if video_id_from_path(f) in test_videos]

    print(f"Found {len(json_files)} JSONs from {len(unique_videos)} videos.")
    print(f"Split by video: {len(train_videos)} train videos ({len(train_files)} JSONs), {len(test_videos)} test videos ({len(test_files)} JSONs)")

    # 3. Process function: writes answer-only; optionally also CoT variant
    def process_files(file_list, split_name):
        all_rows: list[dict] = []

        for json_path in file_list:
            try:
                # Load Data
                motion_json = generator.load_motion_json(json_path)
                subject_descriptor = generator.resolve_subject_descriptor(motion_json) if descriptor else None
                # Match generate_spatial_question_v5 --description: prefix when subject phrase is absent.
                require_descriptor_in_text = bool(descriptor and subject_descriptor)
                events = generator.extract_events(motion_json)
                events = generator._exclude_very_short_events(events)
                
                # Get Stats
                num_frames = motion_json.get("num_frames", len(events))
                fps = motion_json.get("fps", 30.0)
                rng = random.Random(hash(json_path)) # Deterministic per file
                stats = generator.compute_stats(events, num_frames, fps, rng)
                duration = stats["duration"]

                # Generate POOLS per category; sample using same rules as generate_spatial_question_v5
                # Axis categories: pool then sample_questions(pool, n, rng) as in v5 main()
                numerical_qa = generator.sample_questions(
                    generator.generate_numerical_qa_pool(events, stats, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                comparative_qa = generator.sample_questions(
                    generator.generate_comparative_qa_pool(events, stats, fps, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                dominant_qa = generator.sample_questions(
                    generator.generate_dominant_qa_pool(events, stats, fps, duration, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                temporal_qa = generator.sample_questions(
                    generator.generate_temporal_qa_pool(events, stats, fps, duration, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                ordering_qa = generator.sample_questions(
                    generator.generate_ordering_qa_pool(events, stats, fps, duration, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                trajectory_qa = generator.sample_questions(
                    generator.generate_trajectory_affordance_qa_pool(motion_json, stats, rng),
                    samples_per_category,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )
                # Existence: same as v3 — sample_one_per_concept then use full list (no sample_questions)
                existence_pool = generator.generate_existence_qa_pool(events, stats, motion_json)
                negation_pool = generator.generate_negation_qa_pool(events, stats, motion_json)
                existence_qa = generator.sample_one_per_concept(
                    existence_pool,
                    negation_pool,
                    rng,
                    descriptor=subject_descriptor,
                    require_descriptor_in_text=require_descriptor_in_text,
                )

                selected_qa = (
                    _qa_with_task(numerical_qa, "numerical")
                    + _qa_with_task(comparative_qa, "comparative")
                    + _qa_with_task(dominant_qa, "dominant")
                    + _qa_with_task(temporal_qa, "temporal")
                    + _qa_with_task(ordering_qa, "ordering")
                    + _qa_with_task(trajectory_qa, "trajectory_affordance")
                    + _qa_with_task(existence_qa, "existence")
                )

                # Get Video Path (e.g. .../vid_name/person_0_motionx_spatial_caption.json -> video in vid_name/)
                parent_dir = os.path.dirname(json_path)
                vid_name = os.path.basename(parent_dir)
                video_path = motion_json.get("motion_path", "").replace(".npy", ".mp4")
                if not video_path or not os.path.exists(video_path):
                    video_path = json_path.replace("_spatial_caption.json", ".mp4")
                if not os.path.exists(video_path):
                    video_path = os.path.join(parent_dir, f"{vid_name}.mp4")
                if video_base_dir:
                    video_path = os.path.normpath(os.path.join(video_base_dir, f"{vid_name}.mp4"))

                for qa in selected_qa:
                    all_rows.append({"video_path": video_path, "qa": qa})

            except Exception as e:
                print(f"Error processing {json_path}: {e}")
                continue

        # Rebalance correct-letter distribution per task (swap option texts; same as aggregate_by_category)
        if not no_balance:
            _apply_mc_balance_to_rows(all_rows)

        entries_answer_only = []
        entries_cot = [] if cot else None

        rephrase_requests = []
        for row_idx, row in enumerate(all_rows):
            qa = row["qa"]
            if rewriter and random.random() < rewrite_prob:
                rephrase_requests.append((row_idx, "question", qa["question"]))

        rephrased = {}
        if rephrase_requests:
            items = [(text, typ) for _, typ, text in rephrase_requests]
            out_texts = rephrase_batch(items, rewriter, batch_size=rewrite_batch_size)
            for (row_idx, typ, _), text in zip(rephrase_requests, out_texts):
                rephrased[(row_idx, typ)] = text

        for row_idx, row in enumerate(all_rows):
            qa = row["qa"]
            video_path = row["video_path"]
            q_text = rephrased.get((row_idx, "question"), qa["question"])

            options_str = "\n".join([f"{k}. {v}" for k, v in qa["options"].items()])
            user_content = f"{q_text}\n{options_str}\nAnswer with the option letter."

            assistant_answer_only = f"Answer: {qa['answer']}"
            user_msg = {"from": "user", "value": f"<video>\n{user_content}"}

            entry_answer_only = {
                "video": [video_path],
                "task": qa["task"],
                "conversations": [
                    user_msg,
                    {"from": "assistant", "value": assistant_answer_only}
                ]
            }
            entries_answer_only.append(entry_answer_only)
            if cot:
                entry_cot = {
                    "video": [video_path],
                    "task": qa["task"],
                    "conversations": [
                        user_msg,
                        {"from": "assistant", "value": assistant_answer_only}
                    ],
                    "evidence": generator.round_evidence(qa.get("evidence", {})),
                }
                entries_cot.append(entry_cot)

        # Save JSONL variant(s)
        out_answer = os.path.join(output_dir, f"{split_name}_answer_only.jsonl")
        with open(out_answer, 'w') as f:
            for e in entries_answer_only:
                f.write(json.dumps(e) + "\n")
        print(f"Saved {len(entries_answer_only)} samples to {out_answer} (answer only)")
        if cot:
            out_cot = os.path.join(output_dir, f"{split_name}_cot.jsonl")
            with open(out_cot, 'w') as f:
                for e in entries_cot:
                    f.write(json.dumps(e) + "\n")
            print(f"Saved {len(entries_cot)} samples to {out_cot} (CoT)")

    # Run generation
    os.makedirs(output_dir, exist_ok=True)
    process_files(train_files, "train")
    process_files(test_files, "test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SFT data for QwenVL")
    parser.add_argument(
        "--input_dir",
        nargs="+",
        metavar="DIR",
        required=True,
        help="One or more directories (searched recursively) containing *_spatial_caption.json files",
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save train/test_answer_only.jsonl and train/test_cot.jsonl")
    parser.add_argument("--rewrite_prob", type=float, default=0.0, help="Probability to rewrite question/CoT (0.0 to 1.0)")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Path or HuggingFace ID for rewriter model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for rewriter when rewrite_prob > 0 (1 = no batching)")
    parser.add_argument("--video_base_dir", type=str, default=None, help="Base directory for video paths in JSONL (e.g. /data/videos -> /data/videos/<vid_name>.mp4)")
    parser.add_argument("--samples_per_category", type=int, default=10, help="Max questions to sample per QA category (numerical, comparative, dominant, temporal, ordering, trajectory_affordance, existence)")
    parser.add_argument("--cot", action="store_true", help="Also write {split}_cot.jsonl with an extra key 'evidence' per entry (raw QA evidence dict)")
    parser.add_argument("--no-balance", action="store_true", dest="no_balance", help="Skip rebalancing correct-option letters; keep the as-generated option order")
    parser.add_argument(
        "--descriptor",
        action="store_true",
        help="Use motion JSON 'descriptor' like generate_spatial_question_v5 --description: "
        "normalize (one clothing clause, fix wore/wearing), personalize questions, and prefix "
        "'For <subject>, …' when the subject phrase would otherwise be missing.",
    )
    args = parser.parse_args()

    create_dataset(
        args.input_dir,
        args.output_dir,
        samples_per_category=args.samples_per_category,
        rewrite_prob=args.rewrite_prob,
        model_path=args.model_path,
        rewrite_batch_size=args.batch_size,
        video_base_dir=args.video_base_dir,
        cot=args.cot,
        descriptor=args.descriptor,
        no_balance=args.no_balance,
    )
