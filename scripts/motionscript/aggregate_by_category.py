#!/usr/bin/env python3
"""
Aggregate all QA JSON files by question category.
Writes one JSON per category with video_path, question, answer (e.g. "A. significantly").

When balancing (default), correct-option letters are round-robined per category using the
expected option layout: existence A/B (yes/no), comparative and dominant A/B/C (3-way MC),
all other categories A–D (4-way MC). Rows whose option keys do not match that layout are
left unchanged (e.g. legacy 4-option comparative until the generator is updated).

CLI: one or more input directories, then output_dir as the final path (entries from all
inputs are merged in order).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

MC_LETTERS = ("A", "B", "C", "D")

# Expected MC keys per category (for answer-key balancing). Others default to 4-way A–D.
CATEGORY_MC_LETTERS = {
    "existence": ("A", "B"),
    "comparative": ("A", "B", "C"),
    "dominant": ("A", "B", "C"),
}

# Categories excluded from answer-letter balancing (kept as generated).
BALANCE_EXCLUDED = {"existence"}

# Categories produced by generate_spatial_qa_questions.py (qa_<category>.json)
CATEGORIES = [
    "numerical",
    "comparative",
    "dominant",
    "temporal",
    "existence",
    "ordering",
    "trajectory_affordance",
]


def _expected_letters(category: str) -> tuple[str, ...]:
    return CATEGORY_MC_LETTERS.get(category, MC_LETTERS)


def _matches_category_options(category: str, options: dict) -> bool:
    return bool(options) and set(options.keys()) == set(_expected_letters(category))


def _relabel_correct_letter(
    options: dict, old_letter: str, desired_letter: str
) -> tuple[dict, str]:
    """Swap option texts so the correct answer moves from old_letter to desired_letter."""
    if old_letter == desired_letter:
        return dict(options), old_letter
    out = dict(options)
    out[old_letter], out[desired_letter] = out[desired_letter], out[old_letter]
    return out, desired_letter


def _format_line_object(
    qid: str,
    video_path,
    question_stem: str,
    options: dict,
    answer_letter,
) -> dict:
    question = question_stem
    if options:
        opts_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        question = f"{question_stem}\nOptions:\n{opts_text}"
    if answer_letter and options and answer_letter in options:
        answer = f"{answer_letter}. {options[answer_letter]}"
    else:
        answer = answer_letter
    return {
        "id": qid,
        "video_path": video_path,
        "question": question,
        "answer": answer,
    }


def _balance_mc_distribution(category: str, entries: list[dict]) -> list[dict]:
    """Sort by (video_path, stem), then round-robin correct letters for this category's MC size."""
    letters = _expected_letters(category)
    n = len(letters)
    sorted_entries = sorted(
        entries,
        key=lambda e: (e.get("video_path") or "", e.get("question_stem") or ""),
    )
    out: list[dict] = []
    for i, e in enumerate(sorted_entries):
        stem = e["question_stem"]
        opts = e["options"]
        ans = e["answer_letter"]
        desired = letters[i % n]
        if _matches_category_options(category, opts) and ans in opts:
            opts, ans = _relabel_correct_letter(opts, ans, desired)
        out.append(
            {
                "video_path": e["video_path"],
                "question_stem": stem,
                "options": opts,
                "answer_letter": ans,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate QA JSON by category")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Do not rebalance MC correct-letter counts within each category",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more input roots with qa_*.json files, then output_dir as the last path",
    )
    args = parser.parse_args()
    resolved = [p.resolve() for p in args.paths]
    if len(resolved) < 2:
        parser.error("need at least one input directory and one output directory")
    output_dir = resolved[-1]
    input_dirs = resolved[:-1]
    output_dir.mkdir(parents=True, exist_ok=True)

    by_category = defaultdict(list)
    allowed = set(CATEGORIES)

    for input_dir in input_dirs:
        for path in input_dir.rglob("*.json"):
            if path.name.startswith("qa_") and output_dir not in path.parents:
                # e.g. qa_existence.json -> existence
                category = path.stem.replace("qa_", "")
                if category not in allowed:
                    continue
                try:
                    data = json.loads(path.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Skip {path}: {e}")
                    continue
                video_path = data.get("video_path")
                qa_pairs = data.get("qa_pairs", [])
                for pair in qa_pairs:
                    question_stem = pair.get("question") or ""
                    options = pair.get("options") or {}
                    answer_option = pair.get("answer")  # e.g. "A", "B"
                    by_category[category].append({
                        "video_path": video_path,
                        "question_stem": question_stem,
                        "options": options,
                        "answer_letter": answer_option,
                    })

    for category in CATEGORIES:
        entries = by_category.get(category, [])
        if not args.no_balance and category not in BALANCE_EXCLUDED:
            entries = _balance_mc_distribution(category, entries)
        cat_dir = output_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_path = cat_dir / f"{category}.json"
        lines = []
        for i, e in enumerate(entries, start=1):
            line_obj = _format_line_object(
                f"q{i}",
                e["video_path"],
                e["question_stem"],
                e["options"],
                e["answer_letter"],
            )
            lines.append(json.dumps(line_obj))
        out_path.write_text("\n".join(lines))
        print(f"Wrote {out_path} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
