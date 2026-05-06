"""
Convert SFT CoT JSONL to <observe> <reasoning> <answer> format.

Reads JSONL where each line has:
  - conversations: [user_msg, assistant_msg with "Answer: X"]
  - evidence: { events_used, computation, raw_values, optionally segments_used, ... }

Writes JSONL where the assistant value is replaced by:
  <observe>
  [ { event dicts } ]
  </observe>
  <reasoning>key=value,...</reasoning>
  <answer>X</answer>

- observe: raw JSON array of events_used (or segments_used); empty [] if none.
  With --short: [spatial_code(intensity,velocity,spatial_category,temporal_category), ...].
- reasoning: type-specific format:
  - numerical → value + choice (option label from prompt when present); multi-metric uses raw keys (e.g. left_count=)
  - total/value → value + choice (option label from prompt when present)
  - dominant → compared totals + dominant label
  - ordering → anchor + next event
  - existence → present/absent
  - comparative → compared values + winner
  - trajectory_affordance → metric-by-quarter + best quarter (or returnability/overall)
- answer: option letter only (e.g. B).
"""

import argparse
import json
import re
import sys


DEFAULT_FPS = 30.0


def _round_val(v):
    """Round floats to 2 decimal places for reasoning output."""
    if isinstance(v, float):
        r = round(v, 2)
        # Canonicalize -0.0 → 0.0 (e.g. most-leftward / per-quarter metrics).
        return 0.0 if r == 0 else r
    return v


def _short_observe_item(item: dict, fps: float = 30.0) -> str:
    """
    One item in short observe format: spatial_code(intensity,velocity,spatial_category,temporal_category).
    item may be an event (code, intensity, velocity, spatial_category, temporal_category) or
    a segment (axis, start_frame, end_frame, intensity, spatial_category).
    """
    code = item.get("code") or item.get("axis") or "?"
    intensity = item.get("intensity")
    if intensity is not None and isinstance(intensity, (int, float)):
        intensity = round(intensity, 2)
    else:
        intensity = "-"
    velocity = item.get("velocity")
    if velocity is not None and isinstance(velocity, (int, float)):
        velocity = round(velocity, 2)
    else:
        # segments: optional derive from intensity/duration
        sf, ef = item.get("start_frame"), item.get("end_frame")
        if sf is not None and ef is not None and fps > 0 and (ef - sf) > 0 and isinstance(intensity, (int, float)):
            velocity = round(abs(intensity) / ((ef - sf) / fps), 2)
        else:
            velocity = "-"
    spatial_category = item.get("spatial_category") or "-"
    temporal_category = item.get("temporal_category") or "-"
    return f"{code}({intensity},{velocity},{spatial_category},{temporal_category})"


def build_observe(evidence: dict, short: bool = False, fps: float = 30.0) -> str:
    """
    Build <observe> content: raw JSON array of motion events, or short format when short=True.
    Uses events_used; if empty, uses segments_used; else [].
    When short=True: [spatial_code(intensity,velocity,spatial_category,temporal_category), ...]
    """
    events = evidence.get("events_used") or []
    segments = evidence.get("segments_used") or []
    drop_keys = {"t0", "t1", "duration"}

    if short:
        if events:
            sorted_items = sorted(events, key=lambda x: x.get("t0", 0))
        elif segments:
            sorted_items = sorted(segments, key=lambda x: x.get("start_frame", 0))
        else:
            return "[]"
        parts = [_short_observe_item(it, fps) for it in sorted_items]
        return "[" + ",".join(parts) + "]"

    def strip_temporal(d):
        return {k: v for k, v in d.items() if k not in drop_keys}

    if events:
        sorted_events = sorted(events, key=lambda x: x.get("t0", 0))
        return json.dumps([strip_temporal(e) for e in sorted_events], indent=2)
    if segments:
        sorted_segments = sorted(segments, key=lambda x: x.get("start_frame", 0))
        return json.dumps([strip_temporal(s) for s in sorted_segments], indent=2)
    return "[]"


STRENGTH_CATEGORY = {0: "none", 1: "slight", 2: "moderate", 3: "significant"}

QUARTER_LABELS = {1: "first quarter", 2: "second quarter", 3: "third quarter", 4: "fourth quarter"}


def _quarter_label(q: int) -> str:
    """Return 'first quarter', 'second quarter', etc. for q in 1..4."""
    return QUARTER_LABELS.get(q, f"Q{q}")

# Keys to omit from raw_values when building generic key=value (temporal bucket / numerical metrics handled explicitly)
REASONING_SKIP_KEYS = frozenset({
    "quarter",
    "time_range",
    "first_event_time",
    "option_range",
    "range",
    "matched_range",
})

# Tie → task-meaningful choice label for comparative/dominant (keys are sorted pairs)
TIE_CHOICE_BY_PAIR = {
    ("clockwise", "counterclockwise"): "no_turns",
    ("backward", "forward"): "no_leaning",
    ("left", "right"): "roughly_equal",
    ("away", "towards"): "roughly_equal",
}


def _reasoning_chain(parts: list, answer_letter: str = "") -> str:
    """Join parts with '; ' and append '; option=<letter>' when given. Full decision chain."""
    s = "; ".join(p for p in parts if p)
    if answer_letter:
        s = f"{s}; option={answer_letter}" if s else f"option={answer_letter}"
    return s


def _tie_choice(label1: str, label2: str) -> str:
    """Map tie to task-meaningful choice: roughly_equal, no_turns, no_leaning, both_equally."""
    key = tuple(sorted([label1, label2]))
    return TIE_CHOICE_BY_PAIR.get(key, "both_equally")


def _reasoning_trajectory_affordance(raw: dict) -> list:
    """trajectory_affordance → statistic; choice; (option appended by _reasoning_chain)."""
    # Per-quarter metric + answer_quarter
    per_q = (
        raw.get("per_quarter_net_dx_units")
        or raw.get("per_quarter_net_dz_units")
        or raw.get("per_quarter_path_units")
        or raw.get("per_quarter_rightward_dx_units")
        or raw.get("per_quarter_leftward_dx_units")
        or raw.get("per_quarter_abs_net_dx_units")
        or raw.get("min_dist_by_quarter_units")
    )
    if per_q is None:
        for key, value in raw.items():
            if (
                isinstance(value, dict)
                and (key.startswith("per_quarter_") or key == "min_dist_by_quarter_units")
            ):
                per_q = value
                break
    best = raw.get("answer_quarter")
    if per_q is not None and isinstance(per_q, dict) and best is not None:
        parts = [f"{_quarter_label(q)}={_round_val(per_q.get(str(q), 0))}" for q in range(1, 5) if str(q) in per_q]
        parts.append(f"choice={_quarter_label(best)}")
        return parts
    # Returnability
    ret = raw.get("returnability")
    if ret is not None:
        parts = [f"returnability={ret}"]
        for k in ("max_dist_units", "return_threshold_units", "away_threshold_units", "min_dist_after_away_units", "returned"):
            if k in raw and raw[k] is not None:
                parts.append(f"{k}={_round_val(raw[k])}" if isinstance(raw[k], (int, float)) else f"{k}={raw[k]}")
        parts.append(f"choice={ret}")
        return parts
    # Overall net_dx 4-way
    if "answer" in raw and raw.get("net_dx_units") is not None:
        ans = raw["answer"]
        return [f"net_dx={_round_val(raw['net_dx_units'])}", f"path_x={_round_val(raw.get('path_x_units', 0))}", f"choice={ans}"]
    return []


def _reasoning_ordering(evidence: dict, fps: float = DEFAULT_FPS) -> list:
    """ordering or temporal presence → anchor (and next_event) as spatial_code(intensity,velocity,spatial_category,temporal_category); choice = bucket or option label."""
    event_a = evidence.get("event_A") or evidence.get("first_event")
    event_b = evidence.get("event_B")
    if event_a is not None:
        parts = []
        # Temporal presence: "when did X first occur?" (first_event only, no event_B)
        if event_b is None and evidence.get("first_event") is not None:
            parts.append("presence=first")
        anchor_short = _short_observe_item(event_a, fps)
        parts.append(f"anchor={anchor_short}")
        if event_b is not None:
            next_short = _short_observe_item(event_b, fps)
            parts.append(f"next_event={next_short}")
        bucket = evidence.get("bucket", "")
        if bucket:
            parts.append(f"choice={bucket}")
        return parts
    return []


def _apply_option_label_to_parts(parts: list, option_label: str = "") -> list:
    """If option_label is given, set choice to that label so <reasoning> matches <answer>."""
    if not option_label or not parts:
        return parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].startswith("choice="):
            parts = list(parts)
            parts[i] = f"choice={option_label}"
            return parts
    parts = list(parts)
    parts.append(f"choice={option_label}")
    return parts


def _normalize_reasoning_text(reasoning_text: str, answer_letter: str = "", option_label: str = "") -> str:
    """
    Ensure the emitted reasoning is consistent with the final answer.
    - Always force option=<answer_letter> when an answer is available.
    - When the prompt exposes option labels, always force choice=<option_label>.
    """
    parts = [part.strip() for part in reasoning_text.split(";") if part.strip()]

    if option_label:
        parts = _apply_option_label_to_parts(parts, option_label)

    if answer_letter:
        option_part = f"option={answer_letter}"
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].startswith("option="):
                parts[i] = option_part
                break
        else:
            parts.append(option_part)

    return "; ".join(parts)


def build_reasoning(evidence: dict, answer_letter: str = "", option_label: str = "", fps: float = DEFAULT_FPS) -> str:
    """
    Build <reasoning> with full decision chain: statistic; semantic choice; option letter.
    - Use '; ' between parts. Always append '; option=<letter>' when available.
    - When option_label is provided (from user message), use it for choice so <reasoning> matches <answer>.
    - Discrete options: choice comes from the prompt label (option_label), not numeric ranges.
    - Replace winner=tie/dominant=tie with choice=roughly_equal, no_turns, no_leaning, both_equally.
    """
    raw = evidence.get("raw_values") or {}
    computation = evidence.get("computation", "")

    # ---- ordering: anchor + next event ----
    if evidence.get("event_A") is not None or evidence.get("first_event") is not None:
        parts = _reasoning_ordering(evidence)
        if parts:
            parts = _apply_option_label_to_parts(parts, option_label)
            return _reasoning_chain(parts, answer_letter)

    # ---- trajectory_affordance ----
    if "answer_quarter" in raw or "returnability" in raw or ("answer" in raw and "net_dx_units" in raw):
        parts = _reasoning_trajectory_affordance(raw)
        if parts:
            parts = _apply_option_label_to_parts(parts, option_label)
            return _reasoning_chain(parts, answer_letter)

    raw = {k: v for k, v in raw.items() if k not in REASONING_SKIP_KEYS}
    if not raw:
        base = computation.strip() if computation else ""
        parts = [base] if base else []
        if option_label:
            parts = _apply_option_label_to_parts(parts, option_label)
        return _reasoning_chain(parts, answer_letter)

    # ---- total/value (total movement) → value=... first so we never treat as numerical bucket ----
    if "total_units" in raw:
        val = _round_val(raw["total_units"])
        parts = [f"value={val}"]
        choice = option_label or val
        parts.append(f"choice={choice}")
        return _reasoning_chain(parts, answer_letter)

    # ---- numerical → value= for single scalar; else raw metric keys (e.g. left_count=); choice = option label ----
    numerical_keys = [k for k in raw if k.endswith("_count") or k == "count"]
    if numerical_keys:
        if len(numerical_keys) == 1:
            num_val = raw.get("count") or raw.get(numerical_keys[0])
            if num_val is not None:
                disp = _round_val(num_val) if isinstance(num_val, float) else num_val
                parts = [f"value={disp}"]
                choice = option_label or disp
                parts.append(f"choice={choice}")
                return _reasoning_chain(parts, answer_letter)
        else:
            # Multiple metrics (e.g. left_count/right_count, pitch_forward_count/pitch_backward_count)
            parts = []
            for k in sorted(numerical_keys):
                v = raw.get(k)
                if v is None:
                    continue
                disp = _round_val(v) if isinstance(v, float) else v
                parts.append(f"{k}={disp}")
            if parts:
                if evidence.get("bucket"):
                    parts = [f"bucket={evidence['bucket']}"] + parts
                choice = option_label or max(numerical_keys, key=lambda k: raw.get(k) or 0)
                parts.append(f"choice={choice}")
                return _reasoning_chain(parts, answer_letter)

    # ---- existence → present/absent; choice = option label ----
    if "S_left" in raw or "S_right" in raw or "S_away" in raw or "S_towards" in raw or "S_clockwise" in raw or "S_counterclockwise" in raw or "S_forward" in raw or "S_backward" in raw or "jump_count" in raw:
        present = raw.get("jump_count", 0) > 0 or (answer_letter == "A")
        choice = option_label or ("present" if present else "absent")
        return _reasoning_chain([choice], answer_letter)

    # ---- comparative → left=X; right=Y; choice = option label ----
    comp_pairs = [
        ("left_intensity", "right_intensity", "left", "right"),
        ("away_intensity", "towards_intensity", "forward", "backward"),
        ("clockwise", "counterclockwise", "clockwise", "counterclockwise"),
        ("forward", "backward", "forward", "backward"),
        ("left", "right", "left", "right"),
    ]
    for k1, k2, label1, label2 in comp_pairs:
        v1 = raw.get(k1) if k1 in raw else raw.get(label1)
        v2 = raw.get(k2) if k2 in raw else raw.get(label2)
        if v1 is not None and v2 is not None:
            try:
                v1, v2 = float(v1), float(v2)
                choice = option_label or (label1 if v1 > v2 else label2 if v2 > v1 else _tie_choice(label1, label2))
                parts = [f"{label1}={_round_val(v1)}", f"{label2}={_round_val(v2)}", f"choice={choice}"]
                return _reasoning_chain(parts, answer_letter)
            except (TypeError, ValueError):
                pass

    # ---- dominant: 4-way (left, right, forward, backward) — show all 4 so choice matches answer ----
    four_way_keys = ("left", "right", "forward", "backward")
    if all(k in raw for k in four_way_keys):
        try:
            parts = [f"{k}={_round_val(raw[k])}" for k in four_way_keys]
            choice = option_label or max(four_way_keys, key=lambda k: float(raw[k]))
            parts.append(f"choice={choice}")
            return _reasoning_chain(parts, answer_letter)
        except (TypeError, ValueError):
            pass

    # ---- dominant: 2-way pairs (clockwise/counterclockwise, away/towards, etc.) ----
    pair_keys = [
        ("left", "right"),
        ("forward", "backward"),
        ("away", "towards"),
        ("clockwise", "counterclockwise"),
    ]
    for a, b in pair_keys:
        if a in raw and b in raw:
            va, vb = raw[a], raw[b]
            try:
                va, vb = float(va), float(vb)
                choice = option_label or (a if va > vb else b if vb > va else _tie_choice(a, b))
                parts = [f"{a}={_round_val(va)}", f"{b}={_round_val(vb)}", f"choice={choice}"]
                return _reasoning_chain(parts, answer_letter)
            except (TypeError, ValueError):
                pass

    # ---- strength → choice = option label ----
    if "strongest_strength" in raw:
        v = raw["strongest_strength"]
        cat = STRENGTH_CATEGORY.get(int(v) if v is not None else 0, str(v))
        choice = option_label or cat
        return _reasoning_chain([f"strength={cat}", f"choice={choice}"], answer_letter)

    # Fallback: statistic; option
    out = []
    for k, v in sorted(raw.items()):
        if isinstance(v, dict):
            continue
        out.append(f"{k}={_round_val(v)}")
    if out and option_label:
        out = _apply_option_label_to_parts(out, option_label)
    if out:
        return _reasoning_chain(out, answer_letter)
    return _reasoning_chain([computation.strip()] if computation.strip() else [], answer_letter)


def extract_answer_letter(assistant_value: str) -> str:
    """Extract option letter from 'Answer: X' or similar."""
    if not assistant_value:
        return ""
    m = re.search(r"Answer:\s*([A-E])", assistant_value, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    stripped = assistant_value.strip().upper()
    if len(stripped) == 1 and stripped in "ABCDE":
        return stripped
    return ""


def parse_option_labels_from_user(user_value: str) -> dict:
    """
    Parse option letter -> label from user message. Lines like 'A. First quarter' or 'B. Yes'.
    Returns dict mapping letter -> option text after the delimiter.
    """
    out = {}
    if not user_value:
        return out
    for line in user_value.split("\n"):
        line = line.strip()
        m = re.match(r"^([A-E])\s*[\.\):]\s*(.+)$", line, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            label = m.group(2).strip()
            out[letter] = label
    return out


def convert_record(
    record: dict,
    fps: float = DEFAULT_FPS,
    short: bool = False,
    reasoning_answer_only: bool = False,
) -> dict:
    """
    Convert one JSONL record: replace assistant value with <observe> <reason> <answer>.
    Returns a new dict; does not mutate the input.
    When short=True, observe uses format [spatial_code(intensity,velocity,spatial_category,temporal_category), ...].
    When reasoning_answer_only=True, assistant value is only <reasoning> and <answer> (no <observe>).
    """
    conversations = list(record.get("conversations") or [])
    evidence = record.get("evidence") or {}
    if not conversations:
        return record

    # Find assistant turn
    assistant_idx = None
    for i, c in enumerate(conversations):
        if c.get("from") == "assistant":
            assistant_idx = i
            break
    if assistant_idx is None:
        return record

    old_value = conversations[assistant_idx].get("value", "")
    letter = extract_answer_letter(old_value)
    if not letter:
        letter = "?"

    # Parse option labels from user message so reasoning choice matches the answer option
    user_msg = next((c.get("value", "") for c in conversations if c.get("from") == "user"), "")
    options = parse_option_labels_from_user(user_msg)
    option_label = options.get(letter, "")

    reasoning_text = build_reasoning(
        evidence,
        answer_letter=letter,
        option_label=option_label,
        fps=fps,
    )
    reasoning_text = _normalize_reasoning_text(
        reasoning_text,
        answer_letter=letter,
        option_label=option_label,
    )
    if reasoning_answer_only:
        new_value = f"<reasoning>{reasoning_text}</reasoning>\n<answer>{letter}</answer>"
    else:
        observe_text = build_observe(evidence, short=short, fps=fps)
        new_value = f"<observe>\n{observe_text}\n</observe>\n<reasoning>{reasoning_text}</reasoning>\n<answer>{letter}</answer>"

    out = dict(record)
    out["conversations"] = [dict(c) for c in conversations]
    out["conversations"][assistant_idx] = {"from": "assistant", "value": new_value}
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convert CoT JSONL to <observe> <reason> <answer> format."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        dest="input_jsonl",
        help="Input JSONL (e.g. train_cot.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSONL path (default: <input_base>_observe_reason_answer.jsonl)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second for time formatting (default: 30)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file with converted data (ignores --output)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Use short observe format: [spatial_code(intensity,velocity,spatial_category,temporal_category), ...]",
    )
    parser.add_argument(
        "--reasoning-answer-only",
        action="store_true",
        dest="reasoning_answer_only",
        help="Output only <reasoning> and <answer> in assistant value (no <observe> block).",
    )
    args = parser.parse_args()

    if args.in_place:
        out_path = args.input_jsonl
    elif args.output:
        out_path = args.output
    else:
        base = args.input_jsonl.rsplit(".jsonl", 1)[0]
        out_path = base + "_observe_reason_answer.jsonl"

    records = []
    with open(args.input_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skip invalid JSON line: {e}", file=sys.stderr)

    converted = [
        convert_record(
            r,
            fps=args.fps,
            short=args.short,
            reasoning_answer_only=args.reasoning_answer_only,
        )
        for r in records
    ]

    write_to = out_path
    if args.in_place:
        write_to = out_path + ".tmp"
    with open(write_to, "w") as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.in_place:
        import os
        os.replace(write_to, out_path)

    print(f"Converted {len(converted)} records -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
