#!/usr/bin/env python3
"""
Generate template-based QA pairs from spatial motion JSON files.
Copy of generate_spatial_qa.py for defining questions in a clearer, more maintainable way.
Outputs 7 JSON files, one per question axis: numerical, comparative, dominant, temporal, ordering,  trajectory_affordance, existence.
Existence axis includes both existence and negation (counterfactual) questions in qa_existence.json.

Input JSON is first-frame normalized: all spatial directions (left/right, towards/away from starting position,
rotation labels) and derived questions are defined w.r.t. the first frame.
"""

import json
import argparse
import random
import os
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


EVIDENCE_FLOAT_DIGITS = 2  # precision for floats in evidence (e.g. 8.84 instead of 8.8459...)


def round_evidence(evidence: Dict[str, Any], ndigits: int = EVIDENCE_FLOAT_DIGITS) -> Dict[str, Any]:
    """Recursively round float values in an evidence dict for readable output. Leaves other types unchanged."""
    if evidence is None:
        return None
    out: Dict[str, Any] = {}
    for k, v in evidence.items():
        if isinstance(v, dict):
            out[k] = round_evidence(v, ndigits)
        elif isinstance(v, float):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out


def load_motion_json(path: str) -> Dict[str, Any]:
    """Load motion JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def _first_clothing_clause(fragment: str) -> str:
    """Use one outfit phrase when the source chains several with commas (e.g. 'shirt, shorts')."""
    s = fragment.strip()
    if not s:
        return s
    s = re.sub(r",+", ",", s)
    # Strip a leading non-clothing location phrase such as "on a chair, wearing a white shirt …"
    m0 = re.match(
        r"^(?:on|at)\s+\S.+?,\s+((?:wearing|dressed\s+in)\s+.+)$",
        s,
        flags=re.IGNORECASE,
    )
    if m0:
        s = m0.group(1).strip()
    # "…, a/an/the <next garment> …"
    m = re.match(r"^(.+?),\s*(?:a|an|the)\s+\S", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # "…, black shorts" (no article on the second item)
    m2 = re.match(
        r"^(.+?),\s+((?:black|white|red|blue|green|yellow|grey|gray|navy|brown|beige|"
        r"striped|plain|denim|leather)\s+\S.+)$",
        s,
        flags=re.IGNORECASE,
    )
    if m2:
        left = m2.group(1).strip()
        if len(left.split()) >= 3:
            return left
    return s


def normalize_descriptor_string(raw: Optional[str]) -> Optional[str]:
    """Normalize a descriptor string (CLI or JSON field): first semicolon segment, one clothing clause."""
    if raw is None or not isinstance(raw, str):
        return None
    descriptor = raw.strip()
    if not descriptor:
        return None
    descriptor_candidates = [part.strip(" ,;") for part in descriptor.split(";") if part.strip(" ,;")]
    descriptor = descriptor_candidates[0] if descriptor_candidates else descriptor.strip()
    descriptor = re.sub(r"\s*,\s*,+\s*", ", ", descriptor)
    descriptor = _first_clothing_clause(descriptor)
    descriptor = re.sub(r"\s+", " ", descriptor).strip(" ,;")
    return descriptor or None


def resolve_subject_descriptor(motion_json: Dict[str, Any]) -> Optional[str]:
    """Return a cleaned descriptor from motion JSON, or None when missing or unusable."""
    descriptor = motion_json.get("descriptor")
    if not isinstance(descriptor, str):
        return None
    return normalize_descriptor_string(descriptor)


def _looks_like_person_noun_phrase(descriptor: str) -> bool:
    """Heuristic: descriptor already names a person rather than only clothing."""
    return bool(re.match(
        r"^(?:(?:young|old|middle-aged|elderly|adult|tall|short|slender|heavyset|male|female)\s+){0,3}"
        r"(person|subject|man|woman|boy|girl|individual|figure|adult|child)\b",
        descriptor,
        flags=re.IGNORECASE,
    ))


def _subject_reference_from_descriptor(descriptor: Optional[str]) -> Optional[Tuple[str, str]]:
    """Build definite subject and possessive forms from a descriptor."""
    if not descriptor:
        return None

    bare_descriptor = re.sub(r"^(a|an|the)\s+", "", descriptor, count=1, flags=re.IGNORECASE).strip()
    if not bare_descriptor:
        subject = "the person"
    elif _looks_like_person_noun_phrase(bare_descriptor):
        subject = f"the {bare_descriptor}"
    else:
        # Keep leading article (a/an/the); bare_descriptor drops it and yields "wearing white shirt".
        clothing_phrase = descriptor.strip()
        clothing_phrase = re.sub(r"^(?:was\s+)?wore\s+", "", clothing_phrase, count=1, flags=re.IGNORECASE)
        clothing_phrase = re.sub(r"^(?:was\s+)?wearing\s+", "", clothing_phrase, count=1, flags=re.IGNORECASE)
        clothing_phrase = re.sub(r"^dressed in\s+", "", clothing_phrase, count=1, flags=re.IGNORECASE).strip()
        if not clothing_phrase:
            subject = "the person"
        elif re.match(r"^in\b", clothing_phrase, flags=re.IGNORECASE):
            subject = f"the person {clothing_phrase}"
        else:
            subject = f"the person wearing {clothing_phrase}"
    # Possessive for extended "the person …" (clothing): use Saxon genitive on the full NP so templates
    # like "the person's first … than their first …" keep one explicit subject + the template's "their",
    # and we avoid subject.endswith("s") misfiring on trailing words like "jeans".
    if subject == "the person":
        possessive = "the person's"
    elif subject.startswith("the person "):
        possessive = f"{subject}'s"
    else:
        possessive = f"{subject}'" if subject.endswith("s") else f"{subject}'s"
    return subject, possessive


# Placeholders must not appear in real question text; private-use chars avoid re-matching after expansion.
_PERSON_SUBJ_U = "\ufdd0SUBJ_U\ufdd1"
_PERSON_SUBJ_L = "\ufdd0SUBJ_L\ufdd1"
_PERSON_POSS_U = "\ufdd0POSS_U\ufdd1"
_PERSON_POSS_L = "\ufdd0POSS_L\ufdd1"


def personalize_question_text(question: str, descriptor: Optional[str]) -> str:
    """Swap generic 'person' references for the descriptor when available."""
    subject_forms = _subject_reference_from_descriptor(descriptor)
    if not subject_forms:
        return question

    subject, possessive = subject_forms
    subj_u = subject[:1].upper() + subject[1:]
    poss_u = possessive[:1].upper() + possessive[1:]

    # Longest phrases first, then replace markers — expanded subject still contains "the person" and
    # would otherwise be substituted twice (e.g. "the person's" -> "the person wearing X'" then
    # "the person" -> "the person wearing X" again inside that).
    out = question
    out = out.replace("The person's", _PERSON_POSS_U)
    out = out.replace("the person's", _PERSON_POSS_L)
    out = out.replace("The person", _PERSON_SUBJ_U)
    out = out.replace("the person", _PERSON_SUBJ_L)
    out = out.replace(_PERSON_POSS_U, poss_u)
    out = out.replace(_PERSON_POSS_L, possessive)
    out = out.replace(_PERSON_SUBJ_U, subj_u)
    out = out.replace(_PERSON_SUBJ_L, subject)
    return out


def ensure_descriptor_in_question(question: str, descriptor: str) -> str:
    """If the resolved subject phrase does not appear in the question, prefix with 'For <subject>, '."""
    subject_forms = _subject_reference_from_descriptor(descriptor)
    if not subject_forms:
        return question
    subject, _ = subject_forms
    if subject.lower() in question.lower():
        return question
    qrest = question
    if len(qrest) > 1 and qrest[0].isupper() and qrest[1:2].islower():
        qrest = qrest[0].lower() + qrest[1:]
    return f"For {subject}, {qrest}"


def personalize_qa_questions(
    qa_list: List[Dict[str, Any]],
    descriptor: Optional[str],
    *,
    require_descriptor_in_text: bool = False,
) -> List[Dict[str, Any]]:
    """Return shallow-copied QA items with personalized question text."""
    if not descriptor:
        return list(qa_list)

    personalized = []
    for qa in qa_list:
        qa_copy = dict(qa)
        if "question" in qa_copy:
            qtext = personalize_question_text(qa_copy["question"], descriptor)
            if require_descriptor_in_text:
                qtext = ensure_descriptor_in_question(qtext, descriptor)
            qa_copy["question"] = qtext
        personalized.append(qa_copy)
    return personalized


def detect_jumps(motion_json: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    """Detect jumps from displacement_y: consecutive segments (by motion_index) where the first is
    moderate/long/very_long up and the next is moderate/long/very_long down.
    Returns (jump_count, events_used) where events_used is a list of event-like dicts (one per jump:
    the up and down segment quants as minimal event dicts for evidence)."""
    segments = motion_json.get("quantitative_values", {}).get("displacement_y", [])
    if not segments:
        return 0, []
    sorted_segments = sorted(segments, key=lambda s: s.get("motion_index", 0))

    def is_moderate_or_more_y(cat: str) -> bool:
        return "moderate" in (cat or "") or "long" in (cat or "")

    def is_up(cat: str) -> bool:
        return "up" in (cat or "").lower() and is_moderate_or_more_y(cat)

    def is_down(cat: str) -> bool:
        return "down" in (cat or "").lower() and is_moderate_or_more_y(cat)

    events_used = []
    for i in range(len(sorted_segments) - 1):
        seg_up = sorted_segments[i]
        seg_down = sorted_segments[i + 1]
        q_up = seg_up.get("quant", {})
        q_down = seg_down.get("quant", {})
        cat_up = q_up.get("spatial_category", "")
        cat_down = q_down.get("spatial_category", "")
        if is_up(cat_up) and is_down(cat_down):
            events_used.append({
                "code": "displacement_y",
                "t0": q_up.get("start_frame"),
                "t1": q_up.get("end_frame"),
                "spatial_category": cat_up,
                "direction": "up",
            })
            events_used.append({
                "code": "displacement_y",
                "t0": q_down.get("start_frame"),
                "t1": q_down.get("end_frame"),
                "spatial_category": cat_down,
                "direction": "down",
            })
    jump_count = len(events_used) // 2
    return jump_count, events_used


# Minimum duration in frames for an event to be included in QA (exclude very short segments)
MIN_EVENT_FRAMES = 5

# Existence questions: meaningful-existence thresholds (Option A: cumulative magnitude)
# Answer Yes for a direction iff sum(|intensity|) >= epsilon for that axis.
EXISTENCE_EPSILON_DISPLACEMENT = 1.0   # displacement_x, displacement_z (in same units as intensity)
EXISTENCE_EPSILON_ROTATION = 5.0       # rotation_yaw, rotation_pitch, rotation_roll
# Dominance filter: for paired directions on same axis, treat smaller as No if min < r * max.
EXISTENCE_DOMINANCE_RATIO = 0.2


def _cumulative_intensity(events: List[Dict[str, Any]]) -> float:
    """Sum of |intensity| for events. Used for existence magnitude threshold."""
    return sum(abs(e.get("intensity", 0)) for e in events)


def _max_intensity_event(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For existence evidence: return the single event with highest |intensity|, or [] if none."""
    if not events:
        return []
    best = max(events, key=lambda e: abs(e.get("intensity", 0)))
    return [best]


def _existence_computation(events_used: List[Dict[str, Any]], full_computation: str) -> str:
    """For existence-type questions: when evidence has only 1 event, show '1 event' instead of full count."""
    if len(events_used) == 1:
        return "1 event"
    return full_computation


def _count_metric_raw_values(event_count: int) -> Dict[str, Any]:
    """Standard raw_values schema for count-based existence/negation families."""
    return {
        "metric_type": "count",
        "event_count": event_count,
    }


def _directional_strength_raw_values(
    target_strength: float,
    opposite_strength: float,
    epsilon: float,
    dominance_ratio: float,
    target_event_count: int,
    opposite_event_count: int,
) -> Dict[str, Any]:
    """Standard raw_values schema for directional existence questions."""
    return {
        "metric_type": "cumulative_intensity",
        "target_strength": target_strength,
        "opposite_strength": opposite_strength,
        "epsilon": epsilon,
        "dominance_ratio": dominance_ratio,
        "target_event_count": target_event_count,
        "opposite_event_count": opposite_event_count,
    }


# Concept IDs for existence/negation: one question per (video, concept), form chosen by coin flip.
EXISTENCE_CONCEPT_IDS = [
    "walking", "jump", "significant_turning", "significant_leaning",
    "move_left", "move_right", "move_away_start", "move_towards_start",
    "turn_clockwise", "turn_counterclockwise", "lean_forward", "lean_backward", "lean_left", "lean_right",
]


def _existence_epsilon_for_code(code: str) -> float:
    """Epsilon threshold for meaningful existence on this axis."""
    if code in ["displacement_x", "displacement_z"]:
        return EXISTENCE_EPSILON_DISPLACEMENT
    if code in ["rotation_yaw", "rotation_pitch", "rotation_roll"]:
        return EXISTENCE_EPSILON_ROTATION
    return 0.0


def _existence_yes_for_direction(
    S_dir: float, S_other: float, epsilon: float, r: float = EXISTENCE_DOMINANCE_RATIO
) -> bool:
    """True iff this direction has meaningful existence: above epsilon and above dominance ratio.
    So smaller component is No when min(S_dir, S_other) < r * max(S_dir, S_other)."""
    S_max = max(S_dir, S_other)
    if S_max <= 0:
        return False
    return S_dir >= epsilon and S_dir >= r * S_max


def sample_one_per_concept(
    existence_pool: List[Dict[str, Any]],
    negation_pool: List[Dict[str, Any]],
    rng: random.Random,
    descriptor: Optional[str] = None,
    *,
    require_descriptor_in_text: bool = False,
) -> List[Dict[str, Any]]:
    """One question per concept. Quota: 3 concepts contribute T/F True, 4 contribute T/F False (when available);
    rest get existence or Yes/No negation. Output alternates 'did it' with 'did not'."""
    by_existence: Dict[str, Dict[str, Any]] = {}
    for q in existence_pool:
        cid = q.get("concept_id")
        if cid is not None:
            by_existence[cid] = q
    by_negation: Dict[str, List[Dict[str, Any]]] = {}
    for q in negation_pool:
        cid = q.get("concept_id")
        if cid is not None:
            by_negation.setdefault(cid, []).append(q)
    all_concepts = sorted(set(by_existence.keys()) | set(by_negation.keys()))

    # Remove aggregate leaning if we have all four directional lean concepts (avoid redundant leaning in same clip)
    directional_leans = {"lean_forward", "lean_backward", "lean_left", "lean_right"}
    if "significant_leaning" in all_concepts and directional_leans.issubset(all_concepts):
        all_concepts = [c for c in all_concepts if c != "significant_leaning"]

    # Partition: concepts that can contribute a T/F with answer True (A) vs False (B)
    can_tf_true = []
    can_tf_false = []
    for cid in all_concepts:
        neg_list = by_negation.get(cid, [])
        tf_neg = [q for q in neg_list if q.get("options", {}).get("A") == "True"]
        if not tf_neg:
            continue
        if tf_neg[0].get("answer") == "A":
            can_tf_true.append(cid)
        else:
            can_tf_false.append(cid)

    # Assign slots: 3 T/F True, 4 T/F False (disjoint concepts)
    rng.shuffle(can_tf_true)
    rng.shuffle(can_tf_false)
    tf_true_slots = can_tf_true[:3]
    # T/F False must not use concepts already used for T/F True
    tf_false_candidates = [c for c in can_tf_false if c not in tf_true_slots]
    tf_false_slots = tf_false_candidates[:4]
    rest_concepts = [c for c in all_concepts if c not in tf_true_slots and c not in tf_false_slots]

    def pick_tf_for_concept(cid: str, answer: str) -> Dict[str, Any]:
        tf_neg = [q for q in by_negation.get(cid, []) if q.get("options", {}).get("A") == "True" and q.get("answer") == answer]
        q = dict(rng.choice(tf_neg))
        q["concept_id"] = cid
        return q

    def yes_no_for_concept(cid: str) -> Optional[Dict[str, Any]]:
        ex = by_existence.get(cid)
        yn_neg = [q for q in by_negation.get(cid, []) if q.get("options", {}).get("A") == "Yes"]
        candidates = ([ex] if ex is not None else []) + yn_neg
        if not candidates:
            return None
        out = dict(rng.choice(candidates))
        out["concept_id"] = cid
        return out

    whole_set: List[Dict[str, Any]] = []
    for cid in tf_true_slots:
        whole_set.append(pick_tf_for_concept(cid, "A"))
    for cid in tf_false_slots:
        whole_set.append(pick_tf_for_concept(cid, "B"))
    for cid in rest_concepts:
        repl = yes_no_for_concept(cid)
        if repl is not None:
            whole_set.append(repl)

    # Split by answer and interleave: "did it" (Yes / False) vs "did not" (No / True)
    did_it = []
    did_not = []
    for q in whole_set:
        ans = q.get("answer", "")
        opts = q.get("options", {})
        is_yes_no = opts.get("A") == "Yes"
        if is_yes_no:
            if ans == "A":
                did_it.append(q)
            else:
                did_not.append(q)
        else:
            if ans == "A":
                did_not.append(q)
            else:
                did_it.append(q)

    # Interleave: one did_it, one did_not, then remainder
    result = []
    i, j = 0, 0
    while i < len(did_it) and j < len(did_not):
        result.append(did_it[i])
        result.append(did_not[j])
        i += 1
        j += 1
    result.extend(did_it[i:])
    result.extend(did_not[j:])

    # Replace 20% of negation questions with the positive (existence) form for the same concept
    def is_negation_question(q: Dict[str, Any]) -> bool:
        opts = q.get("options", {})
        if opts.get("A") == "True":
            return True  # T/F negation
        if opts.get("A") == "Yes" and "Was there no" in q.get("question", ""):
            return True  # Yes/No negation phrasing
        return False

    neg_indices = [i for i, q in enumerate(result) if is_negation_question(q)]
    n_flip = max(0, int(round(0.2 * len(neg_indices))))
    if n_flip > 0 and neg_indices:
        for i in rng.sample(neg_indices, min(n_flip, len(neg_indices))):
            cid = result[i].get("concept_id")
            ex = by_existence.get(cid) if cid else None
            if ex is not None:
                result[i] = {k: v for k, v in ex.items() if k != "concept_id"}
                result[i]["concept_id"] = cid

    # Strip concept_id from output
    for q in result:
        q.pop("concept_id", None)
    return personalize_qa_questions(
        result, descriptor, require_descriptor_in_text=require_descriptor_in_text
    )


def extract_events(motion_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract events that pass threshold and last at least MIN_EVENT_FRAMES from quantitative_values."""
    events = []
    for code, segments in motion_json["quantitative_values"].items():
        for seg in segments:
            q = seg["quant"]
            if not q["passes_threshold"]:
                continue
            if q.get("duration_frames", 0) < MIN_EVENT_FRAMES:
                continue
            events.append({
                "code": code,
                "t0": q["start_frame"],
                "t1": q["end_frame"],
                "duration": q["duration_frames"],
                "intensity": q["intensity"],
                "velocity": abs(q["velocity"]),
                "spatial_category": q["spatial_category"],
                "temporal_category": q["temporal_category"],
            })
    return events


def extract_displacement_segments(motion_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract displacement segments (x, z only; no displacement_y) from quantitative_values, excluding very_short.
    Same motion-track source as events: each segment carries spatial_category and temporal_category from the track.
    Returns list of dicts: axis, start_frame, end_frame, intensity, spatial_category, temporal_category."""
    segments_out = []
    for axis in ("displacement_x", "displacement_z"):
        raw = motion_json.get("quantitative_values", {}).get(axis, [])
        for seg in raw:
            q = seg.get("quant", {})
            sc = q.get("spatial_category") or ""
            if "very_short" in sc:
                continue
            segments_out.append({
                "axis": axis,
                "start_frame": q.get("start_frame"),
                "end_frame": q.get("end_frame"),
                "intensity": q.get("intensity", 0),
                "spatial_category": sc,
                "temporal_category": q.get("temporal_category") or "",
            })
    return segments_out


def compute_trajectory_stats(
    segments: List[Dict[str, Any]], num_frames: int, fps: float, temporal_boundaries: List[float]
) -> Dict[str, Any]:
    """Compute trajectory stats per quarter (1-4) and overall from displacement segments (x, z only; no displacement_y).
    Assigns each segment to the quarter containing its start time (same rule as temporal: get_quarter(start_frame)).
    Returns dict with per_quarter[q]: net_dx, net_dz, path_x, path_z; and overall same."""
    tb = temporal_boundaries
    per_quarter: Dict[int, Dict[str, float]] = {1: _zero_disp(), 2: _zero_disp(), 3: _zero_disp(), 4: _zero_disp()}
    overall = _zero_disp()

    def quarter_for_segment(s: Dict[str, Any]) -> int:
        """Same as temporal: quarter by segment start time (start_frame)."""
        t = s["start_frame"] / fps
        for i in range(4):
            if t < tb[i + 1]:
                return i + 1
        return 4

    for s in segments:
        axis = s["axis"]
        intensity = s["intensity"]
        q = quarter_for_segment(s)
        if axis == "displacement_x":
            per_quarter[q]["net_dx"] += intensity
            per_quarter[q]["path_x"] += abs(intensity)
            overall["net_dx"] += intensity
            overall["path_x"] += abs(intensity)
        elif axis == "displacement_z":
            per_quarter[q]["net_dz"] += intensity
            per_quarter[q]["path_z"] += abs(intensity)
            overall["net_dz"] += intensity
            overall["path_z"] += abs(intensity)

    return {
        "per_quarter": per_quarter,
        "overall": overall,
        "temporal_boundaries": tb,
        "segments_used": segments,
    }


def _zero_disp() -> Dict[str, float]:
    return {"net_dx": 0.0, "net_dy": 0.0, "net_dz": 0.0, "path_x": 0.0, "path_y": 0.0, "path_z": 0.0}


def _argmax_earliest_quarter(pairs: List[Tuple[int, float]]) -> Optional[int]:
    """Given [(quarter, value), ...], return quarter with max value; on tie, earliest quarter (smallest q)."""
    if not pairs:
        return None
    best_val = max(v for (_, v) in pairs)
    candidates = [q for (q, v) in pairs if v == best_val]
    return min(candidates) if candidates else None


def _argmin_earliest_quarter(pairs: List[Tuple[int, float]]) -> Optional[int]:
    """Given [(quarter, value), ...], return quarter with min value; on tie, earliest quarter (smallest q)."""
    if not pairs:
        return None
    best_val = min(v for (_, v) in pairs)
    candidates = [q for (q, v) in pairs if v == best_val]
    return min(candidates) if candidates else None


def build_trajectory_from_segments(segments: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    """Build pos_t trajectory (x, z only) from displacement segments. Each segment contributes at its end_frame.
    Returns list of (frame, pos_x, pos_z) sorted by frame. pos_0 = (0, 0)."""
    events = [(s["end_frame"], s["axis"], s["intensity"]) for s in segments]
    events.sort(key=lambda e: e[0])
    out = [(0, 0.0, 0.0)]
    pos_x, pos_z = 0.0, 0.0
    for frame, axis, intensity in events:
        if axis == "displacement_x":
            pos_x += intensity
        elif axis == "displacement_z":
            pos_z += intensity
        out.append((frame, pos_x, pos_z))
    return out


def dist_to_start(pos_x: float, pos_z: float) -> float:
    """Euclidean distance from (pos_x, pos_z) to start (0, 0)."""
    return math.sqrt(pos_x * pos_x + pos_z * pos_z)


def _count_direction_changes(dists: List[float]) -> int:
    """Count sign changes in the distance-from-start curve (peaks + valleys = direction changes).
    Ignores flat steps (diff < 1e-9). Used to classify trajectory shape as oscillating vs monotone."""
    last_sign = None
    changes = 0
    for i in range(1, len(dists)):
        diff = dists[i] - dists[i - 1]
        if abs(diff) < 1e-9:
            continue
        sign = 1 if diff > 0 else -1
        if last_sign is not None and sign != last_sign:
            changes += 1
        last_sign = sign
    return changes


# Returnability: "come close again" = after first reaching "away" (dist >= left_threshold), min dist after that <= return_threshold
RETURN_ABSOLUTE_MIN_THRESH = 1.0   # absolute min for return_threshold (same units as pos)
RETURN_CLOSE_FRACTION = 0.1        # relative: return_threshold = max(absolute_min, relative * max_dist)
# Minimum distance (fraction of max) to count as "left start" (must reach this before we check "come close again")
RETURN_LEFT_FRACTION = 0.5
# Drop from peak (fraction of max) to count as "oscillates"
RETURN_OSCILLATE_DROP_FRACTION = 0.3
# No movement if max_dist below this (absolute)
NO_MOVEMENT_THRESHOLD = 0.5
# Trajectory shape: min direction-change count to classify as "oscillates"
TRAJ_SHAPE_HIGH_VARIANCE_MIN_CHANGES = 2

# Overall net_dx 4-way: "about same" and "no x-motion"
EPS_DX_ABSOLUTE = 1.0       # minimum eps_dx (same units as intensity)
EPS_DX_RELATIVE = 0.05     # eps_dx = max(absolute, relative * path_x)
NO_X_MOTION_THRESHOLD = 0.01  # path_x below this -> D (no x-motion)

# Explicit policy fields for evidence (not buried in strings)
TIE_EPS = 1e-6
TRACK_UNITS = "track_units"
QUARTER_ASSIGNMENT_SEGMENT = "segment_start"   # segment assigned by its start_frame
QUARTER_ASSIGNMENT_POINT_FRAME = "point_frame"  # trajectory point in quarter by its frame
TIE_BREAK = "earliest_quarter"
EPS_DX_POLICY_STR = "max(1.0, 0.05*path_x)"


def compute_returnability(trajectory: List[Tuple[int, float, float]]) -> str:
    """Returnability: after first reaching "away" (dist >= left_threshold), did they ever come close to start again?
    A = yes (min dist after leaving <= return_threshold), B = no, C = unclear (no movement), D = oscillates but doesn't return."""
    d = compute_returnability_detail(trajectory)
    return d["answer"]


def compute_returnability_detail(trajectory: List[Tuple[int, float, float]]) -> Dict[str, Any]:
    """Return full returnability result for evidence: answer, away_threshold_units, first_away_frame, min_dist_after_away_units, returned."""
    out = {
        "answer": "C",
        "away_threshold_units": None,
        "first_away_frame": None,
        "min_dist_after_away_units": None,
        "returned": False,
        "return_threshold_units": None,
        "max_dist_units": None,
    }
    if len(trajectory) <= 1:
        return out
    dists = [dist_to_start(px, pz) for (_, px, pz) in trajectory]
    frames = [f for (f, _, _) in trajectory]
    max_d = max(dists)
    out["max_dist_units"] = max_d
    if max_d < NO_MOVEMENT_THRESHOLD:
        return out
    return_threshold = max(RETURN_ABSOLUTE_MIN_THRESH, RETURN_CLOSE_FRACTION * max_d)
    left_threshold = RETURN_LEFT_FRACTION * max_d
    out["return_threshold_units"] = return_threshold
    out["away_threshold_units"] = left_threshold
    leave_idx = None
    for i, d in enumerate(dists):
        if d >= left_threshold:
            leave_idx = i
            break
    if leave_idx is None:
        return out
    out["first_away_frame"] = frames[leave_idx]
    min_after_leave = min(dists[leave_idx:])
    out["min_dist_after_away_units"] = min_after_leave
    if min_after_leave <= return_threshold:
        out["answer"] = "A"
        out["returned"] = True
        return out
    peak = dists[leave_idx]
    for i in range(leave_idx + 1, len(dists)):
        if dists[i] > peak:
            peak = dists[i]
        elif peak - dists[i] >= RETURN_OSCILLATE_DROP_FRACTION * max_d and dists[i] > return_threshold:
            out["answer"] = "D"
            return out
    out["answer"] = "B"
    return out


def _return_threshold_for_evidence(max_d: float) -> float:
    """Return threshold used in returnability (for evidence string)."""
    return max(RETURN_ABSOLUTE_MIN_THRESH, RETURN_CLOSE_FRACTION * max_d)


def _quarter_frame_bounds(tb: List[float], fps: float) -> List[Tuple[float, float]]:
    """Return (f_start, f_end) in frames for each quarter (1-4)."""
    return [(tb[i] * fps, tb[i + 1] * fps) for i in range(4)]


def closest_quarter_to_start(
    trajectory: List[Tuple[int, float, float]], tb: List[float], fps: float
) -> Optional[int]:
    """Quarter (1-4) with smallest min distance-to-start within the quarter. Tie-break: earliest quarter."""
    result = closest_quarter_to_start_with_table(trajectory, tb, fps)
    return result["answer_quarter"] if result else None


def closest_quarter_to_start_with_table(
    trajectory: List[Tuple[int, float, float]], tb: List[float], fps: float
) -> Optional[Dict[str, Any]]:
    """Returns {answer_quarter, min_dist_by_quarter_units} or None if no trajectory points in any quarter."""
    bounds = _quarter_frame_bounds(tb, fps)
    pairs: List[Tuple[int, float]] = []
    min_dist_by_quarter_units: Dict[str, float] = {}
    for q, (f_lo, f_hi) in enumerate(bounds, start=1):
        pts = [(f, dist_to_start(px, pz)) for (f, px, pz) in trajectory if f_lo <= f <= f_hi]
        if not pts:
            min_dist_by_quarter_units[str(q)] = None  # no trajectory points in this quarter
            continue
        min_d = min(d for (_, d) in pts)
        min_dist_by_quarter_units[str(q)] = min_d
        pairs.append((q, min_d))
    if not pairs:
        return None
    best_q = _argmin_earliest_quarter(pairs)
    if best_q is None:
        return None
    return {"answer_quarter": best_q, "min_dist_by_quarter_units": min_dist_by_quarter_units}


def _tie_info(pairs: List[Tuple[int, float]], best_quarter: int, tie_eps: float) -> Tuple[bool, List[int]]:
    """Given (quarter, value) pairs and the chosen best_quarter, return is_tie and list of tied quarters."""
    best_val = next((v for (q, v) in pairs if q == best_quarter), None)
    if best_val is None:
        return False, []
    tied = [q for (q, v) in pairs if abs(v - best_val) < tie_eps]
    return len(tied) > 1, sorted(tied)


def _evidence_policy_quarter_segment(best_quarter: Optional[int], pairs: List[Tuple[int, float]]) -> Dict[str, Any]:
    """Common policy fields for segment-based quarter-aggregate questions."""
    policy = {
        "units": TRACK_UNITS,
        "quarter_assignment": QUARTER_ASSIGNMENT_SEGMENT,
        "tie_break": TIE_BREAK,
        "tie_eps": TIE_EPS,
    }
    if best_quarter is not None and pairs:
        is_tie, tied_quarters = _tie_info(pairs, best_quarter, TIE_EPS)
        policy["is_tie"] = is_tie
        policy["tied_quarters"] = tied_quarters
    return policy


def get_direction(event: Dict[str, Any]) -> Optional[str]:
    """Extract direction from spatial_category (e.g., 'long_right' -> 'right')."""
    category = event["spatial_category"]
    code = event["code"]
    
    if code == "displacement_x":
        if "left" in category:
            return "left"
        elif "right" in category:
            return "right"
    elif code == "displacement_y":
        if "down" in category:
            return "down"
        elif "up" in category:
            return "up"
    elif code == "displacement_z":
        if "backward" in category:
            return "backward"
        elif "forward" in category:
            return "forward"
    elif code == "rotation_yaw":
        if "counterclockwise" in category:
            return "counterclockwise"
        elif "clockwise" in category:
            return "clockwise"
    elif code == "rotation_pitch":
        if "backward" in category:
            return "backward"
        elif "forward" in category:
            return "forward"
    elif code == "rotation_roll":
        if "right" in category:
            return "right"
        elif "left" in category:
            return "left"
    
    return None


def is_significant_motion(event: Dict[str, Any]) -> bool:
    """Check if motion is significant enough for semantic questions."""
    significant_categories = ["long", "very_long", "significant"]
    return any(cat in event["spatial_category"] for cat in significant_categories)


# For displacement_z we phrase w.r.t. starting position (first-frame normalized)
Z_DEPTH_LABELS = {"forward": "away from starting position", "backward": "towards starting position"}


def get_rotation_strength(event: Dict[str, Any]) -> int:
    """Return rotation strength from spatial_category: 0=none, 1=slight, 2=moderate, 3=significant.
    Used for rotation (yaw/pitch/roll) questions only; categories like slight_turn_clockwise, moderate_leaning_left."""
    cat = event.get("spatial_category", "") or ""
    if "significant" in cat:
        return 3
    if "moderate" in cat:
        return 2
    if "slight" in cat:
        return 1
    return 0


# Motion codes included in ordering (sequence) questions; cross-axis allowed
ORDERING_MOTION_CODES = ["displacement_x", "displacement_z", "rotation_yaw", "rotation_pitch", "rotation_roll"]
# Next event after A ends: require B.t0 >= A.t1 + this many frames
ORDERING_MIN_GAP_AFTER_A_END_FRAMES = 10

# -----------------------------------------------------------------------------
# Ordering question logic (relaxed, single source of truth)
# -----------------------------------------------------------------------------
# Question: "In the [first|second|third|fourth] quarter of the video, after the
# person [A's canonical label], which of these happens next?" — 4 options, 1 correct.
#
# 1) Scope: one quarter
#    - Pick a quarter (t_start .. t_end from temporal_boundaries).
#    - Anchor A, correct next event, and distractors are taken from this quarter.
#    - No cross-quarter reasoning.
#
# 2) Event eligibility (tagged set)
#    - Motion code in ORDERING_MOTION_CODES:
#        displacement_x, displacement_z, rotation_yaw, rotation_pitch, rotation_roll
#    - Must have a defined direction (get_direction not None).
#    - Moderate or more:
#        displacement = moderate / long / very_long
#        rotation = strength >= 2
#
# 3) Family constraint (softened)
#    - Family = event["code"].
#    - Anchor A and event B can be different family if both are displacement or both are rotation
#    - Distractors can be from anywhere in the quarter (no same-family requirement).
#
# 4) Anchor selection (relaxed)
#    - Canonical label = label without intensity adverbs (e.g. "leans left", "moves right").
#    - Use ANY eligible event as A (no uniqueness requirement).
#    - Disambiguate by occurrence index in the stem:
#        "after the first/second/third/… time the person [label], …"
#      (1-based index among events with the same canonical label in the quarter, ordered by t0).
#    - If the label appears only once in the quarter: "after the person [label], …".
#
# 5) Correct answer B ("next")
#    - B = first eligible event in the same family (same quarter) with:
#        B.t0 >= A.t1 + ORDERING_MIN_GAP_AFTER_A_END_FRAMES
#    - ORDERING_MIN_GAP_AFTER_A_END_FRAMES = 10
#    - If no valid B, skip this A.
#
# 5a) Allow repeats (canonical(B) may equal canonical(A))
#    - If canonical(B) != canonical(A): normal ordering QA (as usual).
#    - If canonical(B) == canonical(A):
#        still allow a normal ordering QA (no special repeat-next required),
#        BUT disambiguate the question as:
#        "after the person [label] the first time / again, what happens next?"
#
# 6) Distractors (much easier to fill)
#    - Total options = 4 (1 correct + 3 distractors).
#    - Distractors are any eligible events in the same quarter
#      excluding the correct event (and excluding A if you want).
#    - No hard ban on including canonical(A) or canonical(B) in options.
#      (If it causes duplicate option strings, see rule 7.)
#    - Prefer time-local distractors (near A or near B), but fallback can be random
#      from the quarter to ensure question yield.
#
# 7) Option labels (canonical only; one per label)
#    - Use canonical labels only (no intensity adverbs).
#    - If multiple events share the same canonical label (e.g. several "moves left"),
#      only keep one in the options (pick one event per distinct label).
#    - Skip if you cannot make 4 distinct option strings.
#
# 8) Output + dedup
#    - One QA per valid (quarter, A, B).
#    - Evidence stores: event_A, event_B, times, bucket, computation string.
#    - Deduplication: drop duplicates by (bucket, event_A.t0, event_B.t0, options, answer)
#      (this avoids collapsing genuinely different buckets/events that happen to share text).
# -----------------------------------------------------------------------------


def _ordering_family(tagged_event: Dict[str, Any]) -> str:
    """Motion family for same-family option rule: one code per family (roll, pitch, yaw, translation-x, translation-z)."""
    return tagged_event["event"]["code"]


def _ordering_super_family(tagged_event: Dict[str, Any]) -> str:
    """Super-family for relaxed ordering: displacement (x,z) vs rotation (yaw, pitch, roll). B can be same super-family as A."""
    code = tagged_event["event"]["code"]
    if code in ["displacement_x", "displacement_z"]:
        return "displacement"
    if code in ["rotation_yaw", "rotation_pitch", "rotation_roll"]:
        return "rotation"
    return "other"


def _ordering_ordinal(n: int) -> str:
    """Ordinal word for occurrence index: 1 -> 'first', 2 -> 'second', etc."""
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    if 1 <= n <= len(ordinals):
        return ordinals[n - 1]
    return f"{n}th"


def is_moderate_or_more(event: Dict[str, Any]) -> bool:
    """True if event is moderate, long, or very long (spatial). Used for ordering and speed questions."""
    code = event.get("code", "")
    cat = event.get("spatial_category", "") or ""
    if code in ["rotation_yaw", "rotation_pitch", "rotation_roll"]:
        return get_rotation_strength(event) >= 2
    if code in ["displacement_x", "displacement_z"]:
        return "moderate" in cat or "long" in cat  # includes very_long
    return False


def get_event_label(event: Dict[str, Any]) -> str:
    """Canonical label for an event: type + direction only, no intensity (e.g. 'leans left', 'moves right').
    Used for ordering question options and for unique-anchor check."""
    direction = get_direction(event)
    if direction is None:
        return "moves or rotates"
    code = event["code"]
    if code == "displacement_x":
        return "moves left" if direction == "left" else "moves right"
    if code == "displacement_z":
        return f"moves {Z_DEPTH_LABELS['forward'].lower()}" if direction == "forward" else f"moves {Z_DEPTH_LABELS['backward'].lower()}"
    if code == "rotation_yaw":
        return "turns clockwise" if direction == "clockwise" else "turns counterclockwise"
    if code == "rotation_pitch":
        return "leans forward" if direction == "forward" else "leans backward"
    if code == "rotation_roll":
        return "leans to the left" if direction == "left" else "leans to the right"
    return "moves or rotates"


def get_ordering_intensity_label(event: Dict[str, Any]) -> str:
    """Short intensity for ordering disambiguation: 'moderate', 'long', 'very long' (displacement) or 'slight', 'moderate', 'significant' (rotation)."""
    code = event.get("code", "")
    cat = (event.get("spatial_category") or "").lower()
    if code in ["rotation_yaw", "rotation_pitch", "rotation_roll"]:
        s = get_rotation_strength(event)
        return "significant" if s == 3 else "moderate" if s == 2 else "slight" if s == 1 else "none"
    if "very_long" in cat:
        return "very long"
    if "long" in cat:
        return "long"
    if "moderate" in cat:
        return "moderate"
    return "moderate"


def _ordering_intensity_to_adverb(intensity: str) -> str:
    """Adverb for option text: 'leans moderately backward' not 'leans backward (moderate)'."""
    m = {
        "moderate": "moderately",
        "long": "considerably",
        "very long": "significantly",
        "slight": "slightly",
        "significant": "significantly",
        "none": "",
    }
    return m.get(intensity, "moderately")


def _ordering_label_with_intensity(label: str, intensity: str) -> str:
    """Insert intensity as adverb: 'leans backward' + 'moderately' -> 'leans moderately backward'."""
    adverb = _ordering_intensity_to_adverb(intensity)
    if not adverb:
        return label
    parts = label.split(maxsplit=1)
    if len(parts) < 2:
        return label
    verb, direction = parts[0], parts[1]
    return f"{verb} {adverb} {direction}"


def _ordering_option_label(tagged_event: Dict[str, Any], among: List[Dict[str, Any]]) -> str:
    """Option string: canonical label (no intensity) by default; only append intensity adverb when
    two different events in among would otherwise produce the same string."""
    label = tagged_event["label"]
    intensity = tagged_event.get("intensity", "")
    same_label = [x for x in among if x["label"] == label]
    if len(same_label) <= 1:
        return label
    return _ordering_label_with_intensity(label, intensity)


def get_event_label_base(event: Dict[str, Any]) -> str:
    """Base form for speed question text: 'move left', 'turn clockwise', etc."""
    direction = get_direction(event)
    if direction is None:
        return "move or rotate"
    code = event["code"]
    if code == "displacement_x":
        return "move left" if direction == "left" else "move right"
    if code == "displacement_z":
        return f"move {Z_DEPTH_LABELS['forward'].lower()}" if direction == "forward" else f"move {Z_DEPTH_LABELS['backward'].lower()}"
    if code == "rotation_yaw":
        return "turn clockwise" if direction == "clockwise" else "turn counterclockwise"
    if code == "rotation_pitch":
        return "lean forward" if direction == "forward" else "lean backward"
    if code == "rotation_roll":
        return "lean to the left" if direction == "left" else "lean to the right"
    return "move or rotate"


# Speed axis: use temporal_category from JSON (very_slow, slow, moderate, fast, very_fast)
SPEED_4_OPTIONS = {"A": "very_slow", "B": "slow", "C": "moderate", "D": "fast/very_fast"}
SPEED_3_OPTIONS = {"A": "slow", "B": "moderate", "C": "fast/very_fast"}
SPEED_5_CATEGORIES = ["very_slow", "slow", "moderate", "fast", "very_fast"]


def _is_speed_eligible(event: Dict[str, Any]) -> bool:
    """True if event passes the speed-axis filter: not very_short, not slight spatial category."""
    cat = (event.get("spatial_category") or "").lower()
    return "very_short" not in cat and "slight" not in cat


def get_temporal_speed_4(temporal_category: str) -> str:
    """Map temporal_category to 4-way option letter for classification questions."""
    cat = (temporal_category or "").strip().lower()
    if cat == "very_slow":
        return "A"
    if cat == "slow":
        return "B"
    if cat == "moderate":
        return "C"
    if cat in ("fast", "very_fast"):
        return "D"
    return "C"  # default moderate if missing


def get_temporal_speed_level(temporal_category: str) -> int:
    """Numeric level for comparison: 0=very_slow .. 4=very_fast."""
    cat = (temporal_category or "").strip().lower()
    if cat == "very_slow":
        return 0
    if cat == "slow":
        return 1
    if cat == "moderate":
        return 2
    if cat == "fast":
        return 3
    if cat == "very_fast":
        return 4
    return 2


def get_temporal_speed_3(temporal_category: str) -> str:
    """Map temporal_category to 3-way for next-translation: slow / moderate / fast."""
    cat = (temporal_category or "").strip().lower()
    if cat in ("very_slow", "slow"):
        return "slow"
    if cat == "moderate":
        return "moderate"
    if cat in ("fast", "very_fast"):
        return "fast/very_fast"
    return "moderate"


def get_semantic_action(event: Dict[str, Any]) -> Optional[str]:
    """Map motion code to semantic action if significant."""
    if not is_significant_motion(event):
        return None
    
    code = event["code"]
    if code in ["displacement_x", "displacement_z"]:
        return "walking"
    elif code == "displacement_y":
        return "vertical_movement"
    elif code == "rotation_yaw":
        return "turning"
    elif code in ["rotation_pitch", "rotation_roll"]:
        return "leaning"
    return None


def frame_to_time(frame: int, fps: float) -> float:
    """Convert frame number to timestamp in seconds."""
    return frame / fps


def event_overlaps_time_range(event: Dict[str, Any], fps: float, t_start: float, t_end: float) -> bool:
    """Return True when an event overlaps the half-open interval [t_start, t_end)."""
    event_start = frame_to_time(event["t0"], fps)
    event_end_frame = event.get("t1", event["t0"])
    event_end = frame_to_time(event_end_frame + 1, fps)
    return event_start < t_end and event_end > t_start


def get_events_in_time_range(events: List[Dict[str, Any]], fps: float, t_start: float, t_end: float) -> List[Dict[str, Any]]:
    """Get events that overlap the given time range."""
    return [e for e in events if event_overlaps_time_range(e, fps, t_start, t_end)]


def get_quarter(frame: int, fps: float, temporal_boundaries: List[float]) -> int:
    """Return quarter 1-4 based on event start time and 4-quarter temporal boundaries."""
    t = frame_to_time(frame, fps)
    for i in range(4):
        if t < temporal_boundaries[i + 1]:
            return i + 1
    return 4


def get_event_quarters(event: Dict[str, Any], fps: float, temporal_boundaries: List[float]) -> List[int]:
    """Return all quarter indices whose time windows overlap the event."""
    quarters = []
    for i in range(len(temporal_boundaries) - 1):
        if event_overlaps_time_range(event, fps, temporal_boundaries[i], temporal_boundaries[i + 1]):
            quarters.append(i + 1)
    return quarters or [get_quarter(event["t0"], fps, temporal_boundaries)]


def events_used_with_quarters(events: List[Dict[str, Any]], fps: float, temporal_boundaries: List[float]) -> List[Dict[str, Any]]:
    """Return copies of events with start-quarter and all overlapping quarters for evidence."""
    return [
        {
            **e,
            "quarter": get_quarter(e["t0"], fps, temporal_boundaries),
            "quarters_active": get_event_quarters(e, fps, temporal_boundaries),
        }
        for e in events
    ]


# Temporal QA: 4 duration-scaled buckets (same relative time for any video length)
NUM_TEMPORAL_BUCKETS = 4
# Labels for the 4 buckets (no exact seconds; works for any video length)
TEMPORAL_BUCKET_LABELS = ["first quarter", "second quarter", "third quarter", "fourth quarter"]
# First-occurrence questions: which of the 4 quarters + "never"
NUM_FIRST_OCCURRENCE_BUCKETS = NUM_TEMPORAL_BUCKETS


def compute_stats(events: List[Dict[str, Any]], num_frames: int, fps: float, rng: random.Random) -> Dict[str, Any]:
    """Compute statistics from events."""
    duration = num_frames / fps
    
    # Time boundaries with jitter (for dominant: early / middle / end)
    jitter = 0.5
    t1 = duration / 3 + rng.uniform(-jitter, jitter)
    t2 = 2 * duration / 3 + rng.uniform(-jitter, jitter)
    t1 = max(0.5, min(t1, duration - 1))
    t2 = max(t1 + 0.5, min(t2, duration - 0.5))
    boundaries = [0, round(t1, 1), round(t2, 1), round(duration, 1)]
    
    # Temporal QA: 4 equal quarters (0-25%, 25-50%, 50-75%, 75-100%)
    temporal_boundaries = [round(i * duration / NUM_TEMPORAL_BUCKETS, 1) for i in range(NUM_TEMPORAL_BUCKETS + 1)]
    temporal_boundaries[-1] = round(duration, 1)
    
    # Counts and sums by direction/type
    stats = {
        "boundaries": boundaries,
        "temporal_boundaries": temporal_boundaries,
        "duration": duration,
        "fps": fps,
        "num_frames": num_frames,
        "displacement_x": {"left": [], "right": []},
        "displacement_y": {"up": [], "down": []},
        "displacement_z": {"forward": [], "backward": []},
        "rotation_yaw": {"clockwise": [], "counterclockwise": []},
        "rotation_pitch": {"forward": [], "backward": []},
        "rotation_roll": {"left": [], "right": []},
    }
    
    for event in events:
        code = event["code"]
        direction = get_direction(event)
        if direction is None:
            continue
        
        if code in stats and direction in stats[code]:
            stats[code][direction].append(event)
    
    return stats


def _events_for_counting(events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Exclude very_short and slight spatial_category so counting only uses meaningful events."""
    return [e for e in events_list if _is_speed_eligible(e)]


def _exclude_very_short_events(events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out events whose spatial_category contains 'very_short' or 'slight'. Use once after extract_events so all QA axes see the same set."""
    return [e for e in events_list if _is_speed_eligible(e)]


def _counting_options_for_count(c: int, rng: random.Random) -> Tuple[Dict[str, str], str]:
    """Four options: correct = c (exact integer) + 3 distinct distractors from [max(0, c-3), c+3].
    Distractors sampled freely so the correct answer can be the lowest, highest, or middle value.
    Options shuffled uniformly."""
    lo = max(0, c - 3)
    hi = c + 3
    pool = [x for x in range(lo, hi + 1) if x != c]
    distractors = rng.sample(pool, min(3, len(pool)))

    # Pad if pool was too small (e.g. c=0 has only 3 candidates above)
    extra = hi + 1
    while len(distractors) < 3:
        if extra != c and extra not in distractors:
            distractors.append(extra)
        extra += 1

    options_list = [(str(c), True)] + [(str(d), False) for d in distractors[:3]]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: label for i, (label, _) in enumerate(options_list)}
    answer = next(letters[i] for i, (_, is_correct) in enumerate(options_list) if is_correct)
    return options, answer


def _magnitude_options_for_value(v: float, rng: random.Random, unit_suffix: str = " units") -> Tuple[Dict[str, str], str]:
    """Four options: correct = round(v) (exact integer) + 3 distinct distractors sampled from
    [max(0, c - spread), c + spread] where spread = max(c * 0.2, 10).
    Distractors sampled freely so the correct answer can be the lowest, highest, or middle value.
    Options shuffled uniformly."""
    c = max(0, int(round(v)))
    spread = int(round(max(c * 0.2, 10)))
    lo = max(0, c - spread)
    hi = c + spread
    pool = [x for x in range(lo, hi + 1) if x != c]

    def fmt(x: int) -> str:
        return f"{x}{unit_suffix}"

    distractors = rng.sample(pool, min(3, len(pool)))

    # Pad upward if pool was too small
    extra = hi + 1
    while len(distractors) < 3:
        if extra not in distractors:
            distractors.append(extra)
        extra += 1

    options_list = [(fmt(c), True)] + [(fmt(d), False) for d in distractors[:3]]
    rng.shuffle(options_list)
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: label for i, (label, _) in enumerate(options_list)}
    answer = next(letters[i] for i, (_, is_correct) in enumerate(options_list) if is_correct)
    return options, answer


STRENGTH_LABELS = ["not at all", "slightly", "moderately", "significantly"]


def _strength_options_shuffled(strength: int, rng: random.Random) -> Tuple[Dict[str, str], str]:
    """Return options A/B/C/D for strength 0-3, with option order shuffled."""
    labels = list(STRENGTH_LABELS)
    correct_label = labels[strength] if 0 <= strength < len(labels) else labels[0]
    # Shuffle order of the four labels, then assign A/B/C/D
    rng.shuffle(labels)
    options = {letter: label for letter, label in zip(["A", "B", "C", "D"], labels)}
    answer = next(letter for letter, label in options.items() if label == correct_label)
    return options, answer


def generate_numerical_qa_pool(events: List[Dict[str, Any]], stats: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Generate pool of numerical questions."""
    qa_pool = []
    
    # Count-based questions: correct option is "around c" with nearby distractors
    # Exclude very_short/slight spatial_category from counts
    left_events = _events_for_counting(stats["displacement_x"]["left"])
    right_events = _events_for_counting(stats["displacement_x"]["right"])
    left_count = len(left_events)
    right_count = len(right_events)
    clockwise_events = _events_for_counting(stats["rotation_yaw"]["clockwise"])
    clockwise_count = len(clockwise_events)

    # Left movements
    opts, ans = _counting_options_for_count(left_count, rng)
    qa_pool.append({
        "question": "How many times does the person move to the left?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": left_events,
            "computation": f"count(left_events, exclude very_short/slight) = {left_count}",
            "raw_values": {"left_count": left_count}
        }
    })

    # Right movements
    opts, ans = _counting_options_for_count(right_count, rng)
    qa_pool.append({
        "question": "How many times does the person move to the right?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": right_events,
            "computation": f"count(right_events, exclude very_short/slight) = {right_count}",
            "raw_values": {"right_count": right_count}
        }
    })

    # Clockwise turns
    opts, ans = _counting_options_for_count(clockwise_count, rng)
    qa_pool.append({
        "question": "How many clockwise turns does the person make?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": clockwise_events,
            "computation": f"count(clockwise_turns, exclude very_short/slight) = {clockwise_count}",
            "raw_values": {"clockwise_count": clockwise_count}
        }
    })
    
    # Magnitude-based questions (same band template: correct around true value, near-low, near-high, far)
    # Left total distance (in units)
    left_total = sum(abs(e["intensity"]) for e in stats["displacement_x"]["left"])
    opts, ans = _magnitude_options_for_value(left_total, rng)
    qa_pool.append({
        "question": "How far does the person move to the left in total?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": stats["displacement_x"]["left"],
            "computation": f"sum(|intensity|) = {left_total} units",
            "raw_values": {"total_units": left_total}
        }
    })

    # Right total distance (in units)
    right_total = sum(abs(e["intensity"]) for e in stats["displacement_x"]["right"])
    opts, ans = _magnitude_options_for_value(right_total, rng)
    qa_pool.append({
        "question": "How far does the person move to the right in total?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": stats["displacement_x"]["right"],
            "computation": f"sum(|intensity|) = {right_total} units",
            "raw_values": {"total_units": right_total}
        }
    })
    
    # Displacement_z: away = forward (+z), towards = backward (-z)
    away_events = _events_for_counting(stats["displacement_z"]["forward"])
    towards_events = _events_for_counting(stats["displacement_z"]["backward"])
    away_count = len(away_events)
    towards_count = len(towards_events)
    opts, ans = _counting_options_for_count(away_count, rng)
    qa_pool.append({
        "question": "How many times does the person move away from starting position?",
        "options": opts,
        "answer": ans,
        "evidence": {"events_used": away_events, "computation": f"count(away_from_start_events, exclude very_short/slight) = {away_count}", "raw_values": {"away_count": away_count}}
    })
    opts, ans = _counting_options_for_count(towards_count, rng)
    qa_pool.append({
        "question": "How many times does the person move towards starting position?",
        "options": opts,
        "answer": ans,
        "evidence": {"events_used": towards_events, "computation": f"count(towards_start_events, exclude very_short/slight) = {towards_count}", "raw_values": {"towards_count": towards_count}}
    })
    # Z: away from starting position total (in units)
    away_total = sum(abs(e["intensity"]) for e in stats["displacement_z"]["forward"])
    opts, ans = _magnitude_options_for_value(away_total, rng)
    qa_pool.append({
        "question": "How far does the person move away from starting position in total?",
        "options": opts,
        "answer": ans,
        "evidence": {"events_used": stats["displacement_z"]["forward"], "computation": f"sum(|intensity|) = {away_total} units", "raw_values": {"total_units": away_total}}
    })
    # Z: towards starting position total (in units)
    towards_total = sum(abs(e["intensity"]) for e in stats["displacement_z"]["backward"])
    opts, ans = _magnitude_options_for_value(towards_total, rng)
    qa_pool.append({
        "question": "How far does the person move towards starting position in total?",
        "options": opts,
        "answer": ans,
        "evidence": {"events_used": stats["displacement_z"]["backward"], "computation": f"sum(|intensity|) = {towards_total} units", "raw_values": {"total_units": towards_total}}
    })
    
    # Clockwise turn counts by exact strength category (slight already excluded upstream)
    clockwise_events = stats["rotation_yaw"]["clockwise"]
    clockwise_moderate_events = [e for e in clockwise_events if get_rotation_strength(e) == 2]
    clockwise_moderate_count = len(clockwise_moderate_events)
    opts, ans = _counting_options_for_count(clockwise_moderate_count, rng)
    qa_pool.append({
        "question": "How many times does the person turn clockwise moderately?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": clockwise_moderate_events,
            "computation": f"count moderate clockwise turning events (exclude very_short/slight, exclude significant) = {clockwise_moderate_count}",
            "raw_values": {"moderate_count": clockwise_moderate_count}
        }
    })
    clockwise_significant_events = [e for e in clockwise_events if get_rotation_strength(e) == 3]
    clockwise_significant_count = len(clockwise_significant_events)
    opts, ans = _counting_options_for_count(clockwise_significant_count, rng)
    qa_pool.append({
        "question": "How many times does the person turn clockwise significantly?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": clockwise_significant_events,
            "computation": f"count significant clockwise turning events (exclude very_short/slight, exclude moderate) = {clockwise_significant_count}",
            "raw_values": {"significant_count": clockwise_significant_count}
        }
    })

    # Counterclockwise turn counts by exact strength category
    counterclockwise_events = stats["rotation_yaw"]["counterclockwise"]
    counterclockwise_moderate_events = [e for e in counterclockwise_events if get_rotation_strength(e) == 2]
    counterclockwise_moderate_count = len(counterclockwise_moderate_events)
    opts, ans = _counting_options_for_count(counterclockwise_moderate_count, rng)
    qa_pool.append({
        "question": "How many times does the person turn counterclockwise moderately?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": counterclockwise_moderate_events,
            "computation": f"count moderate counterclockwise turning events (exclude very_short/slight, exclude significant) = {counterclockwise_moderate_count}",
            "raw_values": {"moderate_count": counterclockwise_moderate_count}
        }
    })
    counterclockwise_significant_events = [e for e in counterclockwise_events if get_rotation_strength(e) == 3]
    counterclockwise_significant_count = len(counterclockwise_significant_events)
    opts, ans = _counting_options_for_count(counterclockwise_significant_count, rng)
    qa_pool.append({
        "question": "How many times does the person turn counterclockwise significantly?",
        "options": opts,
        "answer": ans,
        "evidence": {
            "events_used": counterclockwise_significant_events,
            "computation": f"count significant counterclockwise turning events (exclude very_short/slight, exclude moderate) = {counterclockwise_significant_count}",
            "raw_values": {"significant_count": counterclockwise_significant_count}
        }
    })
    
    # Rotation pitch (forward/backward lean): count and magnitude, same as yaw
    pitch_forward_events = _events_for_counting(stats["rotation_pitch"]["forward"])
    pitch_backward_events = _events_for_counting(stats["rotation_pitch"]["backward"])
    pitch_forward_count = len(pitch_forward_events)
    pitch_backward_count = len(pitch_backward_events)
    for label, events_list, count in [
        ("lean forward", pitch_forward_events, pitch_forward_count),
        ("lean backward", pitch_backward_events, pitch_backward_count),
    ]:
        opts, ans = _counting_options_for_count(count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label}?",
            "options": opts,
            "answer": ans,
            "evidence": {"events_used": events_list, "computation": f"count (exclude very_short/slight) = {count}", "raw_values": {"count": count}}
        })
        moderate_events = [e for e in events_list if get_rotation_strength(e) == 2]
        moderate_count = len(moderate_events)
        opts, ans = _counting_options_for_count(moderate_count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label} moderately?",
            "options": opts,
            "answer": ans,
            "evidence": {
                "events_used": moderate_events,
                "computation": f"count moderate leaning events (exclude very_short/slight, exclude significant) = {moderate_count}",
                "raw_values": {"moderate_count": moderate_count}
            }
        })
        significant_events = [e for e in events_list if get_rotation_strength(e) == 3]
        significant_count = len(significant_events)
        opts, ans = _counting_options_for_count(significant_count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label} significantly?",
            "options": opts,
            "answer": ans,
            "evidence": {
                "events_used": significant_events,
                "computation": f"count significant leaning events (exclude very_short/slight, exclude moderate) = {significant_count}",
                "raw_values": {"significant_count": significant_count}
            }
        })

    # Rotation roll (left/right lean): count and magnitude
    roll_left_events = _events_for_counting(stats["rotation_roll"]["left"])
    roll_right_events = _events_for_counting(stats["rotation_roll"]["right"])
    roll_left_count = len(roll_left_events)
    roll_right_count = len(roll_right_events)
    for label, events_list, count in [
        ("lean to the left", roll_left_events, roll_left_count),
        ("lean to the right", roll_right_events, roll_right_count),
    ]:
        opts, ans = _counting_options_for_count(count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label}?",
            "options": opts,
            "answer": ans,
            "evidence": {"events_used": events_list, "computation": f"count (exclude very_short/slight) = {count}", "raw_values": {"count": count}}
        })
        moderate_events = [e for e in events_list if get_rotation_strength(e) == 2]
        moderate_count = len(moderate_events)
        opts, ans = _counting_options_for_count(moderate_count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label} moderately?",
            "options": opts,
            "answer": ans,
            "evidence": {
                "events_used": moderate_events,
                "computation": f"count moderate leaning events (exclude very_short/slight, exclude significant) = {moderate_count}",
                "raw_values": {"moderate_count": moderate_count}
            }
        })
        significant_events = [e for e in events_list if get_rotation_strength(e) == 3]
        significant_count = len(significant_events)
        opts, ans = _counting_options_for_count(significant_count, rng)
        qa_pool.append({
            "question": f"How many times does the person {label} significantly?",
            "options": opts,
            "answer": ans,
            "evidence": {
                "events_used": significant_events,
                "computation": f"count significant leaning events (exclude very_short/slight, exclude moderate) = {significant_count}",
                "raw_values": {"significant_count": significant_count}
            }
        })

    return qa_pool


# -----------------------------------------------------------------------------
# Comparative question logic (single source of truth)
# -----------------------------------------------------------------------------
# Comparative questions compare opposite directions along the SAME motion axis.
# No cross-axis comparisons (e.g., forward vs right lean) and no threshold or
# existence-style questions. Comparisons are based on summed event magnitudes.
#
# Questions may be GLOBAL (entire video) or LOCAL (per quarter).
#
# 1) Event set
#    - Same as numerical: _events_for_counting() excludes very_short.
#    - Evidence uses events_used_with_quarters(events, fps, temporal_boundaries)
#      so each event has a "quarter" (1–4).
#
# 2) Motion axes used
#    Comparative is defined only on SAME-AXIS opposite directions:
#
#    Translation
#      - displacement_x: left vs right
#      - displacement_z: away vs towards
#
#    Rotation
#      - rotation_yaw: clockwise vs counterclockwise
#      - rotation_pitch: forward vs backward lean
#      - rotation_roll: left vs right lean
#
#    Cross-axis comparisons are NOT allowed:
#      e.g. forward vs right lean, translation vs rotation, etc.
#
# 3) Pool order (10 items)
#
#    GLOBAL comparisons
#
#    [0]  Left vs right displacement (4-way)
#         A=left, B=right, C=roughly equal, D=neither direction.
#
#    [1]  Away vs towards displacement (4-way)
#         A=away from start (+z), B=towards start (-z),
#         C=roughly equal, D=neither direction.
#
#    [2]  Clockwise vs counterclockwise turns (4-way)
#         A=clockwise, B=counterclockwise, C=roughly equal, D=no turns.
#
#    [3]  Forward vs backward lean (pitch) (4-way)
#         A=forward, B=backward, C=roughly equal, D=no leaning.
#
#    [4]  Left vs right lean (roll) (4-way)
#         A=left, B=right, C=roughly equal, D=no leaning.
#
#    LOCAL (quarter-based comparisons)
#
#    [5]  Q1: left vs right displacement
#    [6]  Q2: away vs towards displacement
#    [7]  Q1: clockwise vs counterclockwise turns
#    [8]  Q3: forward vs backward lean
#    [9]  Q4: left vs right lean
#
#    Local questions compare only events within the specified quarter.
#
# 4) 4-way direction comparison rules
#
#    - For the given axis and time scope:
#        side1 = sum(abs(intensity)) for direction A
#        side2 = sum(abs(intensity)) for direction B
#
#    - diff = abs(side1 - side2)
#
#    - threshold = 0.1 * max(side1, side2)
#
#    - If side1 == 0 and side2 == 0:
#          answer = D (no motion / no leaning / no turns)
#
#    - Else if diff < threshold:
#          answer = C (roughly equal)
#
#    - Else:
#          answer = A or B depending on which side is larger.
#
# 5) Quarter logic
#
#    - Same temporal split as temporal QA:
#          temporal_boundaries = [0, t1, t2, t3, duration]
#
#    - Quarters:
#          Q1 = [0, t1)
#          Q2 = [t1, t2)
#          Q3 = [t2, t3)
#          Q4 = [t3, duration)
#
#    - For local comparisons:
#          events = get_events_in_time_range(events, fps, tb[i], tb[i+1])
#
#    - Only events in that quarter contribute to the comparison sums.
#
# 6) Output and sampling
#
#    - Each pool item returns:
#          question
#          options
#          answer
#          evidence (events_used, computation, raw_values)
#
#    - main() samples N questions from the pool:
#          sample_questions(comparative_pool, questions_per_axis, rng)
#
#    - Evidence includes:
#          events_used_with_quarters(...)
#          summed intensities per direction
#          computed diff and threshold
#
# -----------------------------------------------------------------------------


def _comparative_magnitude_winner(
    side1: float,
    side2: float,
    label_1: str,
    label_2: str,
    tie_label: str = "roughly equal",
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (winner_label, detail) for same-axis magnitude comparison.
    Tie rule: diff <= 0.2 * max(side1, side2). Returns None when both sides are zero."""
    if side1 == 0 and side2 == 0:
        return None, {}
    threshold = 0.2 * max(side1, side2) if max(side1, side2) > 0 else 0
    diff = abs(side1 - side2)
    detail = {
        "metric_type": "total_magnitude",
        "diff": diff,
        "tie_threshold": threshold,
        "tie_rule": "diff <= 0.2 * max(side1, side2)",
    }
    if diff <= threshold:
        return tie_label, detail
    return (label_1 if side1 > side2 else label_2), detail


def generate_comparative_qa_pool(
    events: List[Dict[str, Any]],
    stats: Dict[str, Any],
    fps: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate comparative questions: event count, total magnitude, and speed — same-axis only.
    Options are 3-way, shuffled uniformly. Direction order in question text is randomised per call.
    Global: 15 questions (5 pairs x 3 metrics).
    Quarter-local: up to 15 questions from 5 randomly sampled (axis_pair, quarter) combos x 3 metrics."""
    qa_pool = []
    tb = stats["temporal_boundaries"]
    fps_val = stats["fps"]
    quarter_labels = ["first", "second", "third", "fourth"]

    def ev_q(events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return events_used_with_quarters(events_list, fps_val, tb)

    def append_comparative_question(
        question: str,
        events_1: List[Dict[str, Any]],
        events_2: List[Dict[str, Any]],
        label_1: str,
        label_2: str,
        metric_type: str,
        raw_key_1: str,
        raw_key_2: str,
        time_range: Optional[Tuple[float, float]] = None,
        quarter: Optional[int] = None,
        first_ev1: Optional[Dict[str, Any]] = None,
        first_ev2: Optional[Dict[str, Any]] = None,
    ) -> None:
        if metric_type == "event_count":
            value_1: Any = len(events_1)
            value_2: Any = len(events_2)
            if value_1 == 0 and value_2 == 0:
                return
            tie_label = "same number of events"
            winner = tie_label if value_1 == value_2 else (label_1 if value_1 > value_2 else label_2)
            detail: Dict[str, Any] = {"metric_type": "event_count", "diff": abs(value_1 - value_2),
                                       "tie_threshold": 0, "tie_rule": "count_1 == count_2"}
            computation = f"{raw_key_1}_count={value_1}, {raw_key_2}_count={value_2}, diff={detail['diff']}"
        elif metric_type == "total_magnitude":
            value_1 = sum(abs(e["intensity"]) for e in events_1)
            value_2 = sum(abs(e["intensity"]) for e in events_2)
            tie_label = "roughly equal"
            winner, detail = _comparative_magnitude_winner(value_1, value_2, label_1, label_2, tie_label)
            if winner is None:
                return
            computation = (f"{raw_key_1}_magnitude={value_1:.2f}, {raw_key_2}_magnitude={value_2:.2f}, "
                           f"diff={detail['diff']:.2f}, tie_threshold={detail['tie_threshold']:.2f}")
        else:  # speed
            if first_ev1 is None or first_ev2 is None:
                return
            value_1 = get_temporal_speed_level(first_ev1.get("temporal_category") or "moderate")
            value_2 = get_temporal_speed_level(first_ev2.get("temporal_category") or "moderate")
            tie_label = "same category"
            winner = tie_label if value_1 == value_2 else (label_1 if value_1 > value_2 else label_2)
            detail = {"metric_type": "speed", "diff": abs(value_1 - value_2),
                      "tie_threshold": 0, "tie_rule": "level_1 == level_2"}
            computation = f"{raw_key_1}_speed_level={value_1}, {raw_key_2}_speed_level={value_2}"

        opts = [label_1, label_2, tie_label]
        rng.shuffle(opts)
        options = {chr(65 + i): opts[i] for i in range(3)}
        answer = next(k for k, v in options.items() if v == winner)

        suffix = "count" if metric_type == "event_count" else "magnitude" if metric_type == "total_magnitude" else "speed_level"
        raw_values: Dict[str, Any] = {
            f"{raw_key_1}_{suffix}": value_1, f"{raw_key_2}_{suffix}": value_2,
            "metric_type": metric_type, "diff": detail["diff"],
        }
        if metric_type == "total_magnitude":
            raw_values["tie_threshold"] = detail["tie_threshold"]
        if quarter is not None:
            raw_values["quarter"] = quarter
        if time_range is not None:
            raw_values["time_range"] = [time_range[0], time_range[1]]

        ev_used = (events_1 + events_2) if metric_type != "speed" else \
                  ([first_ev1] if first_ev1 else []) + ([first_ev2] if first_ev2 else [])
        qa_pool.append({
            "question": question,
            "options": options,
            "answer": answer,
            "evidence": {"events_used": ev_q(ev_used), "computation": computation, "raw_values": raw_values},
        })

    # Event lists per axis (_events_for_counting excludes very_short/slight)
    left_events = _events_for_counting(stats["displacement_x"]["left"])
    right_events = _events_for_counting(stats["displacement_x"]["right"])
    forward_events = _events_for_counting(stats["displacement_z"]["forward"])
    backward_events = _events_for_counting(stats["displacement_z"]["backward"])
    cw_events = _events_for_counting(stats["rotation_yaw"]["clockwise"])
    ccw_events = _events_for_counting(stats["rotation_yaw"]["counterclockwise"])
    pitch_fwd_events = _events_for_counting(stats["rotation_pitch"]["forward"])
    pitch_bwd_events = _events_for_counting(stats["rotation_pitch"]["backward"])
    roll_left_events = _events_for_counting(stats["rotation_roll"]["left"])
    roll_right_events = _events_for_counting(stats["rotation_roll"]["right"])

    # Pre-compute first speed-eligible event per (code, direction) for global speed questions
    first_speed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in events:
        if e["code"] not in ORDERING_MOTION_CODES or get_direction(e) is None or not _is_speed_eligible(e):
            continue
        key = (e["code"], get_direction(e))
        if key not in first_speed or e["t0"] < first_speed[key]["t0"]:
            first_speed[key] = e

    # Axis-pair metadata: ev1/ev2 are count/magnitude-eligible event lists;
    # count_a/b and mag_a/b are direction phrases for question text;
    # speed_a/b are action labels for the speed question.
    # count_base / mag_base hold the question verb phrase WITHOUT scope.
    axis_pairs = [
        dict(ev1=left_events, ev2=right_events, lbl1="left", lbl2="right", rk1="left", rk2="right",
             code="displacement_x", dir1="left", dir2="right",
             count_a="moving left", count_b="moving right",
             mag_a="left", mag_b="right",
             speed_a="move left", speed_b="move right",
             count_base="event happens more often", mag_base="direction has greater total magnitude"),
        dict(ev1=forward_events, ev2=backward_events,
             lbl1="away from starting position", lbl2="towards starting position", rk1="away", rk2="towards",
             code="displacement_z", dir1="forward", dir2="backward",
             count_a="moving away from starting position", count_b="moving towards starting position",
             mag_a="away from starting position", mag_b="towards starting position",
             speed_a="move away from starting position", speed_b="move towards starting position",
             count_base="event happens more often", mag_base="direction has greater total magnitude"),
        dict(ev1=cw_events, ev2=ccw_events, lbl1="clockwise", lbl2="counterclockwise",
             rk1="clockwise", rk2="counterclockwise",
             code="rotation_yaw", dir1="clockwise", dir2="counterclockwise",
             count_a="clockwise", count_b="counterclockwise",
             mag_a="clockwise", mag_b="counterclockwise",
             speed_a="turn clockwise", speed_b="turn counterclockwise",
             count_base="turning event happens more often", mag_base="turning direction has greater total magnitude"),
        dict(ev1=pitch_fwd_events, ev2=pitch_bwd_events, lbl1="forward", lbl2="backward",
             rk1="forward", rk2="backward",
             code="rotation_pitch", dir1="forward", dir2="backward",
             count_a="forward", count_b="backward",
             mag_a="forward", mag_b="backward",
             speed_a="lean forward", speed_b="lean backward",
             count_base="leaning event happens more often",
             mag_base="leaning direction has greater total magnitude"),
        dict(ev1=roll_left_events, ev2=roll_right_events, lbl1="left", lbl2="right",
             rk1="left", rk2="right",
             code="rotation_roll", dir1="left", dir2="right",
             count_a="left", count_b="right",
             mag_a="left", mag_b="right",
             speed_a="lean to the left", speed_b="lean to the right",
             count_base="leaning event happens more often",
             mag_base="leaning direction has greater total magnitude"),
    ]

    def emit_pair(p: Dict[str, Any], ev1: List, ev2: List,
                  fev1: Optional[Dict], fev2: Optional[Dict],
                  scope: str, time_range: Optional[Tuple[float, float]], quarter: Optional[int],
                  bucket: Optional[str] = None) -> None:
        """Emit count, magnitude, and speed questions for one pair with randomised direction order.

        `scope` is ' overall' (global) or ' in the X quarter of the video' (local).
        `bucket` is None for global, or e.g. 'first' for quarter-local — used to prefix speed question.
        """
        swap = rng.random() < 0.5
        if swap:
            ev1, ev2 = ev2, ev1
            fev1, fev2 = fev2, fev1
            lbl1, lbl2 = p["lbl2"], p["lbl1"]
            rk1, rk2 = p["rk2"], p["rk1"]
            ca, cb = p["count_b"], p["count_a"]
            ma, mb = p["mag_b"], p["mag_a"]
            sa, sb = p["speed_b"], p["speed_a"]
        else:
            lbl1, lbl2 = p["lbl1"], p["lbl2"]
            rk1, rk2 = p["rk1"], p["rk2"]
            ca, cb = p["count_a"], p["count_b"]
            ma, mb = p["mag_a"], p["mag_b"]
            sa, sb = p["speed_a"], p["speed_b"]

        if bucket is None:
            speed_q = f"Is the person's first {sa} faster or slower than their first {sb}?"
        else:
            speed_q = (f"In the {bucket} quarter of the video, is the person's first {sa} "
                       f"faster or slower than their first {sb}?")

        append_comparative_question(
            f"Which {p['count_base']}{scope}: {ca} or {cb}?",
            ev1, ev2, lbl1, lbl2, "event_count", rk1, rk2, time_range, quarter)
        append_comparative_question(
            f"Which {p['mag_base']}{scope}: {ma} or {mb}?",
            ev1, ev2, lbl1, lbl2, "total_magnitude", rk1, rk2, time_range, quarter)
        append_comparative_question(
            speed_q, ev1, ev2, lbl1, lbl2, "speed", rk1, rk2, time_range, quarter,
            first_ev1=fev1, first_ev2=fev2)

    # Global comparisons: 15 questions (5 pairs x 3 metrics)
    for p in axis_pairs:
        fev1 = first_speed.get((p["code"], p["dir1"]))
        fev2 = first_speed.get((p["code"], p["dir2"]))
        emit_pair(p, p["ev1"], p["ev2"], fev1, fev2, " overall", None, None)

    # Quarter-local comparisons: randomly sample 5 (axis_pair, quarter) combos from all 20 → up to 15 questions
    all_combos = [(pair_idx, q_idx) for pair_idx in range(5) for q_idx in range(4)]
    sampled_combos = rng.sample(all_combos, min(5, len(all_combos)))
    sampled_combos.sort()
    for pair_idx, q_idx in sampled_combos:
        p = axis_pairs[pair_idx]
        t_start, t_end = tb[q_idx], tb[q_idx + 1]
        bucket = quarter_labels[q_idx]
        ev1_q = get_events_in_time_range(p["ev1"], fps_val, t_start, t_end)
        ev2_q = get_events_in_time_range(p["ev2"], fps_val, t_start, t_end)
        # Speed: first eligible event within this quarter per direction
        spd1_q = get_events_in_time_range(
            [e for e in p["ev1"] if _is_speed_eligible(e)], fps_val, t_start, t_end)
        spd2_q = get_events_in_time_range(
            [e for e in p["ev2"] if _is_speed_eligible(e)], fps_val, t_start, t_end)
        fev1_q = min(spd1_q, key=lambda e: e["t0"]) if spd1_q else None
        fev2_q = min(spd2_q, key=lambda e: e["t0"]) if spd2_q else None
        emit_pair(p, ev1_q, ev2_q, fev1_q, fev2_q,
                  f" in the {bucket} quarter of the video", (t_start, t_end), q_idx + 1,
                  bucket=bucket)

    # Tie rate cap: ties must not exceed 30% of correct answers across all subtypes combined.
    # Excess ties are dropped preferentially from the speed subtype first.
    _COMPARATIVE_MAX_TIE_RATIO = 0.30
    _TIE_LABELS = {"same number of events", "roughly equal", "same category"}

    def _is_tie(q: Dict[str, Any]) -> bool:
        return q["options"][q["answer"]] in _TIE_LABELS

    ties_speed = [q for q in qa_pool if _is_tie(q) and
                  q["evidence"]["raw_values"].get("metric_type") == "speed"]
    ties_other = [q for q in qa_pool if _is_tie(q) and
                  q["evidence"]["raw_values"].get("metric_type") != "speed"]
    non_ties = [q for q in qa_pool if not _is_tie(q)]

    n_ties = len(ties_speed) + len(ties_other)
    n_total = len(qa_pool)
    if n_total > 0 and n_ties / n_total > _COMPARATIVE_MAX_TIE_RATIO:
        # Maximum allowed ties given the number of non-ties:
        # T / (T + N) <= R  →  T <= R * N / (1 - R)
        target_ties = int(_COMPARATIVE_MAX_TIE_RATIO * len(non_ties) / (1 - _COMPARATIVE_MAX_TIE_RATIO))
        excess = n_ties - max(target_ties, 0)
        rng.shuffle(ties_speed)
        rng.shuffle(ties_other)
        drop_speed = min(excess, len(ties_speed))
        ties_speed = ties_speed[drop_speed:]
        excess -= drop_speed
        if excess > 0:
            ties_other = ties_other[min(excess, len(ties_other)):]
        qa_pool = non_ties + ties_speed + ties_other
        rng.shuffle(qa_pool)

    return qa_pool


def generate_dominant_qa_pool(events: List[Dict[str, Any]], stats: Dict[str, Any], fps: float, duration: float, rng: random.Random) -> List[Dict[str, Any]]:
    """Generate pool of dominant questions. Two families:
    Family 1 — Direction dominance: 3-way (dir_a / dir_b / both equally), tie rule diff < 0.2*max.
    Family 2 — Rotation subtype speed comparison: avg temporal_category level across all events for
    each rotation subtype (turning, forward/backward leaning, left/right leaning); 3 pairwise questions
    (global + quarter-local); 3-way (subtype_A / subtype_B / roughly the same)."""
    qa_pool = []
    temporal_boundaries = stats["temporal_boundaries"]
    quarter_labels = TEMPORAL_BUCKET_LABELS
    DOMINANT_TIE_RATIO = 0.2

    def events_for_window(events_list: List[Dict[str, Any]], time_range: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
        filtered = [e for e in events_list if _is_speed_eligible(e)]
        if time_range is None:
            return filtered
        return get_events_in_time_range(filtered, fps, time_range[0], time_range[1])

    def ev_q(events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return events_used_with_quarters(events_list, fps, temporal_boundaries)

    def append_dominant_question(
        question: str,
        label_a: str, value_a: float, events_a: List[Dict[str, Any]],
        label_b: str, value_b: float, events_b: List[Dict[str, Any]],
        label_tie: str,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if value_a <= 0 and value_b <= 0:
            return  # skip: no motion on either side
        diff = abs(value_a - value_b)
        max_val = max(value_a, value_b)
        tie_threshold = DOMINANT_TIE_RATIO * max_val
        tied_labels = [label_a, label_b] if diff < tie_threshold else []
        winner_label = label_tie if diff < tie_threshold else (label_a if value_a > value_b else label_b)

        opts = [label_a, label_b, label_tie]
        rng.shuffle(opts)
        options = {chr(65 + i): opts[i] for i in range(3)}
        answer = next(k for k, v in options.items() if v == winner_label)

        evidence = {
            "events_used": ev_q(events_a + events_b),
            "computation": (
                f"{label_a}={value_a:.2f}, {label_b}={value_b:.2f}; "
                f"diff={diff:.2f}, tie_rule=diff < {DOMINANT_TIE_RATIO}*max(side1,side2), "
                f"tie_threshold={tie_threshold:.2f}, tied_labels={tied_labels} -> {winner_label}"
            ),
            "raw_values": {
                label_a: value_a,
                label_b: value_b,
                "diff": diff,
                "tie_threshold": tie_threshold,
                "tie_rule": f"diff < {DOMINANT_TIE_RATIO} * max(side1, side2)",
                "tied_labels": tied_labels,
                "pair_labels": [label_a, label_b],
            },
        }
        if time_range is not None:
            evidence["time_range"] = [time_range[0], time_range[1]]
        qa_pool.append({"question": question, "options": options, "answer": answer, "evidence": evidence})

    def _q(prefix: str, stem: str) -> str:
        """Build question: prefix is '' (global) or 'In the X of the video, '."""
        return f"{prefix}{stem}" if prefix else stem

    def append_horizontal_lr_question(prefix: str = "", time_range: Optional[Tuple[float, float]] = None) -> None:
        left_events = events_for_window(stats["displacement_x"]["left"], time_range)
        right_events = events_for_window(stats["displacement_x"]["right"], time_range)
        dirs = "left or right" if rng.random() < 0.5 else "right or left"
        append_dominant_question(
            _q(prefix, f"What is the dominant horizontal movement direction, {dirs}?"),
            "left", sum(abs(e["intensity"]) for e in left_events), left_events,
            "right", sum(abs(e["intensity"]) for e in right_events), right_events,
            "both left and right equally", time_range,
        )

    def append_horizontal_depth_question(prefix: str = "", time_range: Optional[Tuple[float, float]] = None) -> None:
        towards_events = events_for_window(stats["displacement_z"]["backward"], time_range)
        away_events = events_for_window(stats["displacement_z"]["forward"], time_range)
        dirs = "towards or away from" if rng.random() < 0.5 else "away from or towards"
        append_dominant_question(
            _q(prefix, f"What is the dominant depth movement direction, {dirs} the starting position?"),
            "towards starting position", sum(abs(e["intensity"]) for e in towards_events), towards_events,
            "away from starting position", sum(abs(e["intensity"]) for e in away_events), away_events,
            "both towards and away equally", time_range,
        )

    def append_turning_question(prefix: str = "", time_range: Optional[Tuple[float, float]] = None) -> None:
        cw_events = events_for_window(stats["rotation_yaw"]["clockwise"], time_range)
        ccw_events = events_for_window(stats["rotation_yaw"]["counterclockwise"], time_range)
        dirs = "clockwise or counterclockwise" if rng.random() < 0.5 else "counterclockwise or clockwise"
        append_dominant_question(
            _q(prefix, f"What is the primary turning direction, {dirs}?"),
            "clockwise", sum(abs(e["intensity"]) for e in cw_events), cw_events,
            "counterclockwise", sum(abs(e["intensity"]) for e in ccw_events), ccw_events,
            "both equally", time_range,
        )

    def append_leaning_pitch_question(prefix: str = "", time_range: Optional[Tuple[float, float]] = None) -> None:
        fwd_events = events_for_window(stats["rotation_pitch"]["forward"], time_range)
        bwd_events = events_for_window(stats["rotation_pitch"]["backward"], time_range)
        dirs = "forward or backward" if rng.random() < 0.5 else "backward or forward"
        append_dominant_question(
            _q(prefix, f"What is the dominant leaning direction, {dirs}?"),
            "forward", sum(abs(e["intensity"]) for e in fwd_events), fwd_events,
            "backward", sum(abs(e["intensity"]) for e in bwd_events), bwd_events,
            "both forward and backward equally", time_range,
        )

    def append_leaning_roll_question(prefix: str = "", time_range: Optional[Tuple[float, float]] = None) -> None:
        left_events = events_for_window(stats["rotation_roll"]["left"], time_range)
        right_events = events_for_window(stats["rotation_roll"]["right"], time_range)
        dirs = "left or right" if rng.random() < 0.5 else "right or left"
        append_dominant_question(
            _q(prefix, f"What is the dominant leaning direction, {dirs}?"),
            "left", sum(abs(e["intensity"]) for e in left_events), left_events,
            "right", sum(abs(e["intensity"]) for e in right_events), right_events,
            "both left and right equally", time_range,
        )

    append_horizontal_lr_question()
    append_horizontal_depth_question()
    append_turning_question()
    append_leaning_pitch_question()
    append_leaning_roll_question()

    for i, bucket_label in enumerate(quarter_labels):
        time_range = (temporal_boundaries[i], temporal_boundaries[i + 1])
        prefix = f"In the {bucket_label} of the video, "
        append_horizontal_lr_question(prefix, time_range)
        append_horizontal_depth_question(prefix, time_range)
        append_turning_question(prefix, time_range)
        append_leaning_pitch_question(prefix, time_range)
        append_leaning_roll_question(prefix, time_range)

    # Cap tie rate for direction dominance at 10%: randomly drop excess tie questions.
    _DOMINANT_DIR_TIE_LABELS = {
        "both left and right equally", "both towards and away equally",
        "both equally", "both forward and backward equally",
    }
    _DOMINANT_DIR_TIE_CAP = 0.10
    dir_ties = [q for q in qa_pool if q["options"][q["answer"]] in _DOMINANT_DIR_TIE_LABELS]
    dir_non_ties = [q for q in qa_pool if q["options"][q["answer"]] not in _DOMINANT_DIR_TIE_LABELS]
    if len(qa_pool) > 0 and len(dir_ties) / len(qa_pool) > _DOMINANT_DIR_TIE_CAP:
        target = int(_DOMINANT_DIR_TIE_CAP * len(dir_non_ties) / (1 - _DOMINANT_DIR_TIE_CAP))
        rng.shuffle(dir_ties)
        qa_pool = dir_non_ties + dir_ties[:max(target, 0)]

    # ---- Family 2: Rotation subtype speed comparison ----
    # Compare average movement speed across the three rotation subtypes (turning, fwd/bwd lean, lr lean).
    # Speed per subtype = avg temporal_category level over all speed-eligible events for that subtype.
    ROTATION_SPEED_TIE_RATIO = 0.2

    yaw_events = [e for e in events if e["code"] == "rotation_yaw" and _is_speed_eligible(e)]
    pitch_events = [e for e in events if e["code"] == "rotation_pitch" and _is_speed_eligible(e)]
    roll_events = [e for e in events if e["code"] == "rotation_roll" and _is_speed_eligible(e)]

    rotation_subtypes: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("turning", yaw_events),
        ("forward/backward leaning", pitch_events),
        ("left/right leaning", roll_events),
    ]

    def avg_speed_for_events(event_list: List[Dict[str, Any]]) -> Optional[float]:
        if not event_list:
            return None
        levels = [get_temporal_speed_level(e.get("temporal_category") or "moderate") for e in event_list]
        return sum(levels) / len(levels)

    def append_rotation_speed_question(
        speed_a: float, label_a: str, events_a: List[Dict[str, Any]],
        speed_b: float, label_b: str, events_b: List[Dict[str, Any]],
        bucket: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        diff = abs(speed_a - speed_b)
        max_speed = max(speed_a, speed_b)
        tie_threshold = ROTATION_SPEED_TIE_RATIO * max_speed if max_speed > 0 else 0
        if diff < tie_threshold:
            return  # skip: no genuine speed difference
        winner = label_a if speed_a > speed_b else label_b

        # Randomise subtype order in question text to prevent first-mentioned bias
        qa, qb = (label_a, label_b) if rng.random() < 0.5 else (label_b, label_a)
        if bucket is None:
            question = f"Which is faster: the person's {qa} or their {qb}?"
        else:
            question = f"In the {bucket} of the video, which is faster: the person's {qa} or their {qb}?"

        opts = [label_a, label_b, "roughly the same"]
        rng.shuffle(opts)
        options = {chr(65 + i): opts[i] for i in range(3)}
        answer = next(k for k, v in options.items() if v == winner)

        evidence: Dict[str, Any] = {
            "events_used": ev_q(events_a + events_b),
            "computation": (
                f"{label_a}_avg_speed={speed_a:.2f}, {label_b}_avg_speed={speed_b:.2f}; "
                f"diff={diff:.2f}, tie_threshold={tie_threshold:.2f} -> {winner}"
            ),
            "raw_values": {
                f"{label_a}_avg_speed": speed_a,
                f"{label_b}_avg_speed": speed_b,
                "diff": diff,
                "tie_threshold": tie_threshold,
                "tie_rule": f"diff < {ROTATION_SPEED_TIE_RATIO} * max(speed_A, speed_B)",
            },
        }
        if time_range is not None:
            evidence["time_range"] = list(time_range)
        qa_pool.append({"question": question, "options": options, "answer": answer, "evidence": evidence})

    # 3 pairings: turning/pitch, turning/roll, pitch/roll — each global + 4 quarter-local
    rotation_pairs = [(0, 1), (0, 2), (1, 2)]
    for pi, pj in rotation_pairs:
        la, evts_a = rotation_subtypes[pi]
        lb, evts_b = rotation_subtypes[pj]
        sa = avg_speed_for_events(evts_a)
        sb = avg_speed_for_events(evts_b)
        if sa is not None and sb is not None:
            append_rotation_speed_question(sa, la, evts_a, sb, lb, evts_b)
        for q_idx in range(4):
            t_start, t_end = temporal_boundaries[q_idx], temporal_boundaries[q_idx + 1]
            ea_q = get_events_in_time_range(evts_a, fps, t_start, t_end)
            eb_q = get_events_in_time_range(evts_b, fps, t_start, t_end)
            sa_q = avg_speed_for_events(ea_q)
            sb_q = avg_speed_for_events(eb_q)
            if sa_q is not None and sb_q is not None:
                append_rotation_speed_question(
                    sa_q, la, ea_q, sb_q, lb, eb_q,
                    bucket=quarter_labels[q_idx], time_range=(t_start, t_end),
                )

    return qa_pool


def _first_occurrence_bucket_index(first_time: float, duration: float) -> int:
    """Return bucket index 0..NUM_FIRST_OCCURRENCE_BUCKETS-1 for time; -1 means never (caller uses separate option)."""
    if duration <= 0:
        return 0
    frac = first_time / duration
    for i in range(NUM_FIRST_OCCURRENCE_BUCKETS):
        if frac < (i + 1) / NUM_FIRST_OCCURRENCE_BUCKETS:
            return i
    return NUM_FIRST_OCCURRENCE_BUCKETS - 1


def generate_temporal_qa_pool(events: List[Dict[str, Any]], stats: Dict[str, Any], fps: float, duration: float, rng: random.Random) -> List[Dict[str, Any]]:
    """Generate pool of temporal localization questions. Excludes very_short/slight spatial_category.
    Family 1 — Bucket-local: 20 candidates (5 axes x 4 quarters), shuffle options, subsample to
    ≤40% per answer category (prefer removing 'both').
    Family 2 — First-event-in-quarter: earliest event per quarter; 2 displacement + 2 rotation labels.
    Family 3 — Event speed classification: first eligible event per (code, direction);
    correct temporal_category + 3 distractors from SPEED_5_CATEGORIES (4-way)."""
    qa_pool = []
    temporal_boundaries = stats["temporal_boundaries"]
    num_windows = len(temporal_boundaries) - 1

    # Filter out very_short/slight so temporal questions use same event sets as numerical/comparative
    left_x = _events_for_counting(stats["displacement_x"]["left"])
    right_x = _events_for_counting(stats["displacement_x"]["right"])
    forward_z = _events_for_counting(stats["displacement_z"]["forward"])
    backward_z = _events_for_counting(stats["displacement_z"]["backward"])
    cw_yaw = _events_for_counting(stats["rotation_yaw"]["clockwise"])
    ccw_yaw = _events_for_counting(stats["rotation_yaw"]["counterclockwise"])
    pitch_fwd = _events_for_counting(stats["rotation_pitch"]["forward"])
    pitch_bwd = _events_for_counting(stats["rotation_pitch"]["backward"])
    roll_left = _events_for_counting(stats["rotation_roll"]["left"])
    roll_right = _events_for_counting(stats["rotation_roll"]["right"])

    def ev_q(events_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return events_used_with_quarters(events_list, fps, temporal_boundaries)

    # ---- Bucket-local direction questions ----
    def make_bucket_candidate(
        question: str,
        label_dir1: str, label_dir2: str, label_both: str, label_none: str,
        events_dir1: List, events_dir2: List,
        t_start: float, t_end: float, bucket_label: str, raw_keys: Tuple[str, str],
    ) -> Dict[str, Any]:
        """Build a candidate dict (options not yet shuffled; _balance_cat tracks answer category)."""
        n1, n2 = len(events_dir1), len(events_dir2)
        if n1 > 0 and n2 > 0:
            winner, cat = label_both, "both"
        elif n1 > 0:
            winner, cat = label_dir1, "dir1"
        elif n2 > 0:
            winner, cat = label_dir2, "dir2"
        else:
            winner, cat = label_none, "no_movement"
        return {
            "_balance_cat": cat,
            "_winner": winner,
            "_opts": [label_dir1, label_dir2, label_both, label_none],
            "question": question,
            "evidence": {
                "time_range": [t_start, t_end],
                "bucket": bucket_label,
                "events_used": ev_q(events_dir1 + events_dir2),
                "computation": f"{raw_keys[0]}={n1}, {raw_keys[1]}={n2}",
                "raw_values": {raw_keys[0] + "_count": n1, raw_keys[1] + "_count": n2},
            },
        }

    # Generate all 20 candidates (5 axes x 4 quarters)
    bucket_candidates: List[Dict[str, Any]] = []
    for w in range(num_windows):
        t_start = temporal_boundaries[w]
        t_end = temporal_boundaries[w + 1]
        if t_end - t_start < 0.2:
            continue
        bl = TEMPORAL_BUCKET_LABELS[w]
        bucket_candidates.append(make_bucket_candidate(
            f"In the {bl} of the video, what horizontal movement does the person make?",
            "left", "right", "both left and right", "no horizontal movement",
            get_events_in_time_range(left_x, fps, t_start, t_end),
            get_events_in_time_range(right_x, fps, t_start, t_end),
            t_start, t_end, bl, ("left", "right"),
        ))
        bucket_candidates.append(make_bucket_candidate(
            f"In the {bl} of the video, does the person move towards or away from their starting position?",
            "away from starting position", "towards starting position", "both", "no toward/away movement",
            get_events_in_time_range(forward_z, fps, t_start, t_end),
            get_events_in_time_range(backward_z, fps, t_start, t_end),
            t_start, t_end, bl, ("away", "towards"),
        ))
        bucket_candidates.append(make_bucket_candidate(
            f"In the {bl} of the video, what is the person's turning direction?",
            "clockwise", "counterclockwise", "both directions", "no turning",
            get_events_in_time_range(cw_yaw, fps, t_start, t_end),
            get_events_in_time_range(ccw_yaw, fps, t_start, t_end),
            t_start, t_end, bl, ("clockwise", "counterclockwise"),
        ))
        bucket_candidates.append(make_bucket_candidate(
            f"In the {bl} of the video, does the person lean forward or backward?",
            "forward", "backward", "both", "no forward/backward leaning",
            get_events_in_time_range(pitch_fwd, fps, t_start, t_end),
            get_events_in_time_range(pitch_bwd, fps, t_start, t_end),
            t_start, t_end, bl, ("pitch_forward", "pitch_backward"),
        ))
        bucket_candidates.append(make_bucket_candidate(
            f"In the {bl} of the video, does the person lean left or right?",
            "left", "right", "both", "no left/right leaning",
            get_events_in_time_range(roll_left, fps, t_start, t_end),
            get_events_in_time_range(roll_right, fps, t_start, t_end),
            t_start, t_end, bl, ("roll_left", "roll_right"),
        ))

    # Balance: remove items so no answer category > 40%; prefer removing "both" first
    def balance_bucket_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = list(candidates)
        for _ in range(len(candidates)):
            n = len(result)
            if n == 0:
                break
            cat_counts: Dict[str, int] = {}
            for q in result:
                cat_counts[q["_balance_cat"]] = cat_counts.get(q["_balance_cat"], 0) + 1
            overcrowded = {c: cnt for c, cnt in cat_counts.items() if cnt > 0.4 * n}
            if not overcrowded:
                break
            remove_cat = "both" if "both" in overcrowded else max(overcrowded, key=lambda c: overcrowded[c])
            pool = [q for q in result if q["_balance_cat"] == remove_cat]
            result.remove(rng.choice(pool))
        return result

    for cand in balance_bucket_candidates(bucket_candidates):
        opts = list(cand["_opts"])
        rng.shuffle(opts)
        options = {chr(65 + i): opts[i] for i in range(4)}
        answer = next(k for k, v in options.items() if v == cand["_winner"])
        qa_pool.append({
            "question": cand["question"],
            "options": options,
            "answer": answer,
            "evidence": cand["evidence"],
        })

    # ---- First-event-in-quarter questions ----
    # Tag all moderate+ events with label and super-family (same filter as ordering)
    tagged_all: List[Dict[str, Any]] = []
    for e in events:
        if e["code"] not in ORDERING_MOTION_CODES:
            continue
        if get_direction(e) is None:
            continue
        if not is_moderate_or_more(e):
            continue
        tagged_all.append({
            "event": e,
            "t0": e["t0"],
            "time": frame_to_time(e["t0"], fps),
            "label": get_event_label(e),
            "family": _ordering_super_family({"event": e}),
        })

    # Distractor pools: all unique labels per family across the whole video
    disp_labels_all = list(dict.fromkeys(x["label"] for x in tagged_all if x["family"] == "displacement"))
    rot_labels_all = list(dict.fromkeys(x["label"] for x in tagged_all if x["family"] == "rotation"))

    for w in range(num_windows):
        t_start = temporal_boundaries[w]
        t_end = temporal_boundaries[w + 1]
        if t_end - t_start < 0.2:
            continue
        bl = TEMPORAL_BUCKET_LABELS[w]
        in_quarter = [x for x in tagged_all if event_overlaps_time_range(x["event"], fps, t_start, t_end)]
        if not in_quarter:
            continue
        first = min(in_quarter, key=lambda x: x["t0"])
        correct_label = first["label"]
        correct_family = first["family"]

        # Need exactly 2 displacement + 2 rotation total; correct takes 1 slot in its family
        need_disp = 1 if correct_family == "displacement" else 2
        need_rot = 2 if correct_family == "displacement" else 1

        disp_pool = [lbl for lbl in disp_labels_all if lbl != correct_label]
        rot_pool = [lbl for lbl in rot_labels_all if lbl != correct_label]

        if len(disp_pool) < need_disp or len(rot_pool) < need_rot:
            continue

        distractors = rng.sample(disp_pool, need_disp) + rng.sample(rot_pool, need_rot)
        all_labels = [correct_label] + distractors
        if len(set(all_labels)) != 4:
            continue

        rng.shuffle(all_labels)
        options = {chr(65 + i): all_labels[i] for i in range(4)}
        answer = next(k for k, v in options.items() if v == correct_label)
        qa_pool.append({
            "question": f"In the {bl} of the video, what is the first movement or rotation that happens?",
            "options": options,
            "answer": answer,
            "evidence": {
                "quarter": bl,
                "time_range": [t_start, t_end],
                "events_used": ev_q(in_quarter),
                "first_event": first["event"],
                "first_event_time": first["time"],
                "computation": (
                    f"earliest event in {bl} quarter by t0; correct_family={correct_family}; "
                    f"options = 2 displacement + 2 rotation labels"
                ),
            },
        })

    # ---- Family 3: Event speed classification ----
    first_speed_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in events:
        if e["code"] not in ORDERING_MOTION_CODES or get_direction(e) is None:
            continue
        if not _is_speed_eligible(e):
            continue
        key = (e["code"], get_direction(e))
        if key not in first_speed_by_key or e["t0"] < first_speed_by_key[key]["t0"]:
            first_speed_by_key[key] = e

    for e in first_speed_by_key.values():
        tc = (e.get("temporal_category") or "moderate").strip().lower()
        if tc not in SPEED_5_CATEGORIES:
            tc = "moderate"
        base = get_event_label_base(e)
        distractors = rng.sample([c for c in SPEED_5_CATEGORIES if c != tc], 3)
        opts_list = [tc] + distractors
        rng.shuffle(opts_list)
        options = {chr(65 + i): opts_list[i] for i in range(4)}
        answer = next(k for k, v in options.items() if v == tc)
        qa_pool.append({
            "question": f"During the person's first {base}, how fast is the movement?",
            "options": options,
            "answer": answer,
            "evidence": {
                "events_used": ev_q([e]),
                "event": e,
                "temporal_category": tc,
                "speed_level": get_temporal_speed_level(tc),
                "computation": f"temporal_category={tc}; correct + 3 distractors from 5 speed categories; shuffled",
            },
        })

    return qa_pool


def generate_ordering_qa_pool(
    events: List[Dict[str, Any]], stats: Dict[str, Any], fps: float, duration: float, rng: random.Random
) -> List[Dict[str, Any]]:
    """Generate sequence-ordering questions: in a quarter, after event A, which happens next?
    B is the earliest valid event (any super-family) with B.t0 >= A.t1 + 10 frames.
    Every question has exactly 2 displacement + 2 rotation labels across all 4 options,
    preventing the model from eliminating options by super-family membership.
    Dataset-level balance targets: opposite-direction correct ≤30%, same-as-anchor correct ≥10%.
    """
    qa_pool = []
    temporal_boundaries = stats["temporal_boundaries"]
    bucket_labels = TEMPORAL_BUCKET_LABELS

    tagged = []
    for e in events:
        if e["code"] not in ORDERING_MOTION_CODES:
            continue
        if get_direction(e) is None:
            continue
        if not is_moderate_or_more(e):
            continue
        tagged.append({
            "event": e,
            "t0": e["t0"],
            "t1": e.get("t1", e["t0"]),
            "time": frame_to_time(e["t0"], fps),
            "label": get_event_label(e),
            "family": _ordering_super_family({"event": e}),
        })
    if len(tagged) < 2:
        return qa_pool

    def _best_one_per_label(
        pool: List[Dict[str, Any]],
        exclude_labels: set,
        A_t0: int,
        B_t0: int,
    ) -> List[Dict[str, Any]]:
        """One representative per distinct canonical label (excluding given labels), time-local preferred."""
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for x in pool:
            if x["label"] not in exclude_labels:
                by_label.setdefault(x["label"], []).append(x)
        result = []
        for evts in by_label.values():
            best = min(evts, key=lambda e: min(abs(e["t0"] - A_t0), abs(e["t0"] - B_t0)))
            result.append(best)
        result.sort(key=lambda e: min(abs(e["t0"] - A_t0), abs(e["t0"] - B_t0)))
        return result

    num_windows = len(temporal_boundaries) - 1
    for w in range(num_windows):
        t_start = temporal_boundaries[w]
        t_end = temporal_boundaries[w + 1]
        if t_end - t_start < 0.2:
            continue
        in_quarter = [x for x in tagged if event_overlaps_time_range(x["event"], fps, t_start, t_end)]
        in_quarter.sort(key=lambda x: x["t0"])
        if len(in_quarter) < 4:
            continue

        for A in in_quarter:
            threshold = A["t1"] + ORDERING_MIN_GAP_AFTER_A_END_FRAMES
            # B can be any super-family
            B_candidates = [x for x in in_quarter if x["t0"] >= threshold and x is not A]
            if not B_candidates:
                continue
            B = min(B_candidates, key=lambda x: x["t0"])

            # Stem: disambiguate if A's label appears more than once in the quarter
            same_label_sorted = sorted(
                [x for x in in_quarter if x["label"] == A["label"]], key=lambda x: x["t0"]
            )
            occurrence_1based = same_label_sorted.index(A) + 1
            if len(same_label_sorted) == 1:
                mid = f"after the person {A['label']}, "
            else:
                mid = f"after the {_ordering_ordinal(occurrence_1based)} time the person {A['label']}, "

            # Build exactly 2 displacement + 2 rotation labels across all 4 options.
            # B occupies one slot; distractors fill the rest to reach 2+2.
            b_fam = B["family"]
            need_disp = 1 if b_fam == "displacement" else 2
            need_rot = 2 if b_fam == "displacement" else 1

            candidates = [x for x in in_quarter if x is not A and x is not B]
            used_labels = {B["label"]}

            disp_pool = _best_one_per_label(
                [x for x in candidates if x["family"] == "displacement"], used_labels, A["t0"], B["t0"]
            )
            rot_pool = _best_one_per_label(
                [x for x in candidates if x["family"] == "rotation"], used_labels, A["t0"], B["t0"]
            )

            if len(disp_pool) < need_disp or len(rot_pool) < need_rot:
                continue

            distractor_events = disp_pool[:need_disp] + rot_pool[:need_rot]
            chosen_four = [B] + distractor_events
            option_strings = [x["label"] for x in chosen_four]
            if len(set(option_strings)) != 4:
                continue

            correct_label = option_strings[0]
            rng.shuffle(option_strings)
            options = {chr(ord("A") + k): option_strings[k] for k in range(4)}
            answer = next(k for k, v in options.items() if v == correct_label)
            qa_pool.append({
                "question": f"In the {bucket_labels[w]} of the video, {mid}which of these happens next?",
                "options": options,
                "answer": answer,
                "evidence": {
                    "events_used": [A["event"], B["event"]],
                    "bucket": bucket_labels[w],
                    "time_range": [t_start, t_end],
                    "event_A": A["event"],
                    "event_B": B["event"],
                    "event_A_time": A["time"],
                    "event_B_time": B["time"],
                    "computation": (
                        f"ordering: B is earliest event (any super-family) with B.t0 >= A.t1+10; "
                        f"options = 2 displacement + 2 rotation labels; B family={b_fam}"
                    ),
                },
            })

    seen: set = set()
    unique_pool: List[Dict[str, Any]] = []
    for q in qa_pool:
        ev = q["evidence"]
        key = (ev["bucket"], ev["event_A"]["t0"], ev["event_B"]["t0"], frozenset(q["options"].items()), q["answer"])
        if key not in seen:
            seen.add(key)
            unique_pool.append(q)
    return unique_pool


# Translation = displacement only (for "next translation" speed questions)
TRANSLATION_CODES = ["displacement_x", "displacement_z"]
# Min gap (frames) between anchor end and next translation start for Type 3 speed questions
SPEED_NEXT_TRANSLATION_GAP_FRAMES = 10


def _shuffle_speed_options(options: Dict[str, str], answer: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    """Shuffle option order (A/B/C/D), preserve correct answer. Returns (new_options, new_answer)."""
    items = list(options.items())
    rng.shuffle(items)
    new_options = {chr(ord("A") + i): label for i, (_, label) in enumerate(items)}
    correct_label = options[answer]
    new_answer = next(letter for letter, label in new_options.items() if label == correct_label)
    return new_options, new_answer




def generate_trajectory_affordance_qa_pool(
    motion_json: Dict[str, Any], stats: Dict[str, Any], rng: random.Random
) -> List[Dict[str, Any]]:
    """Generate trajectory-affordance QA from motion-track displacement segments (4 quarters, exclude very_short/slight).
    Uses only displacement_x and displacement_z. Tie-break is random (not earliest quarter).
    Templates: largest/smallest net rightward, leftward, forward displacement; largest/smallest path length;
    furthest from start; overall endpoint bias."""
    qa_pool = []
    segments = extract_displacement_segments(motion_json)
    if not segments:
        return qa_pool
    fps = stats["fps"]
    num_frames = stats["num_frames"]
    tb = stats["temporal_boundaries"]
    traj = compute_trajectory_stats(segments, num_frames, fps, tb)
    per_quarter = traj["per_quarter"]
    overall = traj["overall"]
    quarter_labels = ["first", "second", "third", "fourth"]
    segments_x = [s for s in segments if s["axis"] == "displacement_x"]
    segments_z = [s for s in segments if s["axis"] == "displacement_z"]
    TIE_BREAK_RANDOM = "random_quarter"

    def _argmax_rng(pairs: List[Tuple[int, float]]) -> Optional[int]:
        if not pairs:
            return None
        best_val = max(v for (_, v) in pairs)
        return rng.choice([q for (q, v) in pairs if v == best_val])

    def _argmin_rng(pairs: List[Tuple[int, float]]) -> Optional[int]:
        if not pairs:
            return None
        best_val = min(v for (_, v) in pairs)
        return rng.choice([q for (q, v) in pairs if v == best_val])

    def _tie_rng(pairs: List[Tuple[int, float]], best_q: int) -> Tuple[bool, List[int]]:
        best_val = next((v for (q, v) in pairs if q == best_q), None)
        if best_val is None:
            return False, []
        tied = [q for (q, v) in pairs if abs(v - best_val) < TIE_EPS]
        return len(tied) > 1, sorted(tied)

    def make_quarter_opts(best_q: int) -> Tuple[Dict[str, str], str]:
        """Shuffled {A/B/C/D: 'X quarter'} options with correct answer letter."""
        indices = list(range(4))
        rng.shuffle(indices)
        options = {chr(65 + i): f"{quarter_labels[indices[i]]} quarter" for i in range(4)}
        answer = next(k for k, v in options.items() if v == f"{quarter_labels[best_q - 1]} quarter")
        return options, answer

    def append_quarter_qa(question: str, pairs: List[Tuple[int, float]], best_q: int,
                          segments_used: List, raw_key: str, computation: str) -> None:
        options, answer = make_quarter_opts(best_q)
        is_tie, tied = _tie_rng(pairs, best_q)
        qa_pool.append({
            "question": question,
            "options": options,
            "answer": answer,
            "evidence": {
                "units": TRACK_UNITS,
                "quarter_assignment": QUARTER_ASSIGNMENT_SEGMENT,
                "tie_break": TIE_BREAK_RANDOM,
                "tie_eps": TIE_EPS,
                "is_tie": is_tie,
                "tied_quarters": tied,
                "segments_used": segments_used,
                "computation": computation,
                "raw_values": {raw_key: {str(q): v for (q, v) in pairs}, "answer_quarter": best_q},
            },
        })

    # --- 1. Largest net rightward displacement ---
    q_right = [(i, max(per_quarter[i]["net_dx"], 0.0)) for i in range(1, 5)]
    best_q = _argmax_rng(q_right)
    if best_q is not None and any(v > 0 for (_, v) in q_right):
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_right)
        append_quarter_qa(
            "In which quarter does the person move most to the right (largest net rightward displacement)?",
            q_right, best_q, segments_x, "per_quarter_rightward_dx_units",
            f"rightward = max(net_dx, 0) per quarter; argmax random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 2. Smallest net rightward displacement (non-zero; requires >= 2 non-zero quarters) ---
    q_right_nz = [(i, v) for (i, v) in q_right if v > 0]
    if len(q_right_nz) >= 2:
        best_q = _argmin_rng(q_right_nz)
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_right_nz)
        append_quarter_qa(
            "In which quarter does the person move least to the right (smallest non-zero net rightward displacement)?",
            q_right_nz, best_q, segments_x, "per_quarter_rightward_dx_units_nonzero",
            f"rightward = max(net_dx, 0) per quarter (non-zero only); argmin random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 3. Largest net leftward displacement ---
    q_left = [(i, max(-per_quarter[i]["net_dx"], 0.0)) for i in range(1, 5)]
    best_q = _argmax_rng(q_left)
    if best_q is not None and any(v > 0 for (_, v) in q_left):
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_left)
        append_quarter_qa(
            "In which quarter does the person move most to the left (largest net leftward displacement)?",
            q_left, best_q, segments_x, "per_quarter_leftward_dx_units",
            f"leftward = max(-net_dx, 0) per quarter; argmax random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 4. Smallest net leftward displacement (non-zero; requires >= 2 non-zero quarters) ---
    q_left_nz = [(i, v) for (i, v) in q_left if v > 0]
    if len(q_left_nz) >= 2:
        best_q = _argmin_rng(q_left_nz)
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_left_nz)
        append_quarter_qa(
            "In which quarter does the person move least to the left (smallest non-zero net leftward displacement)?",
            q_left_nz, best_q, segments_x, "per_quarter_leftward_dx_units_nonzero",
            f"leftward = max(-net_dx, 0) per quarter (non-zero only); argmin random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 5. Largest net forward (away from start) displacement ---
    q_fwd = [(i, max(per_quarter[i]["net_dz"], 0.0)) for i in range(1, 5)]
    best_q = _argmax_rng(q_fwd)
    if best_q is not None and any(v > 0 for (_, v) in q_fwd):
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_fwd)
        append_quarter_qa(
            "In which quarter does the person move most away from their starting position (largest net forward displacement)?",
            q_fwd, best_q, segments_z, "per_quarter_away_dz_units",
            f"away-from-start = max(net_dz, 0) per quarter; argmax random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 6. Smallest net forward displacement (non-zero; requires >= 2 non-zero quarters) ---
    q_fwd_nz = [(i, v) for (i, v) in q_fwd if v > 0]
    if len(q_fwd_nz) >= 2:
        best_q = _argmin_rng(q_fwd_nz)
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_fwd_nz)
        append_quarter_qa(
            "In which quarter does the person move least away from their starting position (smallest non-zero net forward displacement)?",
            q_fwd_nz, best_q, segments_z, "per_quarter_away_dz_units_nonzero",
            f"away-from-start = max(net_dz, 0) per quarter (non-zero only); argmin random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 7. Overall endpoint bias (right / left / about the same / no horizontal motion) ---
    path_x = overall["path_x"]
    net_dx = overall["net_dx"]
    eps_dx = max(EPS_DX_ABSOLUTE, EPS_DX_RELATIVE * path_x) if path_x > 0 else EPS_DX_ABSOLUTE
    if path_x < NO_X_MOTION_THRESHOLD:
        overall_key = "no significant horizontal movement"
    elif abs(net_dx) < eps_dx:
        overall_key = "same"
    elif net_dx >= eps_dx:
        overall_key = "at the right"
    else:
        overall_key = "at the left"
    all_bias_opts = ["at the right", "at the left", "same", "no significant horizontal movement"]
    rng.shuffle(all_bias_opts)
    options_bias = {chr(65 + i): all_bias_opts[i] for i in range(4)}
    answer_bias = next(k for k, v in options_bias.items() if v == overall_key)
    qa_pool.append({
        "question": "Where is the person horizontally relative to where they started at the end of the video?",
        "options": options_bias,
        "answer": answer_bias,
        "evidence": {
            "units": TRACK_UNITS,
            "eps_dx_policy": EPS_DX_POLICY_STR,
            "eps_dx_units": eps_dx,
            "segments_used": segments_x,
            "computation": (
                f"path_x={path_x:.2f}, net_dx={net_dx:.2f}, eps_dx={eps_dx:.2f}; "
                f"path_x<{NO_X_MOTION_THRESHOLD}->no significant horizontal movement, "
                f"|net_dx|<eps_dx->same, else at the right/at the left -> {overall_key}"
            ),
            "raw_values": {
                "net_dx_units": net_dx,
                "abs_net_dx_units": abs(net_dx),
                "path_x_units": path_x,
                "eps_dx_units": eps_dx,
                "answer": overall_key,
            },
        },
    })

    # --- 8. Largest total path length ---
    q_path = [(i, per_quarter[i]["path_x"] + per_quarter[i]["path_z"]) for i in range(1, 5)]
    best_q = _argmax_rng(q_path)
    if best_q is not None and any(v > 0 for (_, v) in q_path):
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_path)
        append_quarter_qa(
            "In which quarter does the person travel the largest total path length (sum of horizontal and depth movement)?",
            q_path, best_q, traj["segments_used"], "per_quarter_path_units",
            f"path_x+path_z per quarter; argmax random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 9. Smallest total path length (non-zero; requires >= 2 non-zero quarters) ---
    q_path_nz = [(i, v) for (i, v) in q_path if v > 0]
    if len(q_path_nz) >= 2:
        best_q = _argmin_rng(q_path_nz)
        vals_str = ", ".join(f"Q{i}={v:.2f}" for (i, v) in q_path_nz)
        append_quarter_qa(
            "In which quarter does the person travel the smallest total path length (sum of horizontal and depth movement)?",
            q_path_nz, best_q, traj["segments_used"], "per_quarter_path_units_nonzero",
            f"path_x+path_z per quarter (non-zero only); argmin random tie-break. {vals_str} -> Q{best_q}",
        )

    # --- 10. Which quarter ends furthest from starting position ---
    trajectory = build_trajectory_from_segments(segments)
    if len(trajectory) >= 2:
        bounds = _quarter_frame_bounds(tb, fps)
        max_dist_by_q: List[Tuple[int, float]] = []
        max_dist_detail: Dict[str, Optional[float]] = {}
        for q_idx, (f_lo, f_hi) in enumerate(bounds, start=1):
            pts = [dist_to_start(px, pz) for (f, px, pz) in trajectory if f_lo <= f <= f_hi]
            if pts:
                max_d = max(pts)
                max_dist_by_q.append((q_idx, max_d))
                max_dist_detail[str(q_idx)] = max_d
            else:
                max_dist_detail[str(q_idx)] = None
        pairs_far = [(q, d) for (q, d) in max_dist_by_q if d > 0]
        if pairs_far:
            best_q = _argmax_rng(pairs_far)
            options, answer = make_quarter_opts(best_q)
            is_tie, tied = _tie_rng(pairs_far, best_q)
            vals_str = ", ".join(f"Q{q}={d:.2f}" for (q, d) in pairs_far)
            qa_pool.append({
                "question": "In which quarter does the person reach the point furthest from their starting position?",
                "options": options,
                "answer": answer,
                "evidence": {
                    "units": TRACK_UNITS,
                    "quarter_assignment": QUARTER_ASSIGNMENT_POINT_FRAME,
                    "tie_break": TIE_BREAK_RANDOM,
                    "tie_eps": TIE_EPS,
                    "is_tie": is_tie,
                    "tied_quarters": tied,
                    "computation": (
                        f"dist_to_start=||pos_t|| per trajectory point; max within each quarter; "
                        f"argmax random tie-break. {vals_str} -> Q{best_q}"
                    ),
                    "raw_values": {"max_dist_by_quarter_units": max_dist_detail, "answer_quarter": best_q},
                },
            })

        # --- 11. Trajectory shape ---
        all_dists = [dist_to_start(px, pz) for (_, px, pz) in trajectory]
        max_dist_shape = max(all_dists)
        final_dist = all_dists[-1]
        return_threshold = max(RETURN_ABSOLUTE_MIN_THRESH, RETURN_CLOSE_FRACTION * max_dist_shape)
        direction_changes = _count_direction_changes(all_dists)

        if max_dist_shape < NO_MOVEMENT_THRESHOLD:
            shape_answer = "remains near starting position throughout"
        elif final_dist <= return_threshold:
            shape_answer = "returns close to starting position by the end"
        elif direction_changes >= TRAJ_SHAPE_HIGH_VARIANCE_MIN_CHANGES:
            shape_answer = "oscillates but ends far from start"
        else:
            shape_answer = "moves away and stays away"

        shape_opts = [
            "returns close to starting position by the end",
            "moves away and stays away",
            "oscillates but ends far from start",
            "remains near starting position throughout",
        ]
        rng.shuffle(shape_opts)
        options_shape = {chr(65 + i): shape_opts[i] for i in range(4)}
        answer_shape = next(k for k, v in options_shape.items() if v == shape_answer)
        qa_pool.append({
            "question": "How does the person's overall position evolve relative to their starting position?",
            "options": options_shape,
            "answer": answer_shape,
            "evidence": {
                "units": TRACK_UNITS,
                "max_dist_units": max_dist_shape,
                "final_dist_units": final_dist,
                "return_threshold_units": return_threshold,
                "near_threshold_units": NO_MOVEMENT_THRESHOLD,
                "direction_changes": direction_changes,
                "high_variance_min_changes": TRAJ_SHAPE_HIGH_VARIANCE_MIN_CHANGES,
                "computation": (
                    f"max_dist={max_dist_shape:.2f}, final_dist={final_dist:.2f}, "
                    f"return_threshold={return_threshold:.2f}, near_threshold={NO_MOVEMENT_THRESHOLD}, "
                    f"direction_changes={direction_changes}; "
                    f"priority: near -> returns -> oscillates (changes>={TRAJ_SHAPE_HIGH_VARIANCE_MIN_CHANGES}) -> stays away "
                    f"-> {shape_answer}"
                ),
                "raw_values": {
                    "max_dist_units": max_dist_shape,
                    "final_dist_units": final_dist,
                    "return_threshold_units": return_threshold,
                    "direction_changes": direction_changes,
                    "shape": shape_answer,
                },
            },
        })

    return qa_pool


def generate_existence_qa_pool(
    events: List[Dict[str, Any]], stats: Dict[str, Any], motion_json: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generate pool of existence questions. If motion_json is provided, adds jump question (displacement_y up-then-down)."""
    qa_pool = []
    
    # Semantic action questions (always included)
    # Walking - only "very_long" movements count as walking
    walking_events = [e for e in events if e["code"] in ["displacement_x", "displacement_z"] and "very_long" in e["spatial_category"]]
    answer = "A" if len(walking_events) > 0 else "B"
    walking_evidence_used = _max_intensity_event(walking_events)
    qa_pool.append({
        "concept_id": "walking",
        "question": "Is there walking in the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": answer,
        "evidence": {
            "events_used": walking_evidence_used,
            "computation": _existence_computation(walking_evidence_used, f"very_long horizontal movement detected: {len(walking_events)} events"),
            "raw_values": _count_metric_raw_values(len(walking_events)),
        }
    })
    
    # Jump: displacement_y consecutive pair (by motion_index) moderate+/long/verylong_up then moderate+/long/verylong_down
    if motion_json is not None:
        jump_count, jump_events_used = detect_jumps(motion_json)
        answer = "A" if jump_count > 0 else "B"
        jump_evidence_used = _max_intensity_event(jump_events_used)
        qa_pool.append({
            "concept_id": "jump",
            "question": "Is there a jump in the video?",
            "options": {"A": "Yes", "B": "No"},
            "answer": answer,
            "evidence": {
                "events_used": jump_evidence_used,
                "computation": _existence_computation(jump_evidence_used, f"jump = consecutive displacement_y (moderate/long/very_long) up then (moderate/long/very_long) down; count = {jump_count}"),
                "raw_values": _count_metric_raw_values(jump_count),
            },
        })
    
    # Significant turning
    turning_events = [e for e in events if e["code"] == "rotation_yaw" and is_significant_motion(e)]
    answer = "A" if len(turning_events) > 0 else "B"
    turning_evidence_used = _max_intensity_event(turning_events)
    qa_pool.append({
        "concept_id": "significant_turning",
        "question": "Is there significant turning in the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": answer,
        "evidence": {
            "events_used": turning_evidence_used,
            "computation": _existence_computation(turning_evidence_used, f"significant turning detected: {len(turning_events)} events"),
            "raw_values": _count_metric_raw_values(len(turning_events)),
        }
    })
    
    # Significant leaning
    leaning_events = [e for e in events if e["code"] in ["rotation_pitch", "rotation_roll"] and is_significant_motion(e)]
    answer = "A" if len(leaning_events) > 0 else "B"
    leaning_evidence_used = _max_intensity_event(leaning_events)
    qa_pool.append({
        "concept_id": "significant_leaning",
        "question": "Is there significant leaning in the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": answer,
        "evidence": {
            "events_used": leaning_evidence_used,
            "computation": _existence_computation(leaning_evidence_used, f"significant leaning detected: {len(leaning_events)} events"),
            "raw_values": _count_metric_raw_values(len(leaning_events)),
        }
    })
    
    # Direction-specific questions: cumulative magnitude (Option A) + dominance filter
    # Pairs: (code, dir1, dir2, question1, question2) then single evidence keys for events_used/computation/raw_values
    r = EXISTENCE_DOMINANCE_RATIO

    # displacement_x: left vs right
    left_events = stats["displacement_x"]["left"]
    right_events = stats["displacement_x"]["right"]
    S_left = _cumulative_intensity(left_events)
    S_right = _cumulative_intensity(right_events)
    eps_x = _existence_epsilon_for_code("displacement_x")
    yes_left = _existence_yes_for_direction(S_left, S_right, eps_x, r)
    yes_right = _existence_yes_for_direction(S_right, S_left, eps_x, r)
    qa_pool.append({
        "concept_id": "move_left",
        "question": "Does the person move to the left at any point?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_left else "B",
        "evidence": {
            "events_used": _max_intensity_event(left_events),
            "computation": _existence_computation(_max_intensity_event(left_events), f"S_left={S_left:.2f}, S_right={S_right:.2f}, eps={eps_x}, dominance r={r}; yes_left={yes_left}"),
            "raw_values": _directional_strength_raw_values(S_left, S_right, eps_x, r, len(left_events), len(right_events))
        }
    })
    qa_pool.append({
        "concept_id": "move_right",
        "question": "Does the person move to the right at any point?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_right else "B",
        "evidence": {
            "events_used": _max_intensity_event(right_events),
            "computation": _existence_computation(_max_intensity_event(right_events), f"S_right={S_right:.2f}, S_left={S_left:.2f}, eps={eps_x}, dominance r={r}; yes_right={yes_right}"),
            "raw_values": _directional_strength_raw_values(S_right, S_left, eps_x, r, len(right_events), len(left_events))
        }
    })

    # displacement_z: away (forward) vs towards (backward)
    fwd_events = stats["displacement_z"]["forward"]
    bwd_events = stats["displacement_z"]["backward"]
    S_fwd = _cumulative_intensity(fwd_events)
    S_bwd = _cumulative_intensity(bwd_events)
    eps_z = _existence_epsilon_for_code("displacement_z")
    yes_fwd = _existence_yes_for_direction(S_fwd, S_bwd, eps_z, r)
    yes_bwd = _existence_yes_for_direction(S_bwd, S_fwd, eps_z, r)
    qa_pool.append({
        "concept_id": "move_away_start",
        "question": "Does the person move away from starting position at any point?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_fwd else "B",
        "evidence": {
            "events_used": _max_intensity_event(fwd_events),
            "computation": _existence_computation(_max_intensity_event(fwd_events), f"S_away={S_fwd:.2f}, S_towards={S_bwd:.2f}, eps={eps_z}, dominance r={r}; yes_away={yes_fwd}"),
            "raw_values": _directional_strength_raw_values(S_fwd, S_bwd, eps_z, r, len(fwd_events), len(bwd_events))
        }
    })
    qa_pool.append({
        "concept_id": "move_towards_start",
        "question": "Does the person move towards starting position at any point?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_bwd else "B",
        "evidence": {
            "events_used": _max_intensity_event(bwd_events),
            "computation": _existence_computation(_max_intensity_event(bwd_events), f"S_towards={S_bwd:.2f}, S_away={S_fwd:.2f}, eps={eps_z}, dominance r={r}; yes_towards={yes_bwd}"),
            "raw_values": _directional_strength_raw_values(S_bwd, S_fwd, eps_z, r, len(bwd_events), len(fwd_events))
        }
    })

    # rotation_yaw: clockwise vs counterclockwise
    cw_events = stats["rotation_yaw"]["clockwise"]
    ccw_events = stats["rotation_yaw"]["counterclockwise"]
    S_cw = _cumulative_intensity(cw_events)
    S_ccw = _cumulative_intensity(ccw_events)
    eps_yaw = _existence_epsilon_for_code("rotation_yaw")
    yes_cw = _existence_yes_for_direction(S_cw, S_ccw, eps_yaw, r)
    yes_ccw = _existence_yes_for_direction(S_ccw, S_cw, eps_yaw, r)
    qa_pool.append({
        "concept_id": "turn_clockwise",
        "question": "Does the person ever turn clockwise?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_cw else "B",
        "evidence": {
            "events_used": _max_intensity_event(cw_events),
            "computation": _existence_computation(_max_intensity_event(cw_events), f"S_cw={S_cw:.2f}, S_ccw={S_ccw:.2f}, eps={eps_yaw}, dominance r={r}; yes_cw={yes_cw}"),
            "raw_values": _directional_strength_raw_values(S_cw, S_ccw, eps_yaw, r, len(cw_events), len(ccw_events))
        }
    })
    qa_pool.append({
        "concept_id": "turn_counterclockwise",
        "question": "Does the person ever turn counterclockwise?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_ccw else "B",
        "evidence": {
            "events_used": _max_intensity_event(ccw_events),
            "computation": _existence_computation(_max_intensity_event(ccw_events), f"S_ccw={S_ccw:.2f}, S_cw={S_cw:.2f}, eps={eps_yaw}, dominance r={r}; yes_ccw={yes_ccw}"),
            "raw_values": _directional_strength_raw_values(S_ccw, S_cw, eps_yaw, r, len(ccw_events), len(cw_events))
        }
    })

    # rotation_pitch: forward vs backward
    pitch_fwd_events = stats["rotation_pitch"]["forward"]
    pitch_bwd_events = stats["rotation_pitch"]["backward"]
    S_pitch_fwd = _cumulative_intensity(pitch_fwd_events)
    S_pitch_bwd = _cumulative_intensity(pitch_bwd_events)
    eps_pitch = _existence_epsilon_for_code("rotation_pitch")
    yes_pitch_fwd = _existence_yes_for_direction(S_pitch_fwd, S_pitch_bwd, eps_pitch, r)
    yes_pitch_bwd = _existence_yes_for_direction(S_pitch_bwd, S_pitch_fwd, eps_pitch, r)
    qa_pool.append({
        "concept_id": "lean_forward",
        "question": "Does the person lean forward during the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_pitch_fwd else "B",
        "evidence": {
            "events_used": _max_intensity_event(pitch_fwd_events),
            "computation": _existence_computation(_max_intensity_event(pitch_fwd_events), f"S_forward={S_pitch_fwd:.2f}, S_backward={S_pitch_bwd:.2f}, eps={eps_pitch}, dominance r={r}; yes_forward={yes_pitch_fwd}"),
            "raw_values": _directional_strength_raw_values(S_pitch_fwd, S_pitch_bwd, eps_pitch, r, len(pitch_fwd_events), len(pitch_bwd_events))
        }
    })
    qa_pool.append({
        "concept_id": "lean_backward",
        "question": "Does the person lean backward during the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_pitch_bwd else "B",
        "evidence": {
            "events_used": _max_intensity_event(pitch_bwd_events),
            "computation": _existence_computation(_max_intensity_event(pitch_bwd_events), f"S_backward={S_pitch_bwd:.2f}, S_forward={S_pitch_fwd:.2f}, eps={eps_pitch}, dominance r={r}; yes_backward={yes_pitch_bwd}"),
            "raw_values": _directional_strength_raw_values(S_pitch_bwd, S_pitch_fwd, eps_pitch, r, len(pitch_bwd_events), len(pitch_fwd_events))
        }
    })

    # rotation_roll: left vs right
    roll_left_events = stats["rotation_roll"]["left"]
    roll_right_events = stats["rotation_roll"]["right"]
    S_roll_left = _cumulative_intensity(roll_left_events)
    S_roll_right = _cumulative_intensity(roll_right_events)
    eps_roll = _existence_epsilon_for_code("rotation_roll")
    yes_roll_left = _existence_yes_for_direction(S_roll_left, S_roll_right, eps_roll, r)
    yes_roll_right = _existence_yes_for_direction(S_roll_right, S_roll_left, eps_roll, r)
    qa_pool.append({
        "concept_id": "lean_left",
        "question": "Does the person lean to the left during the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_roll_left else "B",
        "evidence": {
            "events_used": _max_intensity_event(roll_left_events),
            "computation": _existence_computation(_max_intensity_event(roll_left_events), f"S_left={S_roll_left:.2f}, S_right={S_roll_right:.2f}, eps={eps_roll}, dominance r={r}; yes_left={yes_roll_left}"),
            "raw_values": _directional_strength_raw_values(S_roll_left, S_roll_right, eps_roll, r, len(roll_left_events), len(roll_right_events))
        }
    })
    qa_pool.append({
        "concept_id": "lean_right",
        "question": "Does the person lean to the right during the video?",
        "options": {"A": "Yes", "B": "No"},
        "answer": "A" if yes_roll_right else "B",
        "evidence": {
            "events_used": _max_intensity_event(roll_right_events),
            "computation": _existence_computation(_max_intensity_event(roll_right_events), f"S_right={S_roll_right:.2f}, S_left={S_roll_left:.2f}, eps={eps_roll}, dominance r={r}; yes_right={yes_roll_right}"),
            "raw_values": _directional_strength_raw_values(S_roll_right, S_roll_left, eps_roll, r, len(roll_right_events), len(roll_left_events))
        }
    })
    
    return qa_pool


def generate_negation_qa_pool(
    events: List[Dict[str, Any]], stats: Dict[str, Any], motion_json: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Generate counterfactual/negation questions.
    Directional negation uses the logical negation of the positive existence rule (same magnitude + dominance as existence).
    Phrasings: no X at any time, never X, at no point does person X, it is true that did not X (no 'avoid')."""
    qa_pool = []
    r = EXISTENCE_DOMINANCE_RATIO

    def add_negation_qa(concept_id: str, question: str, options: Dict[str, str], statement_true: bool, events_list: List, computation: str, raw_values: Dict) -> None:
        answer = "A" if statement_true else "B"
        events_used = _max_intensity_event(events_list)
        qa_pool.append({
            "concept_id": concept_id,
            "question": question,
            "options": options,
            "answer": answer,
            "evidence": {
                "events_used": events_used,
                "computation": _existence_computation(events_used, computation),
                "raw_values": {**raw_values, "statement_true": statement_true},
            },
        })

    # Direction-specific: (base, past, gerund) for phrasing variants
    left_events = stats["displacement_x"]["left"]
    right_events = stats["displacement_x"]["right"]

    fwd_events = stats["displacement_z"]["forward"]
    bwd_events = stats["displacement_z"]["backward"]

    cw_events = stats["rotation_yaw"]["clockwise"]
    ccw_events = stats["rotation_yaw"]["counterclockwise"]

    pitch_fwd_events = stats["rotation_pitch"]["forward"]
    pitch_bwd_events = stats["rotation_pitch"]["backward"]

    roll_left_events = stats["rotation_roll"]["left"]
    roll_right_events = stats["rotation_roll"]["right"]

    # Directional negation uses the same thresholds as positive existence, then negates the result.
    S_left = _cumulative_intensity(left_events)
    S_right = _cumulative_intensity(right_events)
    eps_x = _existence_epsilon_for_code("displacement_x")
    yes_left = _existence_yes_for_direction(S_left, S_right, eps_x, r)
    yes_right = _existence_yes_for_direction(S_right, S_left, eps_x, r)

    S_fwd = _cumulative_intensity(fwd_events)
    S_bwd = _cumulative_intensity(bwd_events)
    eps_z = _existence_epsilon_for_code("displacement_z")
    yes_fwd = _existence_yes_for_direction(S_fwd, S_bwd, eps_z, r)
    yes_bwd = _existence_yes_for_direction(S_bwd, S_fwd, eps_z, r)

    S_cw = _cumulative_intensity(cw_events)
    S_ccw = _cumulative_intensity(ccw_events)
    eps_yaw = _existence_epsilon_for_code("rotation_yaw")
    yes_cw = _existence_yes_for_direction(S_cw, S_ccw, eps_yaw, r)
    yes_ccw = _existence_yes_for_direction(S_ccw, S_cw, eps_yaw, r)

    S_pitch_fwd = _cumulative_intensity(pitch_fwd_events)
    S_pitch_bwd = _cumulative_intensity(pitch_bwd_events)
    eps_pitch = _existence_epsilon_for_code("rotation_pitch")
    yes_pitch_fwd = _existence_yes_for_direction(S_pitch_fwd, S_pitch_bwd, eps_pitch, r)
    yes_pitch_bwd = _existence_yes_for_direction(S_pitch_bwd, S_pitch_fwd, eps_pitch, r)

    S_roll_left = _cumulative_intensity(roll_left_events)
    S_roll_right = _cumulative_intensity(roll_right_events)
    eps_roll = _existence_epsilon_for_code("rotation_roll")
    yes_roll_left = _existence_yes_for_direction(S_roll_left, S_roll_right, eps_roll, r)
    yes_roll_right = _existence_yes_for_direction(S_roll_right, S_roll_left, eps_roll, r)

    # (base, past, gerund, events_list, statement_true, computation, raw_values)
    direction_negation_items = [
        (
            "move to the left", "moved to the left", "moving to the left", left_events, not yes_left,
            f"statement_true = not yes_left where yes_left={yes_left}; S_left={S_left:.2f}, S_right={S_right:.2f}, eps={eps_x}, dominance r={r}",
            _directional_strength_raw_values(S_left, S_right, eps_x, r, len(left_events), len(right_events)),
        ),
        (
            "move to the right", "moved to the right", "moving to the right", right_events, not yes_right,
            f"statement_true = not yes_right where yes_right={yes_right}; S_right={S_right:.2f}, S_left={S_left:.2f}, eps={eps_x}, dominance r={r}",
            _directional_strength_raw_values(S_right, S_left, eps_x, r, len(right_events), len(left_events)),
        ),
        (
            "move away from starting position", "moved away from starting position", "moving away from starting position", fwd_events, not yes_fwd,
            f"statement_true = not yes_away where yes_away={yes_fwd}; S_away={S_fwd:.2f}, S_towards={S_bwd:.2f}, eps={eps_z}, dominance r={r}",
            _directional_strength_raw_values(S_fwd, S_bwd, eps_z, r, len(fwd_events), len(bwd_events)),
        ),
        (
            "move towards starting position", "moved towards starting position", "moving towards starting position", bwd_events, not yes_bwd,
            f"statement_true = not yes_towards where yes_towards={yes_bwd}; S_towards={S_bwd:.2f}, S_away={S_fwd:.2f}, eps={eps_z}, dominance r={r}",
            _directional_strength_raw_values(S_bwd, S_fwd, eps_z, r, len(bwd_events), len(fwd_events)),
        ),
        (
            "turn clockwise", "turned clockwise", "turning clockwise", cw_events, not yes_cw,
            f"statement_true = not yes_cw where yes_cw={yes_cw}; S_cw={S_cw:.2f}, S_ccw={S_ccw:.2f}, eps={eps_yaw}, dominance r={r}",
            _directional_strength_raw_values(S_cw, S_ccw, eps_yaw, r, len(cw_events), len(ccw_events)),
        ),
        (
            "turn counterclockwise", "turned counterclockwise", "turning counterclockwise", ccw_events, not yes_ccw,
            f"statement_true = not yes_ccw where yes_ccw={yes_ccw}; S_ccw={S_ccw:.2f}, S_cw={S_cw:.2f}, eps={eps_yaw}, dominance r={r}",
            _directional_strength_raw_values(S_ccw, S_cw, eps_yaw, r, len(ccw_events), len(cw_events)),
        ),
        (
            "lean forward", "leaned forward", "leaning forward", pitch_fwd_events, not yes_pitch_fwd,
            f"statement_true = not yes_forward where yes_forward={yes_pitch_fwd}; S_forward={S_pitch_fwd:.2f}, S_backward={S_pitch_bwd:.2f}, eps={eps_pitch}, dominance r={r}",
            _directional_strength_raw_values(S_pitch_fwd, S_pitch_bwd, eps_pitch, r, len(pitch_fwd_events), len(pitch_bwd_events)),
        ),
        (
            "lean backward", "leaned backward", "leaning backward", pitch_bwd_events, not yes_pitch_bwd,
            f"statement_true = not yes_backward where yes_backward={yes_pitch_bwd}; S_backward={S_pitch_bwd:.2f}, S_forward={S_pitch_fwd:.2f}, eps={eps_pitch}, dominance r={r}",
            _directional_strength_raw_values(S_pitch_bwd, S_pitch_fwd, eps_pitch, r, len(pitch_bwd_events), len(pitch_fwd_events)),
        ),
        (
            "lean to the left", "leaned to the left", "leaning to the left", roll_left_events, not yes_roll_left,
            f"statement_true = not yes_left where yes_left={yes_roll_left}; S_left={S_roll_left:.2f}, S_right={S_roll_right:.2f}, eps={eps_roll}, dominance r={r}",
            _directional_strength_raw_values(S_roll_left, S_roll_right, eps_roll, r, len(roll_left_events), len(roll_right_events)),
        ),
        (
            "lean to the right", "leaned to the right", "leaning to the right", roll_right_events, not yes_roll_right,
            f"statement_true = not yes_right where yes_right={yes_roll_right}; S_right={S_roll_right:.2f}, S_left={S_roll_left:.2f}, eps={eps_roll}, dominance r={r}",
            _directional_strength_raw_values(S_roll_right, S_roll_left, eps_roll, r, len(roll_right_events), len(roll_left_events)),
        ),
    ]

    # Multiple phrasings per direction negation (Yes/No: A=yes=did not do it; True/False: A=True=did not do it)
    direction_concept_ids = [
        "move_left", "move_right", "move_away_start", "move_towards_start",
        "turn_clockwise", "turn_counterclockwise", "lean_forward", "lean_backward", "lean_left", "lean_right",
    ]
    for idx, (base, past, gerund, events_list, statement_true, comp, rv) in enumerate(direction_negation_items):
        cid = direction_concept_ids[idx]
        add_negation_qa(cid, f"Was there no {gerund} at any time?",
            {"A": "Yes", "B": "No"}, statement_true, events_list, comp, rv)
        add_negation_qa(cid, f"The person never {past}. True or False?",
            {"A": "True", "B": "False"}, statement_true, events_list, comp, rv)
        add_negation_qa(cid, f"At no point does the person {base}. True or False?",
            {"A": "True", "B": "False"}, statement_true, events_list, comp, rv)
        add_negation_qa(cid, f"It is true that the person did not {base} during the video.",
            {"A": "True", "B": "False"}, statement_true, events_list, comp, rv)

    # Semantic: walking, significant turning, significant leaning (multiple phrasings each)
    walking_events = [e for e in events if e["code"] in ["displacement_x", "displacement_z"] and "very_long" in e["spatial_category"]]
    no_walking = len(walking_events) == 0
    comp_w = f"no_walking = {no_walking}"
    rv_w = _count_metric_raw_values(len(walking_events))
    add_negation_qa("walking", "Was there no walking at any time?", {"A": "Yes", "B": "No"}, no_walking, walking_events, comp_w, rv_w)
    add_negation_qa("walking", "The person never walked. True or False?", {"A": "True", "B": "False"}, no_walking, walking_events, comp_w, rv_w)
    add_negation_qa("walking", "At no point does the person walk. True or False?", {"A": "True", "B": "False"}, no_walking, walking_events, comp_w, rv_w)
    add_negation_qa("walking", "It is true that there is no walking in the video.", {"A": "True", "B": "False"}, no_walking, walking_events, comp_w, rv_w)

    turning_events = [e for e in events if e["code"] == "rotation_yaw" and is_significant_motion(e)]
    no_turning = len(turning_events) == 0
    comp_t = f"no_significant_turning = {no_turning}"
    rv_t = _count_metric_raw_values(len(turning_events))
    add_negation_qa("significant_turning", "Was there no significant turning at any time?", {"A": "Yes", "B": "No"}, no_turning, turning_events, comp_t, rv_t)
    add_negation_qa("significant_turning", "The person never turned significantly. True or False?", {"A": "True", "B": "False"}, no_turning, turning_events, comp_t, rv_t)
    add_negation_qa("significant_turning", "At no point does the person turn significantly. True or False?", {"A": "True", "B": "False"}, no_turning, turning_events, comp_t, rv_t)
    add_negation_qa("significant_turning", "It is true that there is no significant turning in the video.", {"A": "True", "B": "False"}, no_turning, turning_events, comp_t, rv_t)

    leaning_events = [e for e in events if e["code"] in ["rotation_pitch", "rotation_roll"] and is_significant_motion(e)]
    no_leaning = len(leaning_events) == 0
    comp_l = f"no_significant_leaning = {no_leaning}"
    rv_l = _count_metric_raw_values(len(leaning_events))
    add_negation_qa("significant_leaning", "Was there no significant leaning at any time?", {"A": "Yes", "B": "No"}, no_leaning, leaning_events, comp_l, rv_l)
    add_negation_qa("significant_leaning", "The person never leaned significantly. True or False?", {"A": "True", "B": "False"}, no_leaning, leaning_events, comp_l, rv_l)
    add_negation_qa("significant_leaning", "At no point does the person lean significantly. True or False?", {"A": "True", "B": "False"}, no_leaning, leaning_events, comp_l, rv_l)
    add_negation_qa("significant_leaning", "It is true that there is no significant leaning in the video.", {"A": "True", "B": "False"}, no_leaning, leaning_events, comp_l, rv_l)

    if motion_json is not None:
        jump_count, jump_events_used = detect_jumps(motion_json)
        no_jump = jump_count == 0
        comp_j = f"no_jump = {no_jump}"
        rv_j = _count_metric_raw_values(jump_count)
        add_negation_qa("jump", "Was there no jump at any time?", {"A": "Yes", "B": "No"}, no_jump, jump_events_used, comp_j, rv_j)
        add_negation_qa("jump", "The person never jumped. True or False?", {"A": "True", "B": "False"}, no_jump, jump_events_used, comp_j, rv_j)
        add_negation_qa("jump", "At no point does the person jump. True or False?", {"A": "True", "B": "False"}, no_jump, jump_events_used, comp_j, rv_j)
        add_negation_qa("jump", "It is true that there is no jump in the video.", {"A": "True", "B": "False"}, no_jump, jump_events_used, comp_j, rv_j)

    return qa_pool


def sample_questions(
    pool: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    balance_yes_no: bool = False,
    always_include: List[str] = None,
    descriptor: Optional[str] = None,
    *,
    require_descriptor_in_text: bool = False,
) -> List[Dict[str, Any]]:
    """Randomly sample N questions from pool.
    
    If balance_yes_no=True, ensures roughly equal Yes/No distribution for existence questions.
    If always_include is provided, those questions (matched by question text) are always included.
    """
    if always_include is None:
        always_include = []

    # Deduplicate by exact question text (keep first occurrence)
    seen_questions: set = set()
    deduped_pool = []
    for q in pool:
        if q["question"] not in seen_questions:
            seen_questions.add(q["question"])
            deduped_pool.append(q)
    pool = deduped_pool

    # Find questions to always include
    always_included = []
    remaining_pool = []
    for q in pool:
        if any(always_text in q["question"] for always_text in always_include):
            always_included.append(q)
        else:
            remaining_pool.append(q)
    
    # If we need all questions, return pool
    if len(pool) <= n:
        return personalize_qa_questions(
            pool, descriptor, require_descriptor_in_text=require_descriptor_in_text
        )

    # Calculate how many more we need
    needed = n - len(always_included)
    if needed <= 0:
        return personalize_qa_questions(
            always_included[:n],
            descriptor,
            require_descriptor_in_text=require_descriptor_in_text,
        )

    if not balance_yes_no:
        # Just sample from remaining pool
        sampled = rng.sample(remaining_pool, min(needed, len(remaining_pool)))
        result = always_included + sampled
        rng.shuffle(result)  # Shuffle but keep always_included at start
        return personalize_qa_questions(
            result[:n], descriptor, require_descriptor_in_text=require_descriptor_in_text
        )
    
    # Separate Yes (A) and No (B) answers from remaining pool
    yes_questions = [q for q in remaining_pool if q["answer"] == "A"]
    no_questions = [q for q in remaining_pool if q["answer"] == "B"]
    
    # If one group is empty, just sample normally from remaining
    if len(yes_questions) == 0 or len(no_questions) == 0:
        sampled = rng.sample(remaining_pool, min(needed, len(remaining_pool)))
        result = always_included + sampled
        rng.shuffle(result)
        return personalize_qa_questions(
            result[:n], descriptor, require_descriptor_in_text=require_descriptor_in_text
        )
    
    # Balance: try to get roughly equal Yes/No
    n_yes_target = needed // 2
    n_no_target = needed - n_yes_target

    # Sample up to target from each group (may get fewer No if pool is Yes-heavy)
    n_yes_take = min(n_yes_target, len(yes_questions))
    n_no_take = min(n_no_target, len(no_questions))
    sampled_yes = rng.sample(yes_questions, n_yes_take)
    sampled_no = rng.sample(no_questions, n_no_take)
    sampled = sampled_yes + sampled_no
    rng.shuffle(sampled)

    # Fill remaining slots preferring No to keep balance (pool is often Yes-heavy)
    still_needed = needed - len(sampled)
    if still_needed > 0:
        remaining_yes = [q for q in yes_questions if q not in sampled_yes]
        remaining_no = [q for q in no_questions if q not in sampled_no]
        # Take up to half of fill from No, rest from Yes
        n_no_fill = min(still_needed // 2, len(remaining_no))
        n_yes_fill = still_needed - n_no_fill
        if n_no_fill > 0 and remaining_no:
            sampled.extend(rng.sample(remaining_no, min(n_no_fill, len(remaining_no))))
        if n_yes_fill > 0 and remaining_yes:
            sampled.extend(rng.sample(remaining_yes, min(n_yes_fill, len(remaining_yes))))
        # If we still need more (e.g. ran out of one side), take from the other
        still_needed = needed - len(sampled)
        if still_needed > 0:
            remaining_any = [q for q in remaining_pool if q not in sampled]
            if remaining_any:
                sampled.extend(rng.sample(remaining_any, min(still_needed, len(remaining_any))))

    result = always_included + sampled
    rng.shuffle(result)
    return personalize_qa_questions(
        result[:n], descriptor, require_descriptor_in_text=require_descriptor_in_text
    )


def sample_comparative_questions(pool: List[Dict[str, Any]], rng: random.Random, n: int = 7) -> List[Dict[str, Any]]:
    """Sample n comparative questions from the 10-item pool (5 global + 5 local). Pool order: [0]=dx, [1]=dz, [2]=yaw, [3]=pitch, [4]=roll, [5]=Q1 dx, [6]=Q2 dz, [7]=Q1 yaw, [8]=Q3 pitch, [9]=Q4 roll."""
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


def main():
    parser = argparse.ArgumentParser(description="Generate spatial motion QA pairs from JSON (questions-defined version)")
    parser.add_argument("--input", type=str, required=True, help="Path to motion JSON file")
    parser.add_argument("--video_path", type=str, default=None, help="Path to video file (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for QA JSON files")
    parser.add_argument("--questions_per_axis", type=int, default=5, help="Number of questions to sample per axis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: hash of motion_id)")
    parser.add_argument(
        "--description",
        action="store_true",
        help="Use motion JSON field 'descriptor' in every question (substitute phrasing + lead-in when needed). "
        "No effect if 'descriptor' is missing or empty.",
    )
    args = parser.parse_args()

    # Load JSON
    motion_json = load_motion_json(args.input)
    motion_id = motion_json.get("motion_id", "unknown")
    if args.description:
        descriptor = resolve_subject_descriptor(motion_json)
        require_descriptor_in_text = bool(descriptor)
        if not descriptor:
            print(
                "Warning: --description set but motion JSON has no usable 'descriptor'; "
                "questions stay generic.",
            )
    else:
        descriptor = None
        require_descriptor_in_text = False
    
    # Get video path
    video_path = args.video_path
    if video_path is None:
        # Try to derive from motion_path or use input path
        motion_path = motion_json.get("motion_path", "")
        if motion_path:
            # Replace .npy with .mp4 or similar
            video_path = motion_path.replace(".npy", ".mp4")
        else:
            video_path = args.input.replace("_spatial_caption.json", ".mp4")
    
    # Set up random seed
    if args.seed is None:
        seed = hash(motion_id) % (2**32)
    else:
        seed = args.seed
    rng = random.Random(seed)
    
    # Extract events and drop very_short so all axes (numerical, comparative, dominant, temporal, ordering, speed, existence) use the same set
    events = _exclude_very_short_events(extract_events(motion_json))
    
    # Get metadata
    num_frames = motion_json.get("num_frames", len(events))
    fps = motion_json.get("fps", 30.0)
    
    # Compute stats
    stats = compute_stats(events, num_frames, fps, rng)
    duration = stats["duration"]

    # Generate QA pools
    numerical_pool = generate_numerical_qa_pool(events, stats, rng)
    comparative_pool = generate_comparative_qa_pool(events, stats, fps, rng)
    dominant_pool = generate_dominant_qa_pool(events, stats, fps, duration, rng)
    temporal_pool = generate_temporal_qa_pool(events, stats, fps, duration, rng)
    ordering_pool = generate_ordering_qa_pool(events, stats, fps, duration, rng)
    trajectory_pool = generate_trajectory_affordance_qa_pool(motion_json, stats, rng)
    existence_pool = generate_existence_qa_pool(events, stats, motion_json)
    negation_pool = generate_negation_qa_pool(events, stats, motion_json)
    existence_qa = sample_one_per_concept(
        existence_pool,
        negation_pool,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )

    # Sample questions
    numerical_qa = sample_questions(
        numerical_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )
    comparative_qa = sample_questions(
        comparative_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )
    dominant_qa = sample_questions(
        dominant_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )
    temporal_qa = sample_questions(
        temporal_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )
    ordering_qa = sample_questions(
        ordering_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )
    trajectory_qa = sample_questions(
        trajectory_pool,
        args.questions_per_axis,
        rng,
        descriptor=descriptor,
        require_descriptor_in_text=require_descriptor_in_text,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write output files
    outputs = {
        "qa_numerical.json": numerical_qa,
        "qa_comparative.json": comparative_qa,
        "qa_dominant.json": dominant_qa,
        "qa_temporal.json": temporal_qa,
        "qa_ordering.json": ordering_qa,
        "qa_trajectory_affordance.json": trajectory_qa,
        "qa_existence.json": existence_qa,
    }
    
    for filename, qa_pairs in outputs.items():
        for item in qa_pairs:
            if "evidence" in item and item["evidence"]:
                item["evidence"] = round_evidence(item["evidence"])
        output_data = {
            "video_path": video_path,
            "motion_id": motion_id,
            "qa_pairs": qa_pairs
        }
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Written {len(qa_pairs)} questions to {output_path}")


if __name__ == "__main__":
    main()
