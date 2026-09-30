"""Build a small set of model cards as Markdown from data/models.json.

Selection: only entries with enough description text, as many different
tasks as possible (text, images, translation, audio, ...), and per task the
models with the most votes.

Usage:  python3 scripts/build_cards.py
"""

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "models.json"
OUT_DIR = ROOT / "data" / "model_cards"
N_CARDS = 20
MIN_TEXT_LEN = 300

# Tasks that matter for the sample questions are served first.
PREFERRED_TASKS = [
    "translation",
    "image classification",
    "object detection",
    "image segmentation",
    "text generation",
    "text classification",
    "speech",
    "audio",
    "text-to-image",
    "question answering",
    "summarization",
    "embedding",
]


def parse_json_field(value):
    """Many fields are JSON stored as a string – parse them safely."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def tasks_of(entry):
    """Return the lower-case task names of an entry."""
    names = []
    for kw in parse_json_field(entry.get("keywords")):
        if isinstance(kw, dict):
            name = kw.get("name") or kw.get("ref")
            if name:
                names.append(name.strip().lower())
    return names


def slug(model_id):
    """Turn a model ID like "Microsoft/phi" into a filename like "Microsoft__phi"."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id.replace("/", "__"))


def to_markdown(entry):
    """Build the Markdown text of one model card."""
    tasks = tasks_of(entry)
    frameworks = parse_json_field(entry.get("frameworks"))
    lines = [
        f"# {entry.get('name') or entry['modelId']}",
        "",
        f"- Model ID: {entry['modelId']}",
        f"- License: {entry.get('license') or 'unknown'}",
        f"- Tasks: {', '.join(tasks) or 'unknown'}",
        f"- Frameworks: {', '.join(frameworks) or 'unknown'}",
        f"- Shared by: {entry.get('sharedBy') or 'unknown'}",
        f"- Source: {entry.get('url') or 'unknown'}",
        "",
        "## Description",
        "",
        entry["intendedUse"].strip(),
    ]
    trained_on = parse_json_field(entry.get("trainedOn"))
    if trained_on:
        lines += ["", "## Trained on / based on", ""]
        lines += [f"- {t}" for t in trained_on]
    if entry.get("referencePublication"):
        lines += ["", "## References", "", entry["referencePublication"].strip()]
    return "\n".join(lines) + "\n"


def select(entries):
    """Pick N_CARDS entries with enough text, spread over different tasks.

    Returns the chosen entries and the number of candidates with enough text.
    """
    candidates = [
        e for e in entries if len((e.get("intendedUse") or "").strip()) >= MIN_TEXT_LEN
    ]
    candidates.sort(key=lambda e: e.get("voteCount") or 0, reverse=True)

    by_task = defaultdict(list)
    for e in candidates:
        for t in tasks_of(e) or ["unknown"]:
            by_task[t].append(e)

    # Order: preferred tasks first, then the rest by frequency.
    ordered = []
    for pref in PREFERRED_TASKS:
        ordered += [t for t in by_task if pref in t and t not in ordered]
    ordered += sorted(
        (t for t in by_task if t not in ordered and t != "unknown"),
        key=lambda t: -len(by_task[t]),
    )

    chosen, seen = [], set()
    # Round-robin over the tasks so the dataset is mixed.
    while len(chosen) < N_CARDS and any(by_task[t] for t in ordered):
        for t in ordered:
            while by_task[t] and by_task[t][0]["modelId"] in seen:
                by_task[t].pop(0)
            if by_task[t]:
                e = by_task[t].pop(0)
                chosen.append(e)
                seen.add(e["modelId"])
                if len(chosen) == N_CARDS:
                    break
    # If there are too few tasks: fill up with the best remaining entries.
    for e in candidates:
        if len(chosen) >= N_CARDS:
            break
        if e["modelId"] not in seen:
            chosen.append(e)
            seen.add(e["modelId"])
    return chosen, len(candidates)


def main():
    """Write the selected cards to OUT_DIR (replacing old ones) and list them."""
    entries = json.loads(SOURCE.read_text(encoding="utf-8"))
    chosen, n_candidates = select(entries)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for e in chosen:
        (OUT_DIR / f"{slug(e['modelId'])}.md").write_text(
            to_markdown(e), encoding="utf-8"
        )

    print(f"{len(entries)} entries in total, {n_candidates} with enough text.")
    print(f"{len(chosen)} cards written to {OUT_DIR.relative_to(ROOT)}/:\n")
    for e in chosen:
        print(
            f"- {slug(e['modelId'])}.md | {e.get('license') or '?'} | "
            f"{', '.join(tasks_of(e)) or '?'}"
        )


if __name__ == "__main__":
    main()
