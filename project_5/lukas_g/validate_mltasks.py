"""Validate task IDs and annotation metadata for every sampled model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from starter.tools.hf_task_taxonomy import HuggingFaceTaskCatalogError, huggingface_tasks

_DATA_FILE = Path(__file__).resolve().parent / "data" / "output" / "sampled_models.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List Hugging Face task IDs and check that every model has exactly one "
            "current ID in its mlTask property and valid mlTaskAnnotation metadata. "
            "Persisted confidence must be MEDIUM or HIGH; LOW is not written."
        )
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the available Hugging Face task IDs and labels.",
    )
    args = parser.parse_args()

    try:
        tasks = huggingface_tasks()
    except HuggingFaceTaskCatalogError as exc:
        parser.exit(2, f"{exc}\n")

    if args.list_tasks:
        print("Current Hugging Face task IDs (store the ID, not its label):")
        for task_id, label in sorted(tasks.items()):
            print(f"  {task_id}: {label}")

    try:
        with _DATA_FILE.open(encoding="utf-8") as data_file:
            records = json.load(data_file)
    except (OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Could not read {_DATA_FILE}: {exc}\n")

    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        parser.exit(2, "Dataset must be a JSON array of model objects.\n")

    issues: list[str] = []
    seen_model_ids: set[str] = set()
    for index, record in enumerate(records):
        model_id = record.get("modelId")
        display_id = model_id if isinstance(model_id, str) and model_id else f"record #{index + 1}"
        if not isinstance(model_id, str) or not model_id.strip():
            issues.append(f"{display_id}: missing modelId")
        elif model_id in seen_model_ids:
            issues.append(f"{display_id}: duplicate modelId")
        else:
            seen_model_ids.add(model_id)

        if "mltask" in record:
            issues.append(f"{display_id}: use the mlTask property, not legacy mltask")
        task = record.get("mlTask", "")
        if not isinstance(task, str) or not task.strip():
            issues.append(f"{display_id}: missing mlTask")
        elif task not in tasks:
            issues.append(
                f"{display_id}: {task!r} is not exactly one current Hugging Face task ID"
            )
        else:
            annotation = record.get("mlTaskAnnotation")
            if not isinstance(annotation, dict):
                issues.append(f"{display_id}: missing or invalid mlTaskAnnotation object")
            else:
                confidence = annotation.get("confidence")
                if not isinstance(confidence, str) or confidence not in {"MEDIUM", "HIGH"}:
                    issues.append(
                        f"{display_id}: mlTaskAnnotation confidence must be MEDIUM or HIGH; LOW must not be persisted"
                    )
                if confidence == "MEDIUM":
                    publication_search = annotation.get("publicationSearch")
                    if not isinstance(publication_search, dict):
                        issues.append(
                            f"{display_id}: MEDIUM confidence requires publicationSearch metadata"
                        )
                    else:
                        query = publication_search.get("query")
                        summary = publication_search.get("summary")
                        if not isinstance(query, str) or not query.strip():
                            issues.append(
                                f"{display_id}: MEDIUM publicationSearch query must be non-empty"
                            )
                        if not isinstance(summary, str) or len(summary.strip()) < 20:
                            issues.append(
                                f"{display_id}: MEDIUM publicationSearch summary must contain at least 20 characters"
                            )

                reasoning = annotation.get("reasoning")
                if not isinstance(reasoning, str) or len(reasoning.strip()) < 30:
                    issues.append(
                        f"{display_id}: mlTaskAnnotation reasoning must contain at least 30 characters"
                    )

                source = annotation.get("source")
                if source is not None and (
                    not isinstance(source, str) or not source.strip() or len(source) > 500
                ):
                    issues.append(
                        f"{display_id}: mlTaskAnnotation source must be a non-empty string of at most 500 characters"
                    )

    if issues:
        print(f"\nValidation failed: {len(issues)} issue(s) across {len(records)} records.")
        for issue in issues:
            print(f"- {issue}")
        print(
            "\nAssign exactly one evidence-supported Hugging Face task ID per model and "
            "include valid mlTaskAnnotation metadata."
        )
        return 1

    print(
        f"\nValidation passed: all {len(records)} models have one current Hugging Face task ID "
        "and valid mlTaskAnnotation metadata."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
