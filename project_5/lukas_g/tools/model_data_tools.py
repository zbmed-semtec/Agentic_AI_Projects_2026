"""Read and cautiously annotate ML-task labels in the sampled model dataset."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from starter.tools.hf_task_taxonomy import (
    HuggingFaceTaskCatalogError,
    huggingface_tasks,
)
from starter.tools.publication_search_state import (
    clear_publication_search_completed,
    completed_publication_search_query,
    completed_publication_search_queries,
)


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_INPUT_FILE = _DATA_DIR / "input" / "sampled_models.json"
_OUTPUT_FILE = _DATA_DIR / "output" / "sampled_models.json"
# Kept as a separate name to make the annotation destination explicit in tool I/O.
_DATA_FILE = _OUTPUT_FILE
_WRITE_LOCK = threading.RLock()
_VALID_CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
_VALID_METADATA_CONFIDENCE_LEVELS = {"MEDIUM", "HIGH"}


def _has_valid_metadata_annotation(annotation: object) -> bool:
    """Check that one metadata value has its own confidence, source, and reasoning."""
    return (
        isinstance(annotation, dict)
        and annotation.get("confidence") in _VALID_METADATA_CONFIDENCE_LEVELS
        and isinstance(annotation.get("source"), str)
        and bool(annotation["source"].strip())
        and isinstance(annotation.get("reasoning"), str)
        and len(annotation["reasoning"].strip()) >= 30
    )


def initialize_output_dataset() -> Path:
    """Copy the source dataset into data/output before each annotation run."""
    with _WRITE_LOCK:
        with _INPUT_FILE.open("r", encoding="utf-8") as input_file:
            records = json.load(input_file)
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError("The input model data must be a JSON array of objects.")

        _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_mode = stat.S_IMODE(_INPUT_FILE.stat().st_mode)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=_OUTPUT_FILE.parent,
                prefix=f"{_OUTPUT_FILE.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                json.dump(records, temporary_file, ensure_ascii=False, indent=4)
                temporary_file.write("\n")
            os.chmod(temporary_path, file_mode)
            os.replace(temporary_path, _OUTPUT_FILE)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
    return _OUTPUT_FILE


def _load_records() -> list[dict]:
    # Unit/direct tool use before a CLI run falls back to the input dataset.
    data_path = _DATA_FILE if _DATA_FILE.exists() else _INPUT_FILE
    with data_path.open(encoding="utf-8") as data_file:
        records = json.load(data_file)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("The sampled model data must be a JSON array of objects.")
    return records


def _ml_task(record: dict) -> str:
    """Read either the current spelling or the legacy lowercase key."""
    value = record.get("mlTask", record.get("mltask", ""))
    return value if isinstance(value, str) else ""


def _has_valid_task_annotation(record: dict) -> bool:
    if "mlTaskAnnotation" in record:
        return False
    metadata_annotations = record.get("modelMetadataAnnotations")
    if isinstance(metadata_annotations, dict) and "mlTaskAnnotation" in metadata_annotations:
        return False
    annotation = (
        metadata_annotations.get("mlTask")
        if isinstance(metadata_annotations, dict)
        else None
    )
    if not isinstance(annotation, dict):
        return False

    confidence = annotation.get("confidence")
    reasoning = annotation.get("reasoning")
    source = annotation.get("source")
    valid_base = (
        isinstance(confidence, str)
        and confidence in {"MEDIUM", "HIGH"}
        and isinstance(reasoning, str)
        and len(reasoning.strip()) >= 30
        and (
            source is None
            or isinstance(source, str) and bool(source.strip()) and len(source) <= 500
        )
    )
    if not valid_base:
        return False
    if confidence == "MEDIUM":
        publication_search = annotation.get("publicationSearch")
        return (
            isinstance(publication_search, dict)
            and isinstance(publication_search.get("query"), str)
            and bool(publication_search["query"].strip())
            and isinstance(publication_search.get("summary"), str)
            and len(publication_search["summary"].strip()) >= 20
        )
    return True


class ReadSampledModelsInput(BaseModel):
    model_id: str | None = Field(
        default=None,
        description="Exact modelId to inspect; leave blank to browse by start_index.",
    )
    start_index: int = Field(
        default=0, ge=0, description="Zero-based position among matching records"
    )
    limit: int = Field(default=1, ge=1, le=3, description="Number of records to return (1-3)")
    only_missing: bool = Field(
        default=True,
        description=(
            "When browsing, return only records whose mlTask is unsupported or whose "
            "mlTaskAnnotation is missing or invalid, or whose parameterCount, baseModel, "
            "trainingData, or trainingTokens value is blank."
        ),
    )


class ReadSampledModelsTool(BaseTool):
    name: str = "read_sampled_models"
    description: str = (
        "Read data/output/sampled_models.json (initialized from data/input at run start). "
        "Returns complete model records and their existing fields "
        "so you can assess intendedUse, keywords, descriptions, metadata, and publication "
        "references before deciding whether an mlTask is supported. Browse at most 3 records "
        "per call, or provide an exact modelId. This tool never changes the data."
    )
    args_schema: Type[BaseModel] = ReadSampledModelsInput

    def _run(
        self,
        model_id: str | None = None,
        start_index: int = 0,
        limit: int = 1,
        only_missing: bool = True,
    ) -> str:
        try:
            records = _load_records()
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"Could not read sampled model data: {exc}"

        if model_id:
            matches = [row for row in records if row.get("modelId") == model_id.strip()]
            if len(matches) != 1:
                return f"Expected one record for modelId {model_id!r}; found {len(matches)}."
            selected = matches
        else:
            if only_missing:
                try:
                    valid_tasks = huggingface_tasks()
                except HuggingFaceTaskCatalogError as exc:
                    return str(exc)
                candidates = [
                    row for row in records
                    if _ml_task(row).strip() not in valid_tasks
                    or not _has_valid_task_annotation(row)
                    or any(
                        not isinstance(row.get(field), str) or not row[field].strip()
                        for field in (
                            "parameterCount",
                            "baseModel",
                            "trainingData",
                            "trainingTokens",
                        )
                    )
                ]
            else:
                candidates = records
            selected = candidates[start_index : start_index + limit]

        if not selected:
            return "No matching model records found."

        # Normalize the legacy key in the returned view without mutating the file.
        result = []
        for row in selected:
            view = dict(row)
            view["mlTask"] = _ml_task(row)
            view.pop("mltask", None)
            result.append(view)
        return json.dumps(result, ensure_ascii=False, indent=2)


class WriteModelMLTaskInput(BaseModel):
    model_id: str = Field(description="Exact modelId from read_sampled_models")
    ml_task: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Exactly one task ID from list_huggingface_tasks, e.g. text-classification; "
            "do not provide a label description or multiple tasks"
        ),
    )
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description=(
            "Three-level confidence rating: LOW, MEDIUM, or HIGH. LOW annotations "
            "must not be written; MEDIUM requires a completed publication search."
        ),
    )
    source: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional provenance for the annotation, such as model fields, a publication "
            "title/DOI, or a URL actually consulted"
        ),
    )
    publication_search_summary: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "For MEDIUM confidence, summarize the publication-search outcome and how it "
            "affected the task decision; required after search_publications."
        ),
    )
    reasoning: str = Field(
        min_length=30,
        description=(
            "Explain how the observed evidence supports this exact Hugging Face task ID; "
            "do not claim sources that were not consulted"
        ),
    )


class WriteModelMLTaskTool(BaseTool):
    name: str = "write_model_mltask"
    description: str = (
        "Set exactly one mlTask for an exact modelId. The value must be an ID currently returned "
        "by list_huggingface_tasks, not a descriptive phrase or a list. Writes only with MEDIUM "
        "or HIGH confidence and reasoning; LOW confidence is never persisted. "
        "MEDIUM is accepted only after search_publications has completed for this modelId. "
        "For MEDIUM, include a summary of the search findings and their effect on the decision. "
        "Also stores task confidence, optional source, reasoning, and publicationSearch (when "
        "required) at modelMetadataAnnotations.mlTask. Do not create a separate top-level "
        "annotation property or use mlTaskAnnotation as the nested key. "
        "It will not replace an existing valid task ID, but can add "
        "missing annotation metadata when given that same ID; it can migrate an existing "
        "mlTaskAnnotation annotation key to mlTask; it can replace a legacy free-text "
        "value with a supported ID."
    )
    args_schema: Type[BaseModel] = WriteModelMLTaskInput

    def _run(
        self,
        model_id: str,
        ml_task: str,
        confidence: Literal["LOW", "MEDIUM", "HIGH"],
        reasoning: str,
        source: str | None = None,
        publication_search_summary: str | None = None,
    ) -> str:
        model_id = model_id.strip()
        ml_task = ml_task.strip()
        reasoning = reasoning.strip()
        source = source.strip() if source is not None else None
        if source == "":
            source = None
        publication_search_summary = (
            publication_search_summary.strip()
            if publication_search_summary is not None
            else None
        )
        if not model_id or not ml_task:
            return "No change: modelId and mlTask must be non-empty."
        if not isinstance(confidence, str) or confidence not in _VALID_CONFIDENCE_LEVELS:
            return "No change: confidence must be LOW, MEDIUM, or HIGH."
        if confidence == "LOW":
            return "No change: LOW-confidence annotations must not be added to the records."
        publication_search_query = None
        if confidence == "MEDIUM":
            publication_search_query = completed_publication_search_query(model_id)
            if publication_search_query is None:
                return (
                    "No change: MEDIUM confidence requires a completed search_publications call "
                    "for this exact modelId before writing."
                )
            if not publication_search_summary or len(publication_search_summary) < 20:
                return (
                    "No change: MEDIUM confidence requires a publication-search summary of at "
                    "least 20 characters, explaining the findings and their effect on the decision."
                )
            if len(publication_search_summary) > 1000:
                return "No change: publication-search summary must be no longer than 1000 characters."
        if len(reasoning) < 30:
            return "No change: provide concrete annotation reasoning of at least 30 characters."
        if source is not None and len(source) > 500:
            return "No change: source must be no longer than 500 characters."

        try:
            valid_tasks = huggingface_tasks()
        except HuggingFaceTaskCatalogError as exc:
            return f"No change: {exc}"
        if ml_task not in valid_tasks:
            return (
                f"No change: {ml_task!r} is not a Hugging Face task ID. "
                "Use list_huggingface_tasks and provide exactly one listed ID."
            )

        temporary_path: str | None = None
        try:
            with _WRITE_LOCK:
                records = _load_records()
                matches = [row for row in records if row.get("modelId") == model_id]
                if len(matches) != 1:
                    return f"No change: expected one record for {model_id!r}; found {len(matches)}."
                record = matches[0]
                current_task = _ml_task(record).strip()
                if current_task in valid_tasks:
                    if current_task != ml_task:
                        return (
                            f"No change: {model_id!r} already has valid Hugging Face task "
                            f"{current_task!r}; it will not be overwritten."
                        )
                    if _has_valid_task_annotation(record):
                        return f"No change: {model_id!r} already has valid task metadata annotation."
                else:
                    record["mlTask"] = ml_task
                    record.pop("mltask", None)

                annotation = {"confidence": confidence}
                if source is not None:
                    annotation["source"] = source
                if confidence == "MEDIUM":
                    annotation["publicationSearch"] = {
                        "query": publication_search_query,
                        "summary": publication_search_summary,
                    }
                annotation["reasoning"] = reasoning
                metadata_annotations = record.get("modelMetadataAnnotations")
                if not isinstance(metadata_annotations, dict):
                    metadata_annotations = {}
                metadata_annotations.pop("mlTaskAnnotation", None)
                metadata_annotations["mlTask"] = annotation
                record["modelMetadataAnnotations"] = metadata_annotations
                record.pop("mlTaskAnnotation", None)

                file_mode = stat.S_IMODE(_DATA_FILE.stat().st_mode)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=_DATA_FILE.parent,
                    prefix=f"{_DATA_FILE.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = temporary_file.name
                    json.dump(records, temporary_file, ensure_ascii=False, indent=4)
                    temporary_file.write("\n")
                os.chmod(temporary_path, file_mode)
                os.replace(temporary_path, _DATA_FILE)
                temporary_path = None
                if publication_search_query is not None:
                    clear_publication_search_completed(model_id, publication_search_query)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"Could not save mlTask: {exc}"
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

        return (
            f"Saved mlTask={ml_task!r} ({valid_tasks[ml_task]}) for {model_id!r} "
            f"with modelMetadataAnnotations.mlTask (confidence {confidence})."
        )


class WriteModelMetadataInput(BaseModel):
    model_id: str = Field(description="Exact modelId from read_sampled_models")
    parameter_count: str | None = Field(
        default=None,
        max_length=100,
        description="Exact documented parameter count; do not estimate from file size or architecture.",
    )
    base_model: str | None = Field(
        default=None,
        max_length=500,
        description="Documented identifier or name of the pretrained/base model, if any.",
    )
    training_data: str | None = Field(
        default=None,
        max_length=1000,
        description="Documented training dataset name and/or source URL.",
    )
    training_tokens: str | None = Field(
        default=None,
        max_length=100,
        description="Exact documented number of training tokens; do not infer or estimate.",
    )
    parameter_count_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = Field(default=None)
    parameter_count_source: str | None = Field(default=None, max_length=500)
    parameter_count_reasoning: str | None = Field(default=None, min_length=30, max_length=2000)
    parameter_count_publication_search_summary: str | None = Field(default=None, max_length=1000)
    base_model_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = Field(default=None)
    base_model_source: str | None = Field(default=None, max_length=500)
    base_model_reasoning: str | None = Field(default=None, min_length=30, max_length=2000)
    base_model_publication_search_summary: str | None = Field(default=None, max_length=1000)
    training_data_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = Field(default=None)
    training_data_source: str | None = Field(default=None, max_length=500)
    training_data_reasoning: str | None = Field(default=None, min_length=30, max_length=2000)
    training_data_publication_search_summary: str | None = Field(default=None, max_length=1000)
    training_tokens_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = Field(default=None)
    training_tokens_source: str | None = Field(default=None, max_length=500)
    training_tokens_reasoning: str | None = Field(default=None, min_length=30, max_length=2000)
    training_tokens_publication_search_summary: str | None = Field(default=None, max_length=1000)


class WriteModelMetadataTool(BaseTool):
    name: str = "write_model_metadata"
    description: str = (
        "Fill any evidence-backed values among parameterCount, baseModel, trainingData, and "
        "trainingTokens for an exact modelId. Supply only values directly supported by the "
        "record or a source actually consulted; omit unknown values. For every supplied value, "
        "include its own confidence (MEDIUM or HIGH), source, and reasoning. MEDIUM confidence "
        "requires a completed search_publications call for that exact modelId and property, plus "
        "a per-field summary of the findings; LOW-confidence values are never written. HIGH does "
        "not require a publication search when record evidence is direct. Existing non-empty values are preserved. Stores per-field details in "
        "modelMetadataAnnotations under the corresponding field names and does not change "
        "mlTask or unrelated properties."
    )
    args_schema: Type[BaseModel] = WriteModelMetadataInput

    def _run(
        self,
        model_id: str,
        parameter_count: str | None = None,
        base_model: str | None = None,
        training_data: str | None = None,
        training_tokens: str | None = None,
        parameter_count_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
        parameter_count_source: str | None = None,
        parameter_count_reasoning: str | None = None,
        parameter_count_publication_search_summary: str | None = None,
        base_model_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
        base_model_source: str | None = None,
        base_model_reasoning: str | None = None,
        base_model_publication_search_summary: str | None = None,
        training_data_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
        training_data_source: str | None = None,
        training_data_reasoning: str | None = None,
        training_data_publication_search_summary: str | None = None,
        training_tokens_confidence: Literal["LOW", "MEDIUM", "HIGH"] | None = None,
        training_tokens_source: str | None = None,
        training_tokens_reasoning: str | None = None,
        training_tokens_publication_search_summary: str | None = None,
    ) -> str:
        model_id = model_id.strip()
        supplied = {
            "parameterCount": (
                parameter_count,
                parameter_count_confidence,
                parameter_count_source,
                parameter_count_reasoning,
                parameter_count_publication_search_summary,
            ),
            "baseModel": (
                base_model,
                base_model_confidence,
                base_model_source,
                base_model_reasoning,
                base_model_publication_search_summary,
            ),
            "trainingData": (
                training_data,
                training_data_confidence,
                training_data_source,
                training_data_reasoning,
                training_data_publication_search_summary,
            ),
            "trainingTokens": (
                training_tokens,
                training_tokens_confidence,
                training_tokens_source,
                training_tokens_reasoning,
                training_tokens_publication_search_summary,
            ),
        }
        property_search_terms = {
            "parameterCount": ("parameter", "parameters", "parameter count", "params"),
            "baseModel": ("base model", "pretrained", "parent model"),
            "trainingData": ("training data", "dataset"),
            "trainingTokens": ("training token", "token count", "tokens"),
        }
        completed_queries = completed_publication_search_queries(model_id)
        updates = {}
        used_search_queries: set[str] = set()
        for key, (value, confidence, source, reasoning, search_summary) in supplied.items():
            if not isinstance(value, str) or not value.strip():
                if any(item is not None for item in (confidence, source, reasoning, search_summary)):
                    return f"No change: {key} annotation requires a non-empty value."
                continue
            value = value.strip()
            source = source.strip() if isinstance(source, str) else ""
            reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
            if confidence == "LOW":
                return f"No change: LOW-confidence values for {key} must not be written."
            if confidence not in {"MEDIUM", "HIGH"}:
                return f"No change: provide MEDIUM or HIGH confidence for {key}."
            if not source:
                return f"No change: provide a source for {key}."
            if len(reasoning) < 30:
                return f"No change: provide at least 30 characters of reasoning for {key}."
            publication_search = None
            if confidence == "MEDIUM":
                matching_query = next(
                    (
                        query
                        for query in completed_queries
                        if any(term in query.casefold() for term in property_search_terms[key])
                    ),
                    None,
                )
                if matching_query is None:
                    return (
                        f"No change: MEDIUM confidence for {key} requires a completed "
                        "search_publications call whose query includes this modelId and "
                        "searches for this property."
                    )
                if not isinstance(search_summary, str) or len(search_summary.strip()) < 20:
                    return (
                        f"No change: MEDIUM confidence for {key} requires a publication-search "
                        "summary of at least 20 characters."
                    )
                publication_search = {
                    "query": matching_query,
                    "summary": search_summary.strip(),
                }
                used_search_queries.add(matching_query)
            updates[key] = {
                "value": value,
                "annotation": {
                    "confidence": confidence,
                    "source": source,
                    "reasoning": reasoning,
                    **({"publicationSearch": publication_search} if publication_search else {}),
                },
            }
        if not model_id:
            return "No change: modelId must be non-empty."
        if not updates:
            return "No change: provide at least one non-empty metadata value."

        temporary_path: str | None = None
        try:
            with _WRITE_LOCK:
                records = _load_records()
                matches = [row for row in records if row.get("modelId") == model_id]
                if len(matches) != 1:
                    return f"No change: expected one record for {model_id!r}; found {len(matches)}."
                record = matches[0]
                saved_fields = []
                preserved_fields = []
                annotations = record.get("modelMetadataAnnotations")
                if not isinstance(annotations, dict):
                    annotations = {}
                changed_annotations = False
                annotated_fields = []
                for key, details in updates.items():
                    value = details["value"]
                    current = record.get(key)
                    has_current_value = current is not None and (
                        not isinstance(current, str) or bool(current.strip())
                    )
                    if has_current_value:
                        if (
                            isinstance(current, str)
                            and current.strip() == value
                            and not _has_valid_metadata_annotation(annotations.get(key))
                        ):
                            annotations[key] = details["annotation"]
                            changed_annotations = True
                            annotated_fields.append(key)
                        else:
                            preserved_fields.append(key)
                    else:
                        record[key] = value
                        annotations[key] = details["annotation"]
                        saved_fields.append(key)
                        changed_annotations = True
                        annotated_fields.append(key)

                if not saved_fields and not changed_annotations:
                    return (
                        f"No change: all supplied metadata fields already have values for "
                        f"{model_id!r}; existing values were preserved."
                    )
                if changed_annotations:
                    record["modelMetadataAnnotations"] = annotations

                file_mode = stat.S_IMODE(_DATA_FILE.stat().st_mode)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=_DATA_FILE.parent,
                    prefix=f"{_DATA_FILE.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_path = temporary_file.name
                    json.dump(records, temporary_file, ensure_ascii=False, indent=4)
                    temporary_file.write("\n")
                os.chmod(temporary_path, file_mode)
                os.replace(temporary_path, _DATA_FILE)
                temporary_path = None
                for query in used_search_queries:
                    clear_publication_search_completed(model_id, query)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return f"Could not save model metadata: {exc}"
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass

        result_parts = []
        if saved_fields:
            result_parts.append(f"Saved values for {', '.join(saved_fields)}")
        if annotated_fields:
            result_parts.append(f"saved per-field annotations for {', '.join(annotated_fields)}")
        result = f"{'; '.join(result_parts)} for {model_id!r}."
        if preserved_fields:
            result += f" Preserved existing values for {', '.join(preserved_fields)}."
        return result
