"""
SupportGraph AI — Groq Suggest Golden Labels CLI Script (Phase 5.5)

Generates AI-assisted intent suggestions for the Golden Evaluation Set:
- Advisory suggestions only: never writes to annotation_label or annotator
- Preserves annotation_label = "" and annotation_status = "pending_human_review"
- Supports --dry-run for safe verification
- Supports --resume for graceful continuation after rate limits or interruptions
- Supports --limit and --start-index for controlled smoke tests

Usage:
    # Dry run verification
    python -m backend.scripts.suggest_golden_labels --dry-run

    # Small smoke-test batch (5 records)
    python -m backend.scripts.suggest_golden_labels --limit 5

    # Resume full processing
    python -m backend.scripts.suggest_golden_labels --resume
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from app.core.logging import configure_logging
    from app.evaluation.annotation_assistant import (
        DEFAULT_GROQ_MODEL,
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
        LLMAuthenticationError,
    )
    from app.evaluation.human_review import (
        is_suggestion_complete,
        is_valid_ai_suggestion,
    )
    from app.evaluation.label_validation import validate_required_columns
    from app.nlp.taxonomy_finalization import DEFAULT_CANDIDATE_TAXONOMY_PATH
except ModuleNotFoundError:
    from backend.app.core.logging import configure_logging  # type: ignore[no-redef]
    from backend.app.evaluation.annotation_assistant import (  # type: ignore[no-redef]
        DEFAULT_GROQ_MODEL,
        AISuggestion,
        AnnotationAssistant,
        GroqAnnotationAssistant,
        GroqAuthenticationError,
        LLMAuthenticationError,
    )
    from backend.app.evaluation.human_review import (  # type: ignore[no-redef]
        is_suggestion_complete,
        is_valid_ai_suggestion,
    )
    from backend.app.evaluation.label_validation import (  # type: ignore[no-redef]
        validate_required_columns,
    )
    from backend.app.nlp.taxonomy_finalization import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_TAXONOMY_PATH,
    )

logger = logging.getLogger("suggest_golden_labels")

DEFAULT_INPUT_CSV = "data/golden/golden_set_annotation_template.csv"
DEFAULT_OUTPUT_CSV = "data/golden/golden_set_ai_suggestions.csv"

SUGGESTION_COLUMNS = [
    "model_name",
    "model_suggested_label",
    "model_confidence",
    "model_reasoning_summary",
    "model_needs_human_review",
    "suggestion_timestamp",
    "suggestion_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5.5 AI-Assisted Golden Set Intent Suggestion Generator"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "ollama"],
        default=None,
        help="LLM provider ('groq' or 'ollama'). Defaults to LLM_PROVIDER from settings.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help=f"Input golden set template CSV (default: {DEFAULT_INPUT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output suggestions CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process (e.g. 5 for smoke test)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-indexed start position within eligible rows",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier override (e.g. 'openai/gpt-oss-20b' for groq, 'llama3.2:latest' for ollama)",
    )
    parser.add_argument(
        "--taxonomy-path",
        type=str,
        default=DEFAULT_CANDIDATE_TAXONOMY_PATH,
        help="Path to candidate taxonomy JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment, input schema, and taxonomy without invoking API or saving",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing existing output file, skipping already generated suggestions",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def run_dry_run(
    input_path: Path,
    output_path: Path,
    taxonomy_path: Path,
    assistant: AnnotationAssistant,
) -> int:
    """Execute dry-run validation checks without modifying dataset or calling LLM."""
    logger.info("=" * 60)
    logger.info("PHASE 5.5 — DRY RUN VALIDATION (%s)", assistant.provider.upper())
    logger.info("=" * 60)

    # 1. Input file check
    if not input_path.exists():
        logger.error("[FAIL] Input file does not exist: %s", input_path)
        return 1
    df = pd.read_csv(input_path, dtype=str).fillna("")
    col_errors = validate_required_columns(df)
    if col_errors:
        logger.error("[FAIL] Input file failed schema validation: %s", col_errors)
        return 1
    logger.info("[PASS] Input template exists: %s (%d records)", input_path, len(df))

    # 2. Taxonomy check
    if not taxonomy_path.exists():
        logger.warning("[WARN] Taxonomy candidate JSON not found at %s. Default fallback active.", taxonomy_path)
    else:
        logger.info("[PASS] Taxonomy candidate JSON verified: %d operational intents", len(assistant.allowed_labels))
    logger.info("       Allowed intents: %s", assistant.allowed_labels)

    # 3. LLM provider & access check
    logger.info("[PASS] LLM Provider: %s", assistant.provider.upper())
    logger.info("[PASS] Selected model: %s", assistant.model)
    if assistant.provider == "groq":
        if not assistant.api_key:
            logger.warning("[WARN] GROQ_API_KEY is not set or is empty in backend/.env.")
            logger.warning("       Model suggestions cannot run until GROQ_API_KEY is configured.")
        else:
            logger.info("[PASS] GROQ_API_KEY detected (redacted for security, length=%d)", len(assistant.api_key))

    is_valid, available_models, err_msg = assistant.verify_model_access()
    if is_valid:
        logger.info("[PASS] Configured model '%s' is confirmed accessible.", assistant.model)
        logger.info("       Total accessible models on account/server: %d", len(available_models))
    else:
        logger.warning("[WARN] Model verification issue: %s", err_msg)
        if available_models:
            logger.warning("       Accessible models: %s", available_models)

    # 4. Output destination check
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[PASS] Output destination valid: %s", output_path)

    # 5. Scientific integrity confirmation
    logger.info("-" * 60)
    logger.info("SCIENTIFIC INTEGRITY VERIFICATION:")
    logger.info(" - annotation_label will remain strictly EMPTY ('')")
    logger.info(" - annotator will remain strictly EMPTY ('')")
    logger.info(" - annotation_status will be initialized to 'pending_human_review'")
    logger.info(" - AI suggestions are purely advisory for human annotators")
    logger.info("-" * 60)
    logger.info("DRY RUN PASSED: Environment and schema conform to Phase 5.5 requirements.")
    return 0


def main() -> int:
    args = parse_args()
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level=log_level)

    input_path = Path(args.input)
    output_path = Path(args.output)
    taxonomy_path = Path(args.taxonomy_path)

    assistant = AnnotationAssistant(
        provider=args.provider,
        model=args.model,
        taxonomy_path=taxonomy_path,
    )

    if args.dry_run:
        return run_dry_run(input_path, output_path, taxonomy_path, assistant)

    logger.info("=" * 60)
    logger.info("PHASE 5.5 — %s GOLDEN SET SUGGESTION GENERATOR", assistant.provider.upper())
    logger.info("=" * 60)
    logger.info("Active LLM Provider: %s", assistant.provider)
    logger.info("Configured Model:    %s", assistant.model)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    if assistant.provider == "groq" and not assistant.api_key:
        logger.error("GROQ_API_KEY is not configured in backend/.env. Cannot proceed with suggestions.")
        logger.error("Please add your GROQ_API_KEY to backend/.env and re-run.")
        return 1

    # Verify model access before processing data
    is_valid, available_models, err_msg = assistant.verify_model_access()
    if not is_valid:
        logger.error("=" * 60)
        logger.error("MODEL VERIFICATION FAILED (%s):", assistant.provider.upper())
        logger.error("Error: %s", err_msg)
        logger.error("Configured model: %s", assistant.model)
        logger.error("Available models (%d):", len(available_models))
        for mid in available_models:
            logger.error("  - %s", mid)
        logger.error("=" * 60)
        logger.error("Aborting without generating suggestions. No fake records will be written.")
        return 1

    logger.info("[PASS] Model verified accessible: %s (%s)", assistant.model, assistant.provider)

    # Load or initialize dataframe
    TEXT_COLUMNS = [
        "golden_id",
        "conversation_id",
        "customer_message",
        "normalized_message",
        "candidate_intent",
        "conversation_context",
        "annotation_label",
        "annotation_status",
        "annotator",
        "notes",
        "model_name",
        "model_suggested_label",
        "model_reasoning_summary",
        "suggestion_status",
    ]

    if args.resume and output_path.exists():
        logger.info("Resuming from existing suggestions file: %s", output_path)
        df = pd.read_csv(output_path, dtype=str)
    else:
        logger.info("Loading input template from: %s", input_path)
        df = pd.read_csv(input_path, dtype=str)

    # Clean text columns, coerce to object dtype, eliminate float NaNs and literal 'nan'
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(object)
            df[col] = df[col].apply(lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip())

    for col in SUGGESTION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
        if col != "model_confidence":
            df[col] = df[col].apply(lambda x: "" if str(x).strip().lower() in ("nan", "none", "null", "undefined") else str(x).strip())

    # Ensure annotation columns are explicitly object dtype to avoid FutureWarnings on assignment
    df["annotation_label"] = df["annotation_label"].astype(object)
    df["annotator"] = df["annotator"].astype(object)
    df["annotation_status"] = df["annotation_status"].astype(object)
    df["notes"] = df["notes"].astype(object)

    total_rows = len(df)

    # Categorize records for resume & reporting
    valid_completed_indices: list[int] = []
    missing_indices: list[int] = []
    invalid_indices: list[int] = []
    human_reviewed_indices: list[int] = []

    for idx in range(total_rows):
        val = str(df.at[idx, "model_suggested_label"]).strip()
        status = str(df.at[idx, "suggestion_status"]).strip()
        h_label = str(df.at[idx, "annotation_label"]).strip()
        h_status = str(df.at[idx, "annotation_status"]).strip().lower()

        if h_label and h_status not in ("", "pending", "pending_human_review"):
            human_reviewed_indices.append(idx)

        if is_valid_ai_suggestion(val, allowed_labels=assistant.all_valid_labels, suggestion_status=status):
            valid_completed_indices.append(idx)
        elif not val or val.lower() in ("nan", "none", "null", "undefined"):
            missing_indices.append(idx)
        else:
            invalid_indices.append(idx)

    # Pending suggestions to process are missing + invalid
    indices_to_process = missing_indices + invalid_indices

    logger.info("=" * 60)
    logger.info("DATASET STATUS SUMMARY:")
    logger.info("  Total dataset records:                    %d", total_rows)
    logger.info("  Valid completed AI suggestions:            %d", len(valid_completed_indices))
    logger.info("  Pending suggestions to process:           %d", len(indices_to_process))
    logger.info("    - Missing suggestions:                  %d", len(missing_indices))
    logger.info("    - Invalid suggestions to regenerate:     %d", len(invalid_indices))
    logger.info("  Human reviewed records:                   %d", len(human_reviewed_indices))
    logger.info("  Pending human review records:             %d", total_rows - len(human_reviewed_indices))
    logger.info("=" * 60)

    # Apply start-index and limit
    if args.start_index > 0:
        indices_to_process = indices_to_process[args.start_index :]
    if args.limit is not None and args.limit > 0:
        indices_to_process = indices_to_process[: args.limit]

    if not indices_to_process:
        logger.info("No records pending AI suggestions. Output file is already up to date.")
        return 0

    logger.info(
        "Processing batch of %d records using model: %s (%s)",
        len(indices_to_process),
        assistant.model,
        assistant.provider,
    )
    logger.info("-" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_in_run = 0

    for step_num, row_idx in enumerate(indices_to_process, start=1):
        row = df.iloc[row_idx]
        gid = row.get("golden_id", f"row_{row_idx}")
        cust_msg = str(row.get("customer_message", "")).strip()

        logger.info(
            "[%d/%d] Processing %s: \"%.60s...\"",
            step_num,
            len(indices_to_process),
            gid,
            cust_msg,
        )

        try:
            suggestion = assistant.suggest_for_record(row)
        except (GroqAuthenticationError, LLMAuthenticationError) as auth_err:
            logger.error("Authentication failed: %s. Stopping batch.", auth_err)
            return 1
        except Exception as exc:
            logger.warning("Error processing %s: %s", gid, exc)
            now_utc = datetime.now(timezone.utc).isoformat()
            suggestion = AISuggestion(
                model_name=assistant.model,
                model_suggested_label="",
                model_confidence=None,
                model_reasoning_summary="",
                model_needs_human_review=True,
                suggestion_timestamp=now_utc,
                suggestion_status="failed",
            )

        # Write suggestion fields
        if suggestion.suggestion_status == "failed":
            df.at[row_idx, "model_name"] = suggestion.model_name
            df.at[row_idx, "model_suggested_label"] = ""
            df.at[row_idx, "model_confidence"] = ""
            df.at[row_idx, "model_reasoning_summary"] = suggestion.model_reasoning_summary or "Provider error"
            df.at[row_idx, "model_needs_human_review"] = True
            df.at[row_idx, "suggestion_timestamp"] = suggestion.suggestion_timestamp
            df.at[row_idx, "suggestion_status"] = "failed"
        elif suggestion.suggestion_status == "invalid_model_output":
            df.at[row_idx, "model_name"] = suggestion.model_name
            df.at[row_idx, "model_suggested_label"] = suggestion.model_suggested_label
            df.at[row_idx, "model_confidence"] = 0.0
            df.at[row_idx, "model_reasoning_summary"] = suggestion.model_reasoning_summary
            df.at[row_idx, "model_needs_human_review"] = True
            df.at[row_idx, "suggestion_timestamp"] = suggestion.suggestion_timestamp
            df.at[row_idx, "suggestion_status"] = "invalid_model_output"
        else:
            df.at[row_idx, "model_name"] = suggestion.model_name
            df.at[row_idx, "model_suggested_label"] = suggestion.model_suggested_label
            df.at[row_idx, "model_confidence"] = (
                suggestion.model_confidence if suggestion.model_confidence is not None else ""
            )
            df.at[row_idx, "model_reasoning_summary"] = suggestion.model_reasoning_summary
            df.at[row_idx, "model_needs_human_review"] = suggestion.model_needs_human_review
            df.at[row_idx, "suggestion_timestamp"] = suggestion.suggestion_timestamp
            df.at[row_idx, "suggestion_status"] = suggestion.suggestion_status

        # Strictly preserve existing human annotation fields if already reviewed; otherwise keep empty and pending
        existing_status = str(df.at[row_idx, "annotation_status"]).strip().lower()
        existing_label = str(df.at[row_idx, "annotation_label"]).strip()
        if not existing_label or existing_status in ("", "pending", "pending_human_review"):
            df.at[row_idx, "annotation_label"] = ""
            df.at[row_idx, "annotator"] = ""
            df.at[row_idx, "annotation_status"] = "pending_human_review"

        completed_in_run += 1

        # Periodic save after each item to ensure progress is never lost
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

        if suggestion.suggestion_status == "failed":
            logger.warning("       -> Status: FAILED | Label: '' | NeedsReview=True")
        else:
            conf_str = f"{suggestion.model_confidence:.2f}" if suggestion.model_confidence is not None else "N/A"
            logger.info(
                "       -> Suggested: '%s' (conf=%s) | NeedsReview=%s",
                suggestion.model_suggested_label,
                conf_str,
                suggestion.model_needs_human_review,
            )

        # Rate-limiting polite delay
        time.sleep(1.0)

    logger.info("-" * 60)
    logger.info("Batch completed! Saved %d suggestions to: %s", completed_in_run, output_path)
    logger.info("CRITICAL REMINDER: annotation_label remains EMPTY (''). Human review is required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
