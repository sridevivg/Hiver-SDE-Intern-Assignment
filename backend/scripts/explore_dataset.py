"""
SupportGraph AI — Dataset Exploration CLI Script

Usage (from project root):
    python -m backend.scripts.explore_dataset

Outputs:
    data/interim/dataset_schema.json
    data/interim/dataset_statistics.json
    data/interim/account_statistics.csv
    data/interim/brand_candidates.csv          (if brand structure detected)
    artifacts/figures/top_accounts.png
    artifacts/figures/message_length_distribution.png
    artifacts/reports/phase_1_dataset_analysis.md
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a module
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from backend.app.core.config import settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.data.explorer import ExplorationResult, run_exploration

configure_logging(log_level="INFO", log_format="text")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
INTERIM_DIR = settings.interim_data_path
FIGURES_DIR = settings.figures_path
REPORTS_DIR = settings.reports_path


def _ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in [INTERIM_DIR, FIGURES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Terminal printer
# ---------------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_results(result: ExplorationResult) -> None:
    """Print a clean terminal summary of exploration results."""
    s = result.structure
    m = result.message
    a = result.account
    c = result.conversation

    _print_section("A. DATASET STRUCTURE")
    print(f"  File:          {s.filename}")
    print(f"  Size:          {s.file_size_mb:.1f} MB")
    print(f"  Rows:          {s.row_count:,}")
    print(f"  Columns:       {s.column_count}")
    print(f"  Column names:  {s.columns}")
    print()
    print("  Data types:")
    for col, dtype in s.dtypes.items():
        missing_info = ""
        if col in s.missing_by_column:
            m_data = s.missing_by_column[col]
            missing_info = f"  ← {m_data['count']:,} missing ({m_data['pct']:.1f}%)"
        print(f"    {col:<35} {dtype:<12}{missing_info}")
    print()
    print(f"  Duplicate rows: {s.duplicate_count:,} ({s.duplicate_pct:.2f}%)")

    _print_section("B. MESSAGE ANALYSIS")
    if not m.text_column_found:
        print("  ⚠ No text/message column detected.")
    else:
        print(f"  Text column:      '{m.text_column_name}'")
        print(f"  Total messages:   {m.total_messages:,}")
        print(f"  Empty messages:   {m.empty_message_count:,}")
        print("  Length statistics:")
        for k, v in m.length_stats.items():
            print(f"    {k:<10} {v:>8.1f} chars")
        if m.sample_messages:
            print("  Sample messages:")
            for i, msg in enumerate(m.sample_messages, 1):
                preview = textwrap.shorten(msg, width=80, placeholder="…")
                print(f"    [{i}] {preview}")

    _print_section("C. ACCOUNT ANALYSIS")
    if not a.author_column_found:
        print("  ⚠ No author column detected.")
    else:
        print(f"  Author column:    '{a.author_column_name}'")
        print(f"  Unique authors:   {a.total_unique_authors:,}")
        if a.inbound_column_found:
            print(f"  Inbound (customer):  {a.inbound_count:,}")
            print(f"  Outbound (brand):    {a.outbound_count:,}")
        print(f"\n  Top 10 accounts by volume:")
        print(f"  {'Author':<25} {'Total':>8}  {'Inbound':>8}  {'Outbound':>9}")
        print(f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*9}")
        for row in a.top_accounts[:10]:
            inb = row.get("inbound", "–")
            out = row.get("outbound", "–")
            inb_str = f"{inb:>8}" if isinstance(inb, int) else f"{'–':>8}"
            out_str = f"{out:>9}" if isinstance(out, int) else f"{'–':>9}"
            print(f"  {row['author']:<25} {row['total_messages']:>8,}  {inb_str}  {out_str}")

    _print_section("D. CONVERSATION STRUCTURE")
    if c.reply_to_column_found:
        print(f"  Reply-to column:   '{c.reply_to_column_name}'")
        print(f"  Thread starters:   {c.messages_without_parent:,}")
        print(f"  Reply messages:    {c.messages_with_parent:,}")
    else:
        print("  ⚠ No reply-to column detected.")
    if c.response_ids_column_found:
        print(f"  Response-IDs col:  '{c.response_ids_column_name}'")
        print(f"  Messages with responses: {c.messages_with_child:,}")
    if c.thread_id_column_found:
        print(f"  Thread-ID column:  '{c.thread_id_column_name}'")

    _print_section("E. BRAND CANDIDATES")
    if not result.brand_candidates:
        print("  ⚠ No brand candidates identified.")
    else:
        print(f"  {'Brand':<25} {'Total':>8}  {'Outbound':>9}  {'UsableIntr':>11}  {'RespRatio':>10}")
        print(f"  {'-'*25} {'-'*8}  {'-'*9}  {'-'*11}  {'-'*10}")
        for bc in result.brand_candidates[:15]:
            print(
                f"  {bc.author_id:<25} {bc.total_messages:>8,}  "
                f"{bc.outbound_count:>9,}  {bc.usable_interaction_count:>11,}  "
                f"{bc.response_ratio:>10.3f}"
            )

    _print_section("CLEANING ANALYSIS")
    cr = result.cleaning_report
    print(f"  Total null cells:  {cr.total_null_cells:,}")
    print(f"  Duplicate rows:    {cr.duplicate_row_count:,} ({cr.duplicate_row_pct:.2f}%)")
    print(f"  Fully null rows:   {cr.fully_null_rows:,}")
    print("\n  Documented decisions:")
    for d in cr.decisions:
        print(f"    • {textwrap.shorten(d, 90, placeholder='…')}")

    print()


# ---------------------------------------------------------------------------
# Artifact savers
# ---------------------------------------------------------------------------
def _save_schema_json(result: ExplorationResult) -> Path:
    """Save dataset_schema.json to data/interim/."""
    schema = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(result.metadata.filepath),
        "filename": result.metadata.filename,
        "file_size_mb": result.metadata.file_size_mb,
        "row_count": result.structure.row_count,
        "column_count": result.structure.column_count,
        "columns": result.structure.columns,
        "dtypes": result.structure.dtypes,
    }
    out = INTERIM_DIR / "dataset_schema.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved: %s", out)
    return out


def _save_statistics_json(result: ExplorationResult) -> Path:
    """Save dataset_statistics.json to data/interim/."""
    stats: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structure": {
            "row_count": result.structure.row_count,
            "column_count": result.structure.column_count,
            "duplicate_count": result.structure.duplicate_count,
            "duplicate_pct": result.structure.duplicate_pct,
            "missing_by_column": result.structure.missing_by_column,
        },
        "cleaning": {
            "total_null_cells": result.cleaning_report.total_null_cells,
            "fully_null_rows": result.cleaning_report.fully_null_rows,
            "decisions": result.cleaning_report.decisions,
        },
    }
    if result.message.text_column_found:
        stats["message_analysis"] = {
            "text_column": result.message.text_column_name,
            "total_messages": result.message.total_messages,
            "empty_messages": result.message.empty_message_count,
            "length_stats": result.message.length_stats,
        }
    if result.account.author_column_found:
        stats["account_analysis"] = {
            "author_column": result.account.author_column_name,
            "unique_authors": result.account.total_unique_authors,
            "inbound_count": result.account.inbound_count,
            "outbound_count": result.account.outbound_count,
        }
    if result.conversation.reply_to_column_found:
        stats["conversation_analysis"] = {
            "reply_to_column": result.conversation.reply_to_column_name,
            "messages_with_parent": result.conversation.messages_with_parent,
            "messages_without_parent": result.conversation.messages_without_parent,
            "messages_with_child": result.conversation.messages_with_child,
        }

    out = INTERIM_DIR / "dataset_statistics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved: %s", out)
    return out


def _save_account_statistics_csv(result: ExplorationResult) -> Path:
    """Save account_statistics.csv to data/interim/."""
    rows = result.account.top_accounts
    df = pd.DataFrame(rows)
    out = INTERIM_DIR / "account_statistics.csv"
    df.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return out


def _save_brand_candidates_csv(result: ExplorationResult) -> Path | None:
    """Save brand_candidates.csv to data/interim/ if any candidates exist."""
    if not result.brand_candidates:
        logger.info("No brand candidates to save.")
        return None

    rows = [
        {
            "author_id": bc.author_id,
            "total_messages": bc.total_messages,
            "outbound_count": bc.outbound_count,
            "inbound_count": bc.inbound_count,
            "response_ratio": bc.response_ratio,
            "usable_interaction_count": bc.usable_interaction_count,
        }
        for bc in result.brand_candidates
    ]
    df = pd.DataFrame(rows)
    out = INTERIM_DIR / "brand_candidates.csv"
    df.to_csv(out, index=False)
    logger.info("Saved: %s", out)
    return out


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------
def _plot_top_accounts(result: ExplorationResult) -> Path | None:
    """Generate top_accounts.png bar chart."""
    if not result.account.author_column_found or not result.account.top_accounts:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = result.account.top_accounts[:15]
        labels = [r["author"] for r in top]
        values = [r["total_messages"] for r in top]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels[::-1], values[::-1], color="#4C72B0", edgecolor="white")
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
        ax.set_xlabel("Total Messages", fontsize=11)
        ax.set_title(
            "Top 15 Accounts by Message Volume\n(Customer Support on Twitter Dataset)",
            fontsize=13,
            pad=15,
        )
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()

        out = FIGURES_DIR / "top_accounts.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure: %s", out)
        return out
    except Exception as exc:
        logger.warning("Could not generate top_accounts plot: %s", exc)
        return None


def _plot_message_length_distribution(result: ExplorationResult) -> Path | None:
    """Generate message_length_distribution.png histogram."""
    if not result.message.text_column_found:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # We need the raw lengths — re-load from stats
        stats = result.message.length_stats
        if not stats:
            return None

        # Re-load a sample of the data for the plot
        from backend.app.data.loader import discover_dataset, load_dataset
        filepath, _ = discover_dataset()
        df_sample = load_dataset(filepath=filepath, nrows=50000)
        text_col = result.message.text_column_name
        if text_col not in df_sample.columns:
            return None

        lengths = df_sample[text_col].astype(str).str.len()
        lengths = lengths[lengths <= 500]  # clip extreme outliers for visual clarity

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(lengths, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
        ax.axvline(stats["mean"], color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean ({stats['mean']:.0f})")
        ax.axvline(stats["median"], color="#DD8452", linestyle="--", linewidth=1.5, label=f"Median ({stats['median']:.0f})")
        ax.set_xlabel("Message Length (characters)", fontsize=11)
        ax.set_ylabel("Number of Messages", fontsize=11)
        ax.set_title(
            "Distribution of Message Lengths\n(Customer Support on Twitter — first 50k rows)",
            fontsize=13,
            pad=15,
        )
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        out = FIGURES_DIR / "message_length_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure: %s", out)
        return out
    except Exception as exc:
        logger.warning("Could not generate message length distribution plot: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------
def _generate_markdown_report(result: ExplorationResult) -> Path:
    """Generate phase_1_dataset_analysis.md in artifacts/reports/."""
    s = result.structure
    m = result.message
    a = result.account
    c = result.conversation
    cr = result.cleaning_report
    bc = result.brand_candidates

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        "# Phase 1 — Dataset Analysis Report",
        "",
        f"> **Generated:** {now}  ",
        f"> **Source file:** `{s.filename}` ({s.file_size_mb:.1f} MB)  ",
        "> **Status:** Findings based on actual data inspection — no values fabricated.",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "**FACTS FROM THE DATA:**",
        "",
        f"- File: `{s.filename}` ({s.file_size_mb:.1f} MB)",
        f"- Rows: **{s.row_count:,}**",
        f"- Columns: **{s.column_count}**",
        f"- Duplicate rows: **{s.duplicate_count:,}** ({s.duplicate_pct:.2f}%)",
        "",
        "---",
        "",
        "## 2. Schema",
        "",
        "| Column | Data Type | Notes |",
        "|---|---|---|",
    ]

    for col, dtype in s.dtypes.items():
        missing_note = ""
        if col in s.missing_by_column:
            md = s.missing_by_column[col]
            missing_note = f"{md['count']:,} missing ({md['pct']:.1f}%)"
        lines.append(f"| `{col}` | `{dtype}` | {missing_note} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Missing Data",
        "",
        "**FACTS FROM THE DATA:**",
        "",
    ]
    if not s.missing_by_column:
        lines.append("No missing values detected in any column.")
    else:
        lines.append("| Column | Missing Count | Missing % |")
        lines.append("|---|---|---|")
        for col, data in s.missing_by_column.items():
            lines.append(f"| `{col}` | {data['count']:,} | {data['pct']:.2f}% |")

    lines += [
        "",
        "---",
        "",
        "## 4. Duplicate Analysis",
        "",
        "**FACTS FROM THE DATA:**",
        "",
        f"- Total duplicate rows: **{cr.duplicate_row_count:,}** ({cr.duplicate_row_pct:.2f}%)",
        f"- Fully null rows: **{cr.fully_null_rows:,}**",
        "",
        "**INTERPRETATION / RECOMMENDATION:**",
        "",
        "Duplicate rows should be removed before training any model to prevent "
        "data leakage and inflated performance metrics.",
        "",
        "---",
        "",
        "## 5. Message Analysis",
        "",
    ]
    if not m.text_column_found:
        lines.append("_No text/message column detected._")
    else:
        lines += [
            f"**Text column detected:** `{m.text_column_name}`",
            "",
            "**FACTS FROM THE DATA:**",
            "",
            f"- Total messages: **{m.total_messages:,}**",
            f"- Empty messages: **{m.empty_message_count:,}**",
            "",
            "**Message length statistics (characters):**",
            "",
            "| Statistic | Value |",
            "|---|---|",
        ]
        for k, v in m.length_stats.items():
            lines.append(f"| {k} | {v:.1f} |")
        if m.sample_messages:
            lines += ["", "**Sample messages:**", ""]
            for i, msg in enumerate(m.sample_messages, 1):
                safe = msg.replace("|", "\\|")
                lines.append(f"{i}. `{safe[:150]}`")

    lines += [
        "",
        "---",
        "",
        "## 6. Account Analysis",
        "",
    ]
    if not a.author_column_found:
        lines.append("_No author column detected._")
    else:
        lines += [
            f"**Author column:** `{a.author_column_name}`",
            "",
            "**FACTS FROM THE DATA:**",
            "",
            f"- Unique authors: **{a.total_unique_authors:,}**",
        ]
        if a.inbound_column_found:
            lines += [
                f"- Inbound messages (customer): **{a.inbound_count:,}**",
                f"- Outbound messages (brand): **{a.outbound_count:,}**",
            ]
        lines += [
            "",
            "**Top 15 accounts by volume:**",
            "",
            "| Author | Total | Inbound | Outbound |",
            "|---|---|---|---|",
        ]
        for row in a.top_accounts[:15]:
            inb = row.get("inbound", "–")
            out = row.get("outbound", "–")
            inb_str = f"{inb:,}" if isinstance(inb, int) else "–"
            out_str = f"{out:,}" if isinstance(out, int) else "–"
            lines.append(f"| `{row['author']}` | {row['total_messages']:,} | {inb_str} | {out_str} |")

    lines += [
        "",
        "---",
        "",
        "## 7. Conversation Structure",
        "",
        "**FACTS FROM THE DATA:**",
        "",
    ]
    if c.reply_to_column_found:
        lines += [
            f"- Reply-to column: `{c.reply_to_column_name}`",
            f"- Thread starters (no parent): **{c.messages_without_parent:,}**",
            f"- Reply messages (have parent): **{c.messages_with_parent:,}**",
        ]
    else:
        lines.append("- No reply-to column detected.")
    if c.response_ids_column_found:
        lines += [
            f"- Response-IDs column: `{c.response_ids_column_name}`",
            f"- Messages with responses: **{c.messages_with_child:,}**",
        ]
    if c.thread_id_column_found:
        lines.append(f"- Thread-ID column: `{c.thread_id_column_name}`")

    lines += [
        "",
        "**INTERPRETATION:**",
        "",
        "The presence of reply-to and response-ID columns means conversation threads "
        "can be reconstructed by following the parent-child tweet relationships. "
        "This is the foundation for Phase 3 (Conversation Reconstruction).",
        "",
        "---",
        "",
        "## 8. Potential Brand Candidates",
        "",
    ]
    if not bc:
        lines.append("_No brand candidates identified (requires inbound + author columns)._")
    else:
        lines += [
            "**FACTS FROM THE DATA:**",
            "",
            "Brand accounts are identified as non-numeric authors appearing in outbound "
            "(brand-response) rows.",
            "",
            "| Brand | Total Msgs | Outbound | Usable Interactions | Response Ratio |",
            "|---|---|---|---|---|",
        ]
        for cand in bc[:15]:
            lines.append(
                f"| `{cand.author_id}` | {cand.total_messages:,} | "
                f"{cand.outbound_count:,} | {cand.usable_interaction_count:,} | "
                f"{cand.response_ratio:.3f} |"
            )
        lines += [
            "",
            "> ⚠️  **Brand selection is NOT performed in this phase.** "
            "The table above is preparation data for Phase 2 — Evidence-Based Brand Selection.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 9. Data Quality Issues",
        "",
        "**FACTS FROM THE DATA:**",
        "",
    ]
    for decision in cr.decisions:
        lines.append(f"- {decision}")

    lines += [
        "",
        "---",
        "",
        "## 10. What We Learned",
        "",
        "**FACTS FROM THE DATA:**",
        "",
        f"1. The dataset contains **{s.row_count:,} rows** across **{s.column_count} columns**.",
    ]
    if a.author_column_found and a.inbound_column_found:
        lines.append(
            f"2. Clear inbound/outbound separation exists via the `{a.inbound_column_name}` column."
        )
    if c.reply_to_column_found:
        lines.append(
            f"3. Conversation thread structure is available via `{c.reply_to_column_name}` and "
            f"`{c.response_ids_column_name}`."
        )
    if bc:
        lines.append(
            f"4. **{len(bc)} potential brand accounts** identified for Phase 2 selection."
        )
    if m.text_column_found:
        lines.append(
            f"5. Message lengths range from {m.length_stats.get('min', '?'):.0f} to "
            f"{m.length_stats.get('max', '?'):.0f} characters "
            f"(mean: {m.length_stats.get('mean', '?'):.0f})."
        )

    lines += [
        "",
        "---",
        "",
        "## 11. Limitations of This Analysis",
        "",
        "- Brand selection has NOT been performed — all brands are analyzed together.",
        "- Conversation threads have NOT been reconstructed — only raw reply relationships are reported.",
        "- No intent labels exist — this is an unlabeled corpus.",
        "- Analysis uses the full dataset without filtering — quality filtering is a future phase.",
        "- Temporal analysis (date ranges, peak periods) is deferred to Phase 2+.",
        "",
        "---",
        "",
        "## 12. Recommended Next Step",
        "",
        "**Phase 2 — Evidence-Based Brand Selection**",
        "",
        "Using the brand candidates table above (`data/interim/brand_candidates.csv`), "
        "select a primary brand based on:",
        "",
        "1. Volume: minimum ~10,000 usable interactions",
        "2. Conversation depth: high proportion of multi-turn threads",
        "3. Topic diversity: broad enough for meaningful intent clustering",
        "",
        "**Do not choose a brand based on personal preference — choose based on data.**",
        "",
        "---",
        "",
        "_This report was auto-generated by `backend/scripts/explore_dataset.py`. "
        "Do not edit manually — re-run the script to refresh._",
    ]

    report_text = "\n".join(lines)
    out = REPORTS_DIR / "phase_1_dataset_analysis.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Saved Markdown report: %s", out)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 60)
    print("  SupportGraph AI — Phase 1: Dataset Exploration")
    print("=" * 60)
    print()

    _ensure_dirs()

    try:
        result = run_exploration()
    except FileNotFoundError as exc:
        print(f"\n❌ Dataset not found:\n{exc}")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Exploration failed: %s", exc)
        sys.exit(1)

    # Print terminal summary
    _print_results(result)

    # Save artifacts
    _print_section("SAVING ARTIFACTS")
    schema_path = _save_schema_json(result)
    stats_path = _save_statistics_json(result)
    acc_path = _save_account_statistics_csv(result)
    brand_path = _save_brand_candidates_csv(result)
    fig1 = _plot_top_accounts(result)
    fig2 = _plot_message_length_distribution(result)
    report_path = _generate_markdown_report(result)

    print()
    print("  ✅ Artifacts saved:")
    for path in [schema_path, stats_path, acc_path, brand_path, fig1, fig2, report_path]:
        if path:
            print(f"     {path}")

    print()
    print("=" * 60)
    print("  Phase 1 Exploration Complete.")
    print(f"  Full report → {report_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
