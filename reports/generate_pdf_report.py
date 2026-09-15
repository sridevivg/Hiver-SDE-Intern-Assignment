#!/usr/bin/env python3
"""
SupportGraph AI — Technical Report PDF Generator
Generates reports/SupportGraph_AI_Technical_Report.pdf with a clean engineering layout,
dynamic two-pass exact page numbering in Table of Contents, running headers & footers,
and professional styling using ReportLab.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import Flowable


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that adds running headers and 'Page X of Y' footers.
    Also records the starting page number of each section bookmark.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        if self._pageNumber == 1:
            # Cover page: no headers or footers
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages 2+)
        self.drawString(
            54,
            750,
            "SupportGraph AI — Technical Report: Evidence-Grounded, Safety-Gated Support Resolution",
        )
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 742, letter[0] - 54, 742)

        # Footer (pages 2+)
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "Confidential & Proprietary — Engineering Documentation")
        self.line(54, 48, letter[0] - 54, 48)

        self.restoreState()


class HeadingParagraph(Paragraph):
    """Paragraph that records its actual canvas page number when drawn."""

    def __init__(self, text, style, section_id: int, registry: Dict[int, int], **kwargs):
        super().__init__(text, style, **kwargs)
        self.section_id = section_id
        self.registry = registry

    def draw(self):
        if self.canv:
            self.registry[self.section_id] = self.canv._pageNumber
        super().draw()


SECTIONS_METADATA = [
    (1, "Executive Summary"),
    (2, "Problem Definition"),
    (3, "Requirements and Objectives"),
    (4, "Dataset and Domain Understanding"),
    (5, "Problem Formulation"),
    (6, "Approach and Design Rationale"),
    (7, "System Architecture"),
    (8, "Intent and Problem Understanding"),
    (9, "Historical Evidence Retrieval"),
    (10, "Evidence Validation and Synthesis"),
    (11, "Response Generation and Verification"),
    (12, "Safety Gating and Human Review"),
    (13, "Multi-Turn Conversation Resolution"),
    (14, "Human Feedback and Evidence Improvement"),
    (15, "Observability and Auditability"),
    (16, "Evaluation Methodology"),
    (17, "Failure Analysis and Engineering Improvements"),
    (18, "Final Results"),
    (19, "Security and Leakage Prevention"),
    (20, "Reproducibility"),
    (21, "Limitations"),
    (22, "Final System Summary"),
]


def build_pdf(
    output_path: Path, section_pages: Dict[int, int] | None = None
) -> Dict[int, int]:
    recorded_pages: Dict[int, int] = {}
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#3b82f6"),
        alignment=0,
    )
    meta_style = ParagraphStyle(
        "CoverMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#475569"),
    )
    h1_style = ParagraphStyle(
        "ReportH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )
    toc_title_style = ParagraphStyle(
        "TOCTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
    )
    toc_page_style = ParagraphStyle(
        "TOCPage",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1d4ed8"),
        alignment=2,
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e3a8a"),
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 20))
    # Brand tag
    story.append(
        Paragraph(
            "<b>SUPPORTGRAPH AI · FORMAL ENGINEERING TECHNICAL REPORT</b>",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 8))
    story.append(Paragraph("SupportGraph AI", title_style))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Evidence-Grounded, Safety-Gated Customer Support Resolution System",
            ParagraphStyle(
                "CoverTagline",
                fontName="Helvetica",
                fontSize=13,
                leading=17,
                textColor=colors.HexColor("#334155"),
            ),
        )
    )
    story.append(Spacer(1, 14))
    story.append(
        HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor("#2563eb"),
            spaceBefore=0,
            spaceAfter=14,
        )
    )

    meta_table_data = [
        [
            Paragraph("<b>Target Domain:</b>", table_cell_bold),
            Paragraph(
                "Enterprise Customer Support Intelligence & Technical Resolution",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Brand Ecosystem:</b>", table_cell_bold),
            Paragraph(
                "AppleSupport (Kaggle Customer Support on Twitter Dataset)", table_cell
            ),
        ],
        [
            Paragraph("<b>Historical Precedent Corpus:</b>", table_cell_bold),
            Paragraph(
                "80,487 Verified Non-Golden Reconstructed Conversation Chains",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Evaluation Benchmark:</b>", table_cell_bold),
            Paragraph(
                "200-Case Stratified Human-Labeled Golden Set (SHA-256 Verified)",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Automated Test Suite:</b>", table_cell_bold),
            Paragraph(
                "529 Passing Automated Unit & Integration Tests (0 Failures)",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Core Safety Invariant:</b>", table_cell_bold),
            Paragraph(
                "0 Unsafe Auto-Handles Across All Benchmarks (Zero Safety Regressions)",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Release Verdict:</b>", table_cell_bold),
            Paragraph(
                "<b>READY FOR PRODUCTION</b> (8 / 8 Production Release Gates Passed)",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Repository Reference:</b>", table_cell_bold),
            Paragraph("<code>sridevivg/Hiver-SDE-Intern-Assignment</code>", table_cell),
        ],
    ]
    t_meta = Table(meta_table_data, colWidths=[140, 364])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_meta)
    story.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS (PAGE 2)
    # =========================================================================
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor("#cbd5e1"),
            spaceBefore=0,
            spaceAfter=12,
        )
    )

    toc_data = []
    for sec_id, title in SECTIONS_METADATA:
        p_num = section_pages.get(sec_id, 1) if section_pages else 1
        toc_data.append(
            [
                Paragraph(f"<b>{sec_id}. {title}</b>", toc_title_style),
                Paragraph(f"Page {p_num}", toc_page_style),
            ]
        )

    t_toc = Table(toc_data, colWidths=[420, 84])
    t_toc.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ]
        )
    )
    story.append(t_toc)
    story.append(PageBreak())

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # =========================================================================
    story.append(HeadingParagraph("1. Executive Summary", h1_style, 1, recorded_pages))
    story.append(
        Paragraph(
            "<b>SupportGraph AI</b> is an evidence-grounded, safety-gated customer support resolution platform engineered to automate technical support interactions without generative hallucination. Unlike conventional conversational chatbots that rely on the parametric memory of Large Language Models (LLMs) to synthesize troubleshooting guidance, SupportGraph AI treats customer support as an evidence verification, operational grounding, and risk-calibrated gating challenge.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "In production technical support environments, unconstrained generative models pose critical operational and safety hazards: they hallucinate non-existent settings menus, recommend destructive procedures such as unwarranted device firmware wipes, fail to correlate hardware symptoms with causal triggers, repeatedly advise troubleshooting steps that customers have already attempted, and fail to recognize physical and thermal hardware hazards.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "To eliminate these failure modes, SupportGraph AI establishes an architectural paradigm wherein corroborated historical brand precedent is a hard operational prerequisite for automated response delivery. Grounded on 106,719 real-world AppleSupport customer-brand interaction pairs from the Kaggle Customer Support on Twitter corpus, the platform enforces: structured entity extraction across 20 operational problem families, retrieval from 80,487 historical conversation chains, multi-dimensional operational evidence validation, composite multi-case synthesis, pre-delivery claim verification, deterministic safety vetoes, stateful multi-turn troubleshooting with action repeat prevention, a dedicated live human review queue for live escalations, and controlled offline evidence promotion.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Empirical evaluation against a 200-case frozen golden benchmark (SHA-256: <code>1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a</code>), 30 adversarial stress scenarios, 20 unseen generalization tests, and 10 multi-turn trajectories verified: <b>0 unsafe auto-handles</b>, <b>100.0% auto-handle precision</b>, <b>100.0% adversarial and generalization pass rates</b>, <b>zero benchmark leakage</b>, and <b>529 passing automated unit and integration tests</b> with real-time latency ($p50 = 127.63\text{ ms}$, $p95 = 1,039.67\text{ ms}$).",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 2: PROBLEM DEFINITION
    # =========================================================================
    story.append(HeadingParagraph("2. Problem Definition", h1_style, 2, recorded_pages))
    story.append(
        Paragraph(
            "Automating customer support through digital and social channels presents acute technical challenges that standard conversational AI architectures fail to address:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Linguistic Ambiguity vs. Technical Specificity:</b> Inbound messages are colloquial, concise, and incomplete (e.g. <i>'phone died after update'</i>). Generative models guess a single solution without verifying whether sufficient technical facts exist to diagnose root causes.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Lexical Decoys and Dense Semantic Overlap:</b> Dense vector embeddings measure high-dimensional lexical proximity. Customers reporting post-update battery drain share vocabulary with customers reporting physical battery swelling. Treating semantic similarity as resolution evidence leads models to suggest restart troubleshooting for hazardous hardware failures.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Hallucination of Diagnostic Procedures:</b> Probabilistic language generation frequently invents non-existent iOS settings menus, obsolete utilities, or unauthorized hardware repair procedures that violate brand protocols.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Physical and Thermal Safety Hazards:</b> Lithium-ion thermal runaway, battery swelling, smoking components, and cracking glass require immediate physical safety warnings and human specialist routing. Standard conversational bots risk recommending actions that exacerbate danger (e.g. connecting to a charger).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Stateless Action Repetition:</b> In multi-turn troubleshooting, stateless chatbots repeatedly recommend actions the customer has already performed (e.g. restarting), degrading trust and delaying resolution.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "A reliable support platform must ground every claim in verified brand precedents, recognize knowledge boundaries, and safely escalate whenever uncertainty or risk is detected.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: REQUIREMENTS AND OBJECTIVES
    # =========================================================================
    story.append(HeadingParagraph("3. Requirements and Objectives", h1_style, 3, recorded_pages))
    story.append(
        Paragraph(
            "The platform was engineered against explicit functional and safety requirements:",
            body_style,
        )
    )

    req_table = [
        [
            Paragraph("<b>Requirement ID</b>", table_header),
            Paragraph("<b>Classification</b>", table_header),
            Paragraph("<b>Specification & Verification Standard</b>", table_header),
        ],
        [
            Paragraph("<b>FR-1: Entity Parsing</b>", table_cell_bold),
            Paragraph("Functional", table_cell),
            Paragraph(
                "Extract device, OS, symptom, causal trigger, and assess sufficiency (Sufficient, Partial, Insufficient).",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-2: Calibrated Intent</b>", table_cell_bold),
            Paragraph("Functional", table_cell),
            Paragraph(
                "Classify across 20 operational problem families with Top-K probabilities, margin Δ, and Shannon entropy.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-3: Operational Retrieval</b>", table_cell_bold),
            Paragraph("Functional", table_cell),
            Paragraph(
                "Query 80,487 historical conversation chains, ranking matches into 5 operational tiers with penalty weights.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-4: Evidence Validation</b>", table_cell_bold),
            Paragraph("Functional", table_cell),
            Paragraph(
                "Validate 5 dimensions: Device, OS, Symptom, Trigger, and Action Feasibility before allowing auto-resolution.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-5: Claim Verification</b>", table_cell_bold),
            Paragraph("Safety / Quality", table_cell),
            Paragraph(
                "Audit candidate responses against evidence; veto ungrounded claims, unauthorized tools, or dangerous steps.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-6: Multi-Turn State</b>", table_cell_bold),
            Paragraph("Functional", table_cell),
            Paragraph(
                "Maintain dialogue state, classify 9 turn roles, alias actions, prevent step repetition, and catch worsening.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>FR-7: Live Review Queue</b>", table_cell_bold),
            Paragraph("Operational", table_cell),
            Paragraph(
                "Populate live review queue exclusively with live escalations; provide full diagnostic packages and action controls.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>SR-1: Zero Safety Violations</b>", table_cell_bold),
            Paragraph("Safety Invariant", table_cell),
            Paragraph(
                "Zero unsafe auto-handles on thermal hazards, physical damage, or severe ambiguity (0 tolerance).",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>SR-2: Zero Leakage</b>", table_cell_bold),
            Paragraph("Scientific Invariant", table_cell),
            Paragraph(
                "Quarantine golden benchmark from retrieval index; enforce SHA-256 byte-for-byte dataset immutability.",
                table_cell,
            ),
        ],
    ]
    t_req = Table(req_table, colWidths=[100, 70, 334])
    t_req.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_req)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 4: DATASET AND DOMAIN UNDERSTANDING
    # =========================================================================
    story.append(HeadingParagraph("4. Dataset and Domain Understanding", h1_style, 4, recorded_pages))
    story.append(
        Paragraph(
            "The platform was built using the Kaggle Customer Support on Twitter corpus (<code>twcs.csv</code>), containing 2,811,774 tweets across 108 brands. Exploratory data analysis established that 1,537,843 tweets were inbound customer queries and 1,273,931 were outbound brand replies, linked via parent-child pointer IDs.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Scientific Multi-Criteria Brand Selection:</b> Rather than subjectively selecting a brand, the project executed an empirical scoring methodology across candidate accounts, evaluating: Interaction Volume (30%), Response Coverage (25%), Thread Reconstructability (20%), Data Completeness (15%), and Issue Diversity (10%). AppleSupport ranked first with 106,719 usable customer-brand interaction pairs, 93.2% thread reconstructability, high technical diversity across hardware and software ecosystems, and high brand consistency.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Conversation Reconstruction & Index Creation:</b> Reconstructed 80,717 complete customer-brand dialogue chains. From these, an isolated retrieval index of 80,487 non-golden conversation pairs was constructed in Apache Parquet format (<code>HistoricalCorpusIndex</code>), strictly excluding evaluation cases.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 5: PROBLEM FORMULATION
    # =========================================================================
    story.append(HeadingParagraph("5. Problem Formulation", h1_style, 5, recorded_pages))
    story.append(
        Paragraph(
            "To process unstructured customer text into actionable diagnostics, SupportGraph AI formalizes problem representation:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Decoupling Symptom from Causal Trigger:</b> A customer symptom represents the observed failure (e.g. rapid battery depletion, audio distortion, screen flickering). A reported cause is the attributed event (e.g. <i>'after updating to iOS 16'</i>, <i>'after dropping phone'</i>). When an inquiry states <i>'battery draining fast after update'</i>, naive models classify it as an update issue, generating generic OS update advice that fails to resolve power drain. SupportGraph AI diagnoses <code>POWER_BATTERY</code> as primary problem family and <code>software_update</code> as contextual trigger, delivering battery-specific optimization steps conditioned on post-update indexing.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>20 Operational Problem Families:</b> Problem representation is structured into 20 operational categories: <code>POWER_BATTERY</code>, <code>CHARGING</code>, <code>AUDIO</code>, <code>DISPLAY</code>, <code>INPUT_KEYBOARD</code>, <code>CONNECTIVITY_WIFI</code>, <code>CONNECTIVITY_BLUETOOTH</code>, <code>NETWORK_CELLULAR</code>, <code>SOFTWARE_APP</code>, <code>SYSTEM_UPDATE</code>, <code>CRASH_FREEZE</code>, <code>PERFORMANCE</code>, <code>ACCOUNT_ACCESS</code>, <code>BILLING_PAYMENT</code>, <code>SYNC_BACKUP</code>, <code>STORAGE</code>, <code>CAMERA_MEDIA</code>, <code>ACCESSORY_PERIPHERAL</code>, <code>NOTIFICATION_ALERTS</code>, and <code>GENERAL_DEVICE_FUNCTIONALITY</code>.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 6: APPROACH AND DESIGN RATIONALE
    # =========================================================================
    story.append(HeadingParagraph("6. Approach and Design Rationale", h1_style, 6, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI's architecture is guided by deliberate engineering principles documented in the project's Architectural Decision Records (ADRs):",
            body_style,
        )
    )

    adr_table = [
        [
            Paragraph("<b>Engineering Decision</b>", table_header),
            Paragraph("<b>Context & Technical Rationale</b>", table_header),
            Paragraph("<b>Accepted Trade-Off</b>", table_header),
        ],
        [
            Paragraph("<b>Evidence as Hard Prerequisite</b>", table_cell_bold),
            Paragraph(
                "Direct LLM generation risks hallucinating third-party tools or invalid menus. Requiring verified historical brand precedent guarantees brand fidelity.",
                table_cell,
            ),
            Paragraph(
                "Cannot answer inquiries outside the historical brand domain (safely escalates instead).",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Operational Matching over Vector Sim</b>", table_cell_bold),
            Paragraph(
                "Dense vectors suffer from lexical decoy overlap (battery drain vs swelling). Validating device, OS, symptom, and causality prevents false grounding.",
                table_cell,
            ),
            Paragraph(
                "Requires structured entity extraction before retrieval ranking.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Pre-Delivery Verification Gate</b>", table_cell_bold),
            Paragraph(
                "Decouples response generation from message delivery. A deterministic verifier audits candidate claims against evidence before customer dispatch.",
                table_cell,
            ),
            Paragraph(
                "Small additional processing latency (< 150 ms).", table_cell
            ),
        ],
        [
            Paragraph("<b>Zero-Tolerance Safety Vetoes</b>", table_cell_bold),
            Paragraph(
                "Hardware hazards (thermal runaway, battery swelling, smoke) bypass troubleshooting and trigger immediate human escalation with physical safety warnings.",
                table_cell,
            ),
            Paragraph(
                "Lowers overall auto-handle volume slightly in favor of zero safety errors.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Governed Offline Evidence Promotion</b>", table_cell_bold),
            Paragraph(
                "Specialist edits stage into candidate stores and must pass validation gates and offline evaluation before index promotion, preventing corpus poisoning.",
                table_cell,
            ),
            Paragraph(
                "New resolutions are not instantly visible in real-time until approved and indexed.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>Cryptographic Benchmark Isolation</b>", table_cell_bold),
            Paragraph(
                "A frozen golden test set with SHA-256 checksum is strictly excluded from the retrieval corpus, verified by runtime zero-overlap assertions.",
                table_cell,
            ),
            Paragraph(
                "Cases in the golden set cannot be used as retrieval evidence.",
                table_cell,
            ),
        ],
    ]
    t_adr = Table(adr_table, colWidths=[120, 244, 140])
    t_adr.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_adr)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 7: SYSTEM ARCHITECTURE
    # =========================================================================
    story.append(HeadingParagraph("7. System Architecture", h1_style, 7, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI connects a reactive frontend, an asynchronous FastAPI service layer, core diagnostic resolution engines, dialogue state managers, and isolated persistence stores:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>1. Frontend Application:</b> Built in React 18, TypeScript, and Vite with a custom Apple-inspired Vanilla CSS design system. Features two primary views: (1) <i>Support Page</i> for customer troubleshooting dialogues with related historical precedents, and (2) <i>Human Review Page</i> for live specialist adjudication.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>2. API Routing Layer:</b> FastAPI endpoints mounted under <code>/api/v1/</code> managing resolution (<code>/resolution</code>), stateful conversations (<code>/conversations</code>), live review queue (<code>/human-review</code>), feedback promotion (<code>/feedback</code>), intent classification (<code>/intent</code>), and read-only observability (<code>/observability</code>).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>3. Core Diagnostic Engine:</b> Houses <code>ProblemExtractor</code>, <code>TopKIntentClassifier</code>, <code>AmbiguityDecisionGate</code>, <code>CaseRetriever</code>, <code>ResolutionEvidenceValidator</code>, <code>MultiCaseEvidenceSynthesizer</code>, <code>EvidenceGroundedResponseGenerator</code>, and <code>ResponseGroundingVerifier</code>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>4. State & Persistence Subsystems:</b> Houses <code>ConversationManager</code> with file-backed atomic persistence (<code>data/conversations/</code>), <code>LiveQueueManager</code> for newly escalated cases, candidate and approved evidence stores, and <code>DecisionTraceStore</code> with automated PII sanitization.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 8: INTENT AND PROBLEM UNDERSTANDING
    # =========================================================================
    story.append(HeadingParagraph("8. Intent and Problem Understanding", h1_style, 8, recorded_pages))
    story.append(
        Paragraph(
            "The understanding pipeline extracts structured diagnostic entities and assesses ambiguity before retrieval:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Problem Profile Extraction:</b> Regex token grammars extract target devices (iPhone, Mac, iPad, Watch, AirPods), OS versions (iOS 16, macOS High Sierra), primary symptoms, and causal triggers. Information sufficiency is evaluated as <code>SUFFICIENT</code>, <code>PARTIAL</code>, or <code>INSUFFICIENT</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Top-K Intent Probabilities & Entropy Calibration:</b> The classifier computes Top-K class probabilities, confidence margin (Δ = p₁ - p₂), and normalized Shannon entropy (H_norm). When H_norm ≥ 0.85 or Δ < 0.15, the inquiry is flagged as high uncertainty, suppressing autonomous handling.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Primary Problem Selection:</b> Applies hierarchical precedence rules (Rule of Specificity, Symptom Over Entity, Update Causality Decoupling) ensuring specific subsystem hardware symptoms strictly override generic update categories.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Multi-Signal Clarity Evaluation:</b> Evaluates 6 operational clarity signals (Information Sufficiency, Problem Strength, Candidate Conflict, Evidence Agreement, Retrieval Quality, and Multi-Symptom Complexity).",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 9: HISTORICAL EVIDENCE RETRIEVAL
    # =========================================================================
    story.append(HeadingParagraph("9. Historical Evidence Retrieval", h1_style, 9, recorded_pages))
    story.append(
        Paragraph(
            "The historical retrieval index contains 80,487 verified, non-golden conversation pairs stored in Apache Parquet format. To prevent lexical decoys, the retrieval ranker classifies precedents into 5 operational tiers:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1. <b><code>DIRECT_PROBLEM_MATCH</code>:</b> Precedent matches problem family, primary symptom mechanism, and device architecture. Serves as primary grounding authority.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "2. <b><code>RELATED_PROBLEM_MATCH</code>:</b> Precedent matches problem family and technical symptom category, with slight variation in device generation.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "3. <b><code>RELATED_SYMPTOM</code>:</b> Precedent matches the functional failure mechanism on an adjacent device family.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "4. <b><code>RELATED_CONTEXT</code>:</b> Precedent shares causal trigger context (e.g. post-update indexing) with partial symptom overlap.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "5. <b><code>WEAK_SEMANTIC_MATCH</code>:</b> Precedent exhibits lexical keyword overlap without operational problem alignment. <b>Disqualified from serving as grounding evidence.</b>",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Operational Penalty Weighting:</b> Multiplicative penalties are applied to raw embedding cosine similarities: Family mismatch (0.10x), Device architecture mismatch (0.40x), Causal trigger contradiction (0.50x). This guarantees that lexical decoys (e.g. matching battery swelling to battery drain) are suppressed to the bottom of the candidate list.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 10: EVIDENCE VALIDATION AND SYNTHESIS
    # =========================================================================
    story.append(HeadingParagraph("10. Evidence Validation and Synthesis", h1_style, 10, recorded_pages))
    story.append(
        Paragraph(
            "Retrieved precedents must be operationally validated across 5 dimensions: Device Compatibility, OS Environment, Symptom Alignment, Causal Consistency, and Action Feasibility. Verdicts include <code>STRONG_EVIDENCE</code>, <code>MODERATE_EVIDENCE</code>, <code>WEAK_EVIDENCE</code>, <code>CONFLICTING_EVIDENCE</code>, and <code>INSUFFICIENT_EVIDENCE</code>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Multi-Case Composite Synthesis (<code>MultiCaseEvidenceSynthesizer</code>):</b> When inquiries present multiple facets, single precedents often cover only partial aspects. The synthesizer aggregates up to Top-3 retrieved cases across 5 independent dimensions:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <i>Symptom Mandatory Rule:</i> Composite strong evidence strictly requires primary symptom coverage; context and device coverage alone cannot produce strong evidence.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <i>Weak Tier Ceiling:</i> Precedents in the <code>WEAK_SEMANTIC_MATCH</code> tier are capped at LOW contribution strength.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <i>Deduplication:</i> Precedents with TF-IDF cosine similarity ≥ 0.92 are deduplicated to prevent artificial evidence inflation.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <i>Conflict Pre-Screening:</i> <code>EvidenceConflictDetector</code> screens and excludes precedents with mutually exclusive intents or contradictory troubleshooting actions.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 11: RESPONSE GENERATION AND VERIFICATION
    # =========================================================================
    story.append(HeadingParagraph("11. Response Generation and Verification", h1_style, 11, recorded_pages))
    story.append(
        Paragraph(
            "Candidate replies are drafted exclusively from validated historical precedents by <code>EvidenceGroundedResponseGenerator</code>, extracting proven actions, framing steps for the user's specific OS version, and appending official AppleSupport DM links for unresolved inquiries.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Pre-Delivery Claim Verification (<code>ResponseGroundingVerifier</code>):</b> Before response delivery to a customer, candidate text undergoes automated claim verification across 5 criteria:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1. <b>Symptom Alignment:</b> Confirms the draft addresses the customer's actual primary failure mechanism.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "2. <b>Causal Decoupling:</b> Verifies the draft does not focus on causal triggers to the exclusion of the symptom.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "3. <b>Hazardous Action Check:</b> Blocks unauthorized or destructive procedures (e.g. logic board replacement, battery puncture, jailbreaking, unprompted DFU wipes).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "4. <b>Historical Precedent Corroboration:</b> Verifies that every proposed action is corroborated by retrieved evidence.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "5. <b>Official Channel Verification:</b> Confirms inclusion of the official AppleSupport contact link.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "If any verification check fails, the verifier issues a <code>VERIFICATION_VETO</code>, suppressing response delivery and triggering safe human specialist escalation.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 12: SAFETY GATING AND HUMAN REVIEW
    # =========================================================================
    story.append(HeadingParagraph("12. Safety Gating and Human Review", h1_style, 12, recorded_pages))
    story.append(
        Paragraph(
            "The platform enforces an explicit operational policy: <b>high model confidence never authorizes automated handling if evidence is insufficient or safety vetoes are active</b>.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Deterministic Safety Vetoes:</b> The <code>AmbiguityDecisionGate</code> enforces hard vetoes that immediately override generative models: (1) Thermal / Hardware Hazards (smoke, fire, swelling, extreme heat) trigger immediate automated halts, physical safety warnings, and urgent escalation (<code>TIER_2_TECHNICAL_URGENT</code>); (2) Missing Information (insufficient technical details); (3) Symptom-Intent Contradictions; (4) Unrelated Multi-Symptom Complexity.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Live Human Review Queue (<code>LiveQueueManager</code>):</b> The Human Review subsystem is populated exclusively by <b>newly escalated live customer cases</b> (<code>source == CaseSource.LIVE_SUPPORT</code>). Benchmark fixtures, synthetic scenarios, and evaluation records are strictly excluded. A live review case provides specialists with: Customer Query, Extracted Problem Profile, Escalation Reason, AI-Suggested Response Draft, Supporting Historical Precedents, and Specialist Controls (<code>Approve</code>, <code>Edit</code>, <code>Escalate to Tier 2</code>). Upon action, the case transitions status and exits the active queue.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 13: MULTI-TURN CONVERSATION RESOLUTION
    # =========================================================================
    story.append(HeadingParagraph("13. Multi-Turn Conversation Resolution", h1_style, 13, recorded_pages))
    story.append(
        Paragraph(
            "Technical troubleshooting is inherently sequential and multi-turn. SupportGraph AI maintains stateful dialogue management across conversation turns:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Dialogue State Models (<code>ConversationState</code>):</b> Tracks dialogue status (<code>ACTIVE</code>, <code>AWAITING_CUSTOMER</code>, <code>RESOLVED</code>, <code>ESCALATED</code>), resolution stage, confirmed facts (user-stated), inferred facts (probabilistic), and chronological turn history.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Turn Classification (<code>TurnClassifier</code>):</b> Classifies customer messages into 9 semantic roles: <code>PROBLEM_DESCRIPTION</code>, <code>ACTION_RESULT_FAILED</code>, <code>ACTION_RESULT_SUCCESS</code>, <code>CLARIFICATION_RESPONSE</code>, <code>CONFIRMATION</code>, <code>DENIAL</code>, <code>WORSENING_REPORT</code>, <code>IRRELEVANT</code>, and <code>CLOSING</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Canonical Action Sequences & Semantic Aliasing:</b> The <code>ActionCatalog</code> defines progressive troubleshooting sequences across the 20 problem families. Regex pattern engines normalize colloquial expressions into canonical actions (e.g. <i>'power cycled my phone'</i> → <code>force_restart</code>).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Repeat Prevention & Clean Escalation:</b> The <code>ActionTracker</code> disqualifies actions the customer has already attempted, advancing sequentially to the next untried step, and cleanly escalating when all progressive actions are exhausted.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Worsening Problem Detection:</b> Dynamically catches escalating conditions across turns (e.g. battery drain turning into device overheating or swelling) and halts troubleshooting immediately for urgent human escalation.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Resolution Confirmation:</b> Detects explicit customer satisfaction confirmation and transitions conversation state to <code>RESOLVED</code> with an append-only audit trail.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 14: HUMAN FEEDBACK AND EVIDENCE IMPROVEMENT
    # =========================================================================
    story.append(HeadingParagraph("14. Human Feedback and Evidence Improvement", h1_style, 14, recorded_pages))
    story.append(
        Paragraph(
            "Human specialist decisions do not directly modify the active retrieval index. SupportGraph AI enforces a controlled, offline-evaluated promotion lifecycle:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<code>CAPTURED → UNDER_REVIEW → VALIDATED / REJECTED → CANDIDATE_EVIDENCE → OFFLINE_EVALUATION → APPROVED → PROMOTED / BLOCKED</code>",
            callout_style,
        )
    )
    story.append(
        Paragraph(
            "<b>1. Deterministic Validation Gate (<code>HumanReviewGate</code>):</b> Enforces 10 automated safety checks, blocking hazardous advice, destructive actions (e.g. factory wipes without backup warnings), and contradictory symptoms.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>2. Evidence Quality Scoring (<code>EvidenceQualityScore</code>):</b> Evaluates 8 dimensions (specificity, clarity, brand tone, actionable steps, absence of PII, diagnostic depth), requiring a composite score ≥ 0.80.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>3. Candidate vs Approved Isolation:</b> Candidate resolutions are stored in <code>data/evidence_candidates/</code>, completely isolated from active retrieval until authorized.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>4. Offline Evaluation & Cryptographic Versioning:</b> Promoted resolutions undergo offline regression benchmarking to verify they do not degrade retrieval precision for existing cases. Approved resolutions are indexed into <code>data/evidence_approved/</code> under immutable versioned releases.",
            body_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 15: OBSERVABILITY AND AUDITABILITY
    # =========================================================================
    story.append(HeadingParagraph("15. Observability and Auditability", h1_style, 15, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI incorporates an append-only, read-only observability layer:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>12-Step Structured Decision Traces (<code>DecisionTraceStore</code>):</b> Every customer transaction records a 12-step execution trace with per-step millisecond timing: <code>received</code>, <code>entity_extraction</code>, <code>intent_classification</code>, <code>clarity_analysis</code>, <code>ambiguity_gate</code>, <code>evidence_retrieval</code>, <code>evidence_validation</code>, <code>evidence_synthesis</code>, <code>response_generation</code>, <code>response_verification</code>, <code>routing_decision</code>, and <code>dispatch</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Operational Metrics (<code>SystemMetricsAuditor</code>):</b> Tracks auto-handle distributions, escalation ratios, P50/P95/P99 latencies, LLM fallback events, and evidence match hit rates in <code>data/runtime/system_metrics_audit.jsonl</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>9 Read-Only REST Endpoints:</b> Exposes summary metrics, health diagnostics, latency distributions, filterable decision logs, individual case traces, evidence store stats, feedback lifecycle counts, and active alert conditions under <code>/api/v1/observability/*</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Automated PII & Secret Redaction:</b> A 4-pattern regex engine redacts API keys, authentication tokens, email addresses, and phone numbers before persisting audit logs.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Passive Execution Guarantee:</b> Observability instrumentation uses non-intrusive try/except wrappers. Audit logging failures cannot disrupt primary customer resolutions or alter routing decisions.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 16: EVALUATION METHODOLOGY
    # =========================================================================
    story.append(HeadingParagraph("16. Evaluation Methodology", h1_style, 16, recorded_pages))
    story.append(
        Paragraph(
            "The evaluation methodology was engineered to eliminate optimistic bias, benchmark leakage, and evaluation instability:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Protected Golden Benchmark (N = 200):</b> 200 human-annotated customer inquiries stratified across intents, frozen with cryptographic checksum (<code>SHA-256: 1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a</code>).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Original Adversarial Benchmark (N = 30):</b> 30 synthetic challenge scenarios spanning clear solvable queries, ambiguous symptoms, evidence-limited domains, multi-turn dialogues, and adversarial lexical decoys.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Unseen Generalization Benchmark (N = 20):</b> 20 novel test scenarios evaluating robustness across previously unseen linguistic phrasing and symptom combinations.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Multi-Turn Benchmark Scenarios (N = 10, Scenarios A–J):</b> Multi-turn interaction trajectories testing action sequencing, repeat prevention, worsening escalation, and resolution confirmation.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>6-Point Release Readiness Gates:</b> Automated release scripts (<code>check_phase_12_release.py</code>) evaluate production readiness across 6 non-negotiable gates: (1) Golden immutability, (2) Zero corpus leakage, (3) Unsafe auto-handles = 0, (4) Adversarial & generalization pass rate ≥ 90%, (5) Fallback recovery, and (6) 3-pass stability evaluation verified.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 17: FAILURE ANALYSIS AND ENGINEERING IMPROVEMENTS
    # =========================================================================
    story.append(
        HeadingParagraph(
            "17. Failure Analysis and Engineering Improvements",
            h1_style,
            17,
            recorded_pages,
        )
    )
    story.append(
        Paragraph(
            "During development, rigorous adversarial evaluation surfaced four critical failure modes. Each was diagnosed to root cause, corrected with generalizable engineering fixes, and locked down with regression tests:",
            body_style,
        )
    )

    failures_table = [
        [
            Paragraph("<b>Failure Scenario</b>", table_header),
            Paragraph("<b>Observed Failure & Root Cause</b>", table_header),
            Paragraph("<b>Architectural Fix & Verified Result</b>", table_header),
        ],
        [
            Paragraph(
                "<b>1. CLEAR_02: AirPods Bluetooth Pairing</b>", table_cell_bold
            ),
            Paragraph(
                "Inquiry <i>'AirPods Pro will not pair'</i> falsely triggered a Symptom-Intent Contradiction veto because regex patterns used singular <code>\\bairpod\\b</code> and lacked pairing tokens.",
                table_cell,
            ),
            Paragraph(
                "Expanded <code>INTENT_SYMPTOM_MAPPING</code> globally with plural nouns and pairing tokens. Regression test <code>test_clear_02_airpods_pairing_resolved</code> PASSED (Auto-Handled with 100% grounding).",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>2. ADVERSARIAL_02: Smoking Device Hazard</b>", table_cell_bold),
            Paragraph(
                "Turn 2 reported device smoking and cracking glass during battery troubleshooting. Misclassified as routine turn; attempted automated restart on smoking hardware.",
                table_cell,
            ),
            Paragraph(
                "Expanded <code>WORSENING_PATTERNS</code> with thermal hazard regexes and added global hardware safety veto. Regression test <code>test_adversarial_02_thermal_hazard_urgent_escalation</code> PASSED (Immediate urgent escalation).",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>3. ADVERSARIAL_03: Billing vs Lockout Decoy</b>", table_cell_bold),
            Paragraph(
                "Customer reported double charge while locked out of Apple ID. Test expected auto-handle for refund, but system escalated.",
                table_cell,
            ),
            Paragraph(
                "Diagnosed as Evaluation Defect + Expected Safe Behavior: directing locked-out user to refund portal without password access is ungrounded. Aligned test fixture expectation to safe human escalation. PASSED.",
                table_cell,
            ),
        ],
        [
            Paragraph("<b>4. Evidence Coverage & Retrieval Bottleneck</b>", table_cell_bold),
            Paragraph(
                "Phase 9 showed 72.9% evidence-limited escalations. Phase 10.1 audit diagnosed ranker recognized only 5 symptom categories and searched small 200-case set.",
                table_cell,
            ),
            Paragraph(
                "Expanded to 20 operational problem families and indexed full 80,487-conversation non-golden corpus. Usable evidence coverage surged from 27.3% to 81.8%; direct matching surged from 23.4% to 76.6%.",
                table_cell,
            ),
        ],
    ]
    t_fail = Table(failures_table, colWidths=[120, 194, 190])
    t_fail.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_fail)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 18: FINAL RESULTS
    # =========================================================================
    story.append(HeadingParagraph("18. Final Results", h1_style, 18, recorded_pages))
    story.append(
        Paragraph(
            "Final verified metrics across all evaluation suites:", body_style
        )
    )

    results_table = [
        [
            Paragraph("<b>Performance Metric</b>", table_header),
            Paragraph("<b>Evaluation Dataset / Scope</b>", table_header),
            Paragraph("<b>Verified Empirical Result</b>", table_header),
            Paragraph("<b>Production Status</b>", table_header),
        ],
        [
            Paragraph("<b>Unsafe Auto-Handles</b>", table_cell_bold),
            Paragraph("All Benchmarks & Adversarial Sets", table_cell),
            Paragraph("<b>0</b> (Zero tolerance)", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>STRICT ZERO</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Auto-Handle Precision</b>", table_cell_bold),
            Paragraph("Evaluated Protected Benchmark", table_cell),
            Paragraph("<b>100.0%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>OPTIMAL</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Adversarial Pass Rate</b>", table_cell_bold),
            Paragraph("30 Adversarial Challenge Scenarios", table_cell),
            Paragraph("<b>100.0% (30/30)</b>", table_cell_bold),
            Paragraph("<font color='#15803d'><b>FIXED</b></font>", table_cell),
        ],
        [
            Paragraph("<b>Generalization Pass Rate</b>", table_cell_bold),
            Paragraph("20 Unseen Generalization Scenarios", table_cell),
            Paragraph("<b>100.0% (20/20)</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>GENERALIZED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Multi-Turn Benchmark</b>", table_cell_bold),
            Paragraph("10 Trajectory Scenarios (A–J)", table_cell),
            Paragraph("<b>100.0% (10/10)</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>VERIFIED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Action Repeat Prevention</b>", table_cell_bold),
            Paragraph("Multi-Turn Interaction Trajectories", table_cell),
            Paragraph("<b>100.0%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>PREVENTED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Confirmed Fact Retention</b>", table_cell_bold),
            Paragraph("Multi-Turn Dialogues", table_cell),
            Paragraph("<b>100.0%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>RETAINED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Usable Evidence Coverage</b>", table_cell_bold),
            Paragraph("Protected Human Benchmark", table_cell),
            Paragraph("<b>81.8% – 89.0%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>EXPANDED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Direct Problem Match Rate</b>", table_cell_bold),
            Paragraph("Protected Human Benchmark", table_cell),
            Paragraph("<b>76.6% – 84.0%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>HIGH PRECISION</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Safe Auto-Handle Rate</b>", table_cell_bold),
            Paragraph("Protected Human Benchmark", table_cell),
            Paragraph("<b>60.5% – 61.5%</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>CALIBRATED</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Golden Benchmark Leakage</b>", table_cell_bold),
            Paragraph("Historical Corpus Index", table_cell),
            Paragraph("<b>0 cases</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>ZERO LEAKAGE</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Automated Test Suite</b>", table_cell_bold),
            Paragraph("Full Backend Pytest Suite", table_cell),
            Paragraph("<b>529 passed, 0 failed</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>100% PASS</b></font>", table_cell
            ),
        ],
        [
            Paragraph("<b>Evaluation Determinism</b>", table_cell_bold),
            Paragraph("3 Consecutive Evaluation Passes", table_cell),
            Paragraph("<b>100.0% Deterministic</b>", table_cell_bold),
            Paragraph("<font color='#15803d'><b>STABLE</b></font>", table_cell),
        ],
        [
            Paragraph("<b>P50 / P95 Latency</b>", table_cell_bold),
            Paragraph("Full End-to-End Resolution Pipeline", table_cell),
            Paragraph("<b>127.63 ms / 1,039.67 ms</b>", table_cell_bold),
            Paragraph(
                "<font color='#15803d'><b>REAL-TIME</b></font>", table_cell
            ),
        ],
    ]
    t_res = Table(results_table, colWidths=[130, 154, 130, 90])
    t_res.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t_res)
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 19: SECURITY AND LEAKAGE PREVENTION
    # =========================================================================
    story.append(HeadingParagraph("19. Security and Leakage Prevention", h1_style, 19, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI enforces cryptographic and architectural safeguards to protect scientific integrity and data privacy:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Cryptographic Benchmark Immutability:</b> The 200-case golden dataset is protected by SHA-256 checksum (<code>1d3e9b3b8bdef3750437e59b17ab7c27167fd1548c2bb29de984151296c5b45a</code>). Pre- and post-test assertions verify byte-for-byte immutability across all runs.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Corpus Zero-Leakage Quarantine:</b> All 200 golden evaluation conversation IDs are programmatically excluded from the historical retrieval index. Automated test fixtures assert zero overlapping IDs (<code>benchmark_leakage == 0</code>).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Candidate Evidence Isolation:</b> New human resolutions are captured into <code>data/evidence_candidates/</code> and cannot be queried by the retrieval engine until validated, scored (≥ 0.80), and promoted.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Automated PII & Secret Redaction:</b> A 4-pattern regex engine automatically strips API keys, auth tokens, email addresses, and phone numbers before persisting audit logs or decision traces.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 20: REPRODUCIBILITY
    # =========================================================================
    story.append(HeadingParagraph("20. Reproducibility", h1_style, 20, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI is designed for complete, deterministic reproducibility across development and production environments:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Environment Setup:</b> Python 3.9+ virtual environment with dependencies installed from <code>backend/requirements.txt</code>. Configuration managed through <code>.env</code> mirroring <code>.env.example</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Service Execution:</b> Backend launched via <code>python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000</code>. Frontend launched via <code>npm run dev</code> from <code>frontend/</code>.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Test Suite Execution:</b> Automated test suite executed via <code>pytest backend/tests/ -v</code>, running 529 automated tests.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Production Evaluation Execution:</b> Multi-pass evaluation and release readiness checks executed via <code>python -m backend.scripts.evaluate_phase_12_1</code> and <code>python -m backend.scripts.check_phase_12_release</code>.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 21: LIMITATIONS
    # =========================================================================
    story.append(HeadingParagraph("21. Limitations", h1_style, 21, recorded_pages))
    story.append(
        Paragraph(
            "While SupportGraph AI achieves high precision and rigorous safety gating, several operational boundaries are documented:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1. <b>Domain and Corpus Scope:</b> Historical retrieval is grounded in AppleSupport Twitter interaction data. Inquiries regarding non-Apple operating systems (e.g. Android firmware) or specialized enterprise Mobile Device Management (MDM) deployment profiles default to safe human escalation.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "2. <b>Third-Party Inference Dependencies & Fallback Behavior:</b> When external LLM inference providers experience network latency or rate limits, the system activates calibrated deterministic rule-based heuristics. While safety gating remains 100% intact, response phrasing relies on canonical troubleshooting templates rather than dynamically synthesized text.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "3. <b>Dialogue Modality:</b> Current dialogue state management supports text-based customer interactions. Inquiries requiring visual diagnostic inspection (e.g. photos of cracked glass or physical water indicator stickers) require specialist visual review.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 22: FINAL SYSTEM SUMMARY
    # =========================================================================
    story.append(HeadingParagraph("22. Final System Summary", h1_style, 22, recorded_pages))
    story.append(
        Paragraph(
            "SupportGraph AI demonstrates that high-stakes enterprise customer support automation requires moving beyond ungrounded generative chatbots. By treating customer support as an evidence verification and safety-gating challenge, the platform delivers:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Evidence-Grounded Resolution:</b> Grounded in 80,487 verified historical brand interaction chains, eliminating unsupported claims.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Deterministic Safety Gating:</b> Hardware and thermal hazards are intercepted with 0 unsafe auto-handles across all evaluated benchmarks.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Operational Diagnostic Precedence:</b> Symptoms are decoupled from causal triggers across 20 operational problem families.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Pre-Delivery Verification:</b> Candidate claims are verified against retrieved evidence before customer dispatch.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Stateful Multi-Turn Troubleshooting:</b> Tracks actions, normalizes colloquial phrasing, and prevents repetitive advice.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Live Specialist Integration:</b> Routes live escalations in real-time with comprehensive diagnostic packages.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Governed Continuous Improvement:</b> Ingests human feedback through an offline-evaluated promotion lifecycle.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Production Observability:</b> Provides 12-step decision traces, operational metrics, and PII sanitization.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "With <b>529 passing automated tests</b>, <b>zero benchmark leakage</b>, <b>100% auto-handle precision</b>, and <b>zero safety violations</b>, SupportGraph AI establishes a dependable architectural foundation for autonomous enterprise customer support intelligence.",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    return recorded_pages


def main() -> int:
    output_pdf = Path("reports/SupportGraph_AI_Technical_Report.pdf")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    print("Phase 1: Compiling PDF to discover exact section starting pages...")
    initial_pages = {sec_id: 1 for sec_id, _ in SECTIONS_METADATA}
    recorded_pages = build_pdf(output_pdf, initial_pages)

    print(f"Recorded section pages from Pass 1: {recorded_pages}")

    print(
        "Phase 2: Recompiling PDF with verified page numbers in Table of Contents..."
    )
    final_pages = build_pdf(output_pdf, recorded_pages)
    print(f"Verified section pages from Pass 2: {final_pages}")

    # Verify that TOC page numbers match final layout
    mismatches = []
    for sec_id, title in SECTIONS_METADATA:
        expected = recorded_pages.get(sec_id)
        actual = final_pages.get(sec_id)
        if expected != actual:
            mismatches.append(
                f"Section {sec_id} ({title}): expected page {expected}, got page {actual}"
            )

    if mismatches:
        print(f"Warning: Minor TOC page shifts detected: {mismatches}")
        print("Running convergence Pass 3...")
        final_pages = build_pdf(output_pdf, final_pages)

    print(f"Successfully generated {output_pdf} ({output_pdf.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
