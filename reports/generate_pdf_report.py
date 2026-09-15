#!/usr/bin/env python3
"""
SupportGraph AI — Technical Report PDF Generator
Generates reports/SupportGraph_AI_Technical_Report.pdf based on the uploaded reference PDF report.
Strictly black text on white background, exactly ONE technical diagram (Figure 1),
NO decorative boxes outside the diagram, NO Table of Contents.
Target length: Exactly 6 well-filled pages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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


class NumberedCanvas(canvas.Canvas):
    """
    Canvas that adds monochrome running headers and centered 'Page X' footers.
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
        self.saveState()
        self.setFont("Helvetica-Oblique", 8)
        self.setFillColor(colors.black)
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.5)

        # Running header on all pages
        self.drawRightString(
            letter[0] - 48,
            752,
            "SupportGraph AI — Technical Project Report",
        )
        self.line(48, 745, letter[0] - 48, 745)

        # Running footer: centered "Page X"
        self.setFont("Helvetica", 9)
        page_str = f"Page {self._pageNumber}"
        self.drawCentredString(letter[0] / 2.0, 32, page_str)

        self.restoreState()


def build_pdf(output_path: Path) -> int:
    # Printable area: 612 - 96 = 516 pt width; 792 - 92 = 700 pt height
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    # MONOCHROME TYPOGRAPHY
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceAfter=2,
    )
    report_tag_style = ParagraphStyle(
        "ReportTag",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "ReportH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.black,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.6,
        textColor=colors.black,
        spaceAfter=4.5,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.6,
        textColor=colors.black,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3.5,
    )
    caption_style = ParagraphStyle(
        "DiagramCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.black,
        alignment=1,  # Centered
        spaceBefore=3,
        spaceAfter=5,
    )
    diagram_box_style = ParagraphStyle(
        "DiagramBox",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.black,
        alignment=1,  # Centered
    )
    diagram_arrow_style = ParagraphStyle(
        "DiagramArrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=9.0,
        textColor=colors.black,
        alignment=1,  # Centered
    )
    footprint_style = ParagraphStyle(
        "FootprintStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.8,
        leading=12.0,
        textColor=colors.black,
        spaceBefore=8,
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, 01 — PROJECT OVERVIEW, DIAGRAM (FIGURE 1)
    # =========================================================================
    story.append(Spacer(1, 4))
    story.append(Paragraph("SupportGraph AI", title_style))
    story.append(
        Paragraph(
            "Evidence-Grounded, Safety-Gated Apple Customer Support Resolution System",
            subtitle_style,
        )
    )
    story.append(Paragraph("Technical Project Report", report_tag_style))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.black,
            spaceBefore=2,
            spaceAfter=8,
        )
    )

    story.append(Paragraph("01 — PROJECT OVERVIEW", h1_style))
    story.append(
        Paragraph(
            "SupportGraph AI is a customer-support resolution system built from real AppleSupport conversations. The core engineering problem is not simply generating a helpful sentence; it is determining whether a proposed answer is supported by historical evidence and safe to automate.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The implementation separates problem understanding, intent classification, evidence retrieval, evidence validation, response generation, response verification, safety decisions, multi-turn state and human review. This creates an auditable chain between the customer's problem and the final resolution path.",
            body_style,
        )
    )
    story.append(Spacer(1, 3))

    # EXACT SINGLE TECHNICAL DIAGRAM (FIGURE 1)
    box_w = 260
    flow_data = [
        [Paragraph("CUSTOMER QUERY", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("PROBLEM UNDERSTANDING", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("AMBIGUITY / SAFETY GATE", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("HISTORICAL EVIDENCE RETRIEVAL", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("EVIDENCE VALIDATION", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("RESPONSE GENERATION", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("RESPONSE VERIFICATION", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("AUTO-HANDLE / HUMAN REVIEW", diagram_box_style)],
        [Paragraph("↓", diagram_arrow_style)],
        [Paragraph("RESOLUTION + FEEDBACK", diagram_box_style)],
    ]
    t_flow = Table(flow_data, colWidths=[box_w])
    t_flow.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (0, 0), 0.75, colors.black),
                ("BOX", (0, 2), (0, 2), 0.75, colors.black),
                ("BOX", (0, 4), (0, 4), 0.75, colors.black),
                ("BOX", (0, 6), (0, 6), 0.75, colors.black),
                ("BOX", (0, 8), (0, 8), 0.75, colors.black),
                ("BOX", (0, 10), (0, 10), 0.75, colors.black),
                ("BOX", (0, 12), (0, 12), 0.75, colors.black),
                ("BOX", (0, 14), (0, 14), 0.75, colors.black),
                ("BOX", (0, 16), (0, 16), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (0, -1), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
            ]
        )
    )

    # Center table horizontally
    t_flow_centered = Table([[t_flow]], colWidths=[516])
    t_flow_centered.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(t_flow_centered)
    story.append(Spacer(1, 2))
    story.append(Paragraph("Figure 1: End-to-End Project Technical Flow", caption_style))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "<i>Key design principle: the LLM proposes a response; evidence, verification and safety controls decide whether automation is allowed.</i>",
            body_style,
        )
    )

    # =========================================================================
    # PAGE 2: 02 — PROBLEM STATEMENT, 03 — WHAT WAS IMPLEMENTED, 04 — ARCHITECTURE
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("02 — PROBLEM STATEMENT", h1_style))
    story.append(
        Paragraph(
            "Support conversations are noisy: one message can contain several symptoms, missing context, misleading keywords or a worsening safety condition. A retrieval result can look similar while solving a different problem. A fluent LLM answer can therefore be wrong even when it sounds convincing.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "SupportGraph AI was built to make those failure modes explicit and controllable instead of hiding them behind a single generation step.",
            body_style,
        )
    )

    story.append(Paragraph("03 — WHAT WAS IMPLEMENTED", h1_style))
    story.append(
        Paragraph(
            "The system is implemented as cooperating components, with each component responsible for a specific decision or control. This makes the resolution path inspectable and testable.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Problem understanding:</b> ProblemExtractor, PrimaryProblemSelector, TurnClassifier and ClarificationEngine convert free-form support language into an operational problem representation.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Ambiguity and safety:</b> AmbiguityAnalyzer and AmbiguityDecisionGate determine whether the system has enough information to proceed. Safety logic can veto normal troubleshooting when critical hardware or thermal signals appear.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Retrieval and evidence:</b> CaseRetriever, HistoricalCorpusIndex and EvidenceRanker retrieve historical precedents. MultiCaseEvidenceSynthesizer, EvidenceConflictDetector and EvidenceValidator determine whether those precedents actually support the current case.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Response control:</b> ResponseGenerator drafts the answer. ResponseVerifier checks symptom alignment, unsupported claims, cause decoupling and evidence corroboration. ResolutionAuditor records the resulting decision.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Conversation control:</b> ConversationState, ConversationManager, ActionTracker and ResolutionProgressEngine preserve context, distinguish confirmed from inferred facts and prevent repeated troubleshooting.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Governance and observability:</b> human-feedback stores, controlled promotion, DecisionTrace, DecisionTraceStore and SystemMetricsAuditor provide evidence governance and operational auditability.",
            bullet_style,
        )
    )

    story.append(Paragraph("04 — ARCHITECTURE", h1_style))
    story.append(
        Paragraph(
            "The pipeline moves sequentially through eight control stages: understanding, classification, the ambiguity/safety gate, retrieval, evidence validation, response generation, response verification and the final automation decision. Each stage is implemented as an independent, inspectable component rather than a single end-to-end model call, so the system's behavior at any point in the chain can be examined on its own terms.",
            body_style,
        )
    )

    # =========================================================================
    # PAGE 3: 04 (CONT), 05 — DATASET, 06 — INTENT & RETRIEVAL, 07 — EVIDENCE VALIDATION
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph(
            "The important architectural boundary is between retrieval and evidence validation. Retrieved cases are candidates for evidence, not evidence by default. The response is then verified after generation before an automated outcome can be returned.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Because each checkpoint is separate, a failure can be traced back to the specific stage that caused it instead of being hidden inside a single opaque generation step. This also means one stage can be corrected or extended without redesigning the rest of the pipeline: the retrieval index can be expanded, or the evidence validator's thresholds can be tightened, without changing how responses are generated or how the safety gate evaluates hardware risk. This modularity is what later allowed real failures to be converted into targeted, stage-specific engineering controls rather than broad, unverifiable fixes.",
            body_style,
        )
    )

    story.append(Paragraph("05 — DATASET AND EVALUATION INTEGRITY", h1_style))
    story.append(
        Paragraph(
            "The selected brand is AppleSupport. Brand selection identified 106,719 usable interactions with a selection score of 0.8645, response coverage of 99.93%, reconstructability of 100%, completeness of 0.9997 and diversity of 0.9575.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The final historical retrieval index contains 80,487 non-golden support conversations. A separate frozen set of 200 golden records is reserved for evaluation and excluded from retrieval. This separation protects the benchmark from retrieval leakage.",
            body_style,
        )
    )

    story.append(Paragraph("06 — INTENT UNDERSTANDING AND RETRIEVAL", h1_style))
    story.append(
        Paragraph(
            "The final implementation uses 20 operational problem families to make support issues retrievable and comparable. The system combines problem understanding with contextual signals instead of relying only on surface word overlap.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Retrieval is deliberately conservative. Lexical decoy penalties reduce cases that share words but represent a different support problem. Multiple compatible precedents can be synthesized, but synthesis remains downstream of evidence validation.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<i>Important distinction: semantic similarity is a retrieval signal; it is not proof that a historical resolution applies to the current customer.</i>",
            body_style,
        )
    )

    story.append(Paragraph("07 — EVIDENCE VALIDATION", h1_style))
    story.append(
        Paragraph(
            "Validation compares each retrieved case against the current conversation across several dimensions, including device, operating system or service, primary symptom, underlying cause and customer intent. A case that matches on surface wording but diverges on device or symptom is treated as weak or insufficient evidence rather than being accepted on similarity alone.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Evidence is assessed as STRONG_EVIDENCE, MODERATE_EVIDENCE, WEAK_EVIDENCE, CONFLICTING_EVIDENCE or INSUFFICIENT_EVIDENCE. The verdict determines whether the case can support an automated response, requires corroboration or must be escalated.",
            body_style,
        )
    )

    # =========================================================================
    # PAGE 4: 07 (CONT), 08 — RESPONSE VERIFICATION, 09 — SAFETY, 10 — MULTI-TURN, 11
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph(
            "When multiple retrieved cases point to different causes or resolutions for what looks like the same problem, the evidence is marked as CONFLICTING_EVIDENCE and cannot support an automated response on its own. This distinction between similarity and applicability is what allows the system to separate a resolution that merely looks relevant from one that is actually supported by precedent.",
            body_style,
        )
    )

    story.append(Paragraph("08 — RESPONSE GENERATION AND VERIFICATION", h1_style))
    story.append(
        Paragraph(
            "Only validated evidence is passed into response construction. The generated response is then checked for symptom alignment, unsupported claims, cause decoupling and corroboration with the retrieved evidence.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "A failed verification becomes a verification veto and prevents unsafe auto-handling. This is a key difference from a direct LLM chatbot where generation itself may be treated as the final answer.",
            body_style,
        )
    )

    story.append(Paragraph("09 — SAFETY-GATED AUTOMATION AND HUMAN REVIEW", h1_style))
    story.append(
        Paragraph(
            "Automation is allowed only when the problem is sufficiently understood, usable evidence exists, the response passes verification, and no ambiguity or safety condition requires specialist intervention.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Escalation decisions include HUMAN_REQUIRED, INSUFFICIENT_INFORMATION, VERIFICATION_VETO, EVIDENCE_LIMITED, GENUINE_AMBIGUITY, MULTI_PROBLEM_COMPLEXITY and RECOVERABLE_ESCALATION.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "If any required gate fails, the path moves to human review rather than forcing an automated resolution.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Safety-critical example: a customer reports rapidly draining battery followed by extreme heat, smoke and cracking back glass. The worsening thermal/hardware signals trigger urgent human escalation rather than continuing ordinary troubleshooting.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "A second adversarial case combined a duplicate billing charge with Apple ID lockout. Because the system could not safely verify an appropriate account or refund action, it escalated instead of inventing a financial workflow.",
            body_style,
        )
    )

    story.append(Paragraph("10 — MULTI-TURN RESOLUTION", h1_style))
    story.append(
        Paragraph(
            "The conversation layer maintains confirmed facts separately from inferred facts, tracks canonical troubleshooting actions, recognizes semantic aliases of previously attempted steps, and limits clarification to one decision-critical question per turn.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Verified context retention was 100% and repeat prevention was 100%. Original adversarial scenarios achieved 30/30 passes; the unseen adversarial set achieved 20/20 passes.",
            body_style,
        )
    )

    story.append(Paragraph("11 — HUMAN FEEDBACK AND EVIDENCE GOVERNANCE", h1_style))

    # =========================================================================
    # PAGE 5: 11 (CONT), 12 — OBSERVABILITY, 13 — FAILURES, 14 — EVALUATION INTRO
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph(
            "Human feedback is not written directly into live evidence. It enters a controlled lifecycle so that specialist corrections can be reviewed, evaluated and traced before influencing approved evidence.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Candidate and approved evidence stores remain separate. Provenance, versioning, offline benchmark simulation and leakage checks protect the system from an unsafe feedback loop. There is no automatic online promotion.",
            body_style,
        )
    )

    story.append(Paragraph("12 — OBSERVABILITY AND AUDITABILITY", h1_style))
    story.append(
        Paragraph(
            "The system records a 12-step decision trace covering the important stages of the resolution path. Runtime monitoring includes latency, health, alerts, evidence provenance, feedback monitoring, auditability and PII/secret redaction.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Because every gate decision is written to the trace, a specialist reviewing an escalated or automated case can reconstruct exactly why the system classified the intent the way it did, which evidence it accepted or rejected, and which check ultimately allowed or blocked automation. This turns observability from a passive log into an active audit trail that supports both individual case review and system-wide quality checks.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "The final observability suite reported 33/33 tests passing. The broader implementation was also validated through regression, adversarial and evaluation gates.",
            body_style,
        )
    )

    story.append(Paragraph("13 — REAL FAILURES CONVERTED INTO ENGINEERING CONTROLS", h1_style))
    story.append(
        Paragraph(
            "• <b>AirPods pairing recognition gap:</b> the system initially missed some pairing expressions. The mapping was expanded to cover AirPods, EarPods, headphones, Bluetooth and pairing terminology, followed by regression testing.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Thermal hazard detection gap:</b> worsening-device language was expanded to include smoking, burning, swelling, extreme heat and fire-related signals. A global hardware/thermal safety veto was added.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Billing + account-lockout decoy:</b> verification correctly blocked an unsafe financial/account workflow and escalated the case.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "• <b>Retrieval coverage gap:</b> retrieval coverage was expanded through the 20 operational problem families and the 80,487-case historical index.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "The important point is that failures became new controls and regression/adversarial tests, not merely manual exceptions.",
            body_style,
        )
    )

    story.append(Paragraph("14 — EVALUATION RESULTS", h1_style))
    story.append(
        Paragraph(
            "The final evaluation measures both model/system capability and the safety of the automation boundary.",
            body_style,
        )
    )

    # =========================================================================
    # PAGE 6: 14 (CONT), 15 — WHY DIFFERENT, 16 — TECHNICAL CONTRIBUTION, FOOTPRINT
    # =========================================================================
    story.append(PageBreak())
    story.append(
        Paragraph(
            "<b>Intent and problem understanding:</b> primary intent accuracy 88.3%, top-2 intent accuracy 96.1%, and problem family accuracy 94.8%.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Evidence quality:</b> usable evidence coverage 89.0% and direct problem match 84.0%.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Automation safety:</b> safe auto-handle rate 60.5%, auto-handle precision 100%, unsafe auto-handles 0.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Integrity:</b> evaluation leakage 0; context retention 100%; repeat prevention 100%.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Adversarial validation:</b> 90.0% overall pass rate, with 30/30 original and 20/20 unseen adversarial scenarios passing.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Latency:</b> P50 80.7 ms, P95 414.2 ms, P99 480.8 ms.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Additional verification recorded 100% safe escalation of verification failures and complete handling of tested conflict and ambiguity cases.",
            body_style,
        )
    )

    story.append(Paragraph("15 — WHY THIS SYSTEM IS DIFFERENT", h1_style))
    story.append(
        Paragraph(
            "The uniqueness is the control architecture around the language model. SupportGraph AI does not equate retrieval with evidence, generation with correctness, or confidence with permission to automate. It creates separate gates for understanding, evidence quality, response grounding, safety, conversation state and human governance.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "In a conventional support chatbot, a single generation step is often treated as the final answer: if the response sounds fluent and confident, it is returned to the customer regardless of whether it is grounded in anything verifiable. SupportGraph AI instead treats generation as only one stage in a longer decision process, where a fluent-sounding response can still be rejected if it fails evidence validation or response verification.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "This distinction matters most in edge cases: a worsening thermal condition, a billing dispute tied to an account lockout, or retrieved precedents that disagree with each other. A direct-generation system has no built-in mechanism to recognize that it should stop and defer to a human in these situations, whereas SupportGraph AI's gates are specifically designed to catch these conditions before an automated response is ever sent.",
            body_style,
        )
    )

    story.append(Paragraph("16 — FINAL TECHNICAL CONTRIBUTION", h1_style))
    story.append(
        Paragraph(
            "SupportGraph AI connects a customer's problem to historical support evidence, validates that evidence against the current case, generates a proposed resolution, verifies the response, and then makes a governed decision between automated handling and human escalation.",
            body_style,
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        Paragraph(
            "<b><i>Customer problem → structured understanding → historical evidence → validated resolution → verified response → safe automation or human escalation.</i></b>",
            ParagraphStyle(
                "ChainStyle",
                parent=styles["Normal"],
                fontName="Helvetica-BoldOblique",
                fontSize=9.0,
                leading=13.0,
                textColor=colors.black,
                alignment=1,  # Centered
            ),
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        Paragraph(
            "The result is a support system that can be inspected, tested and improved through evidence rather than relying on fluent generation alone.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=colors.black,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    story.append(
        Paragraph(
            "<b>Implementation footprint:</b> 80,487 indexed historical cases · 20 operational problem families · multi-turn state tracking · evidence validation · response verification · safety escalation · governed human feedback · 12-step decision tracing.",
            footprint_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    return doc.page


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_pdf = repo_root / "reports" / "SupportGraph_AI_Technical_Report.pdf"

    print(f"Generating PDF report: {output_pdf}")
    build_pdf(output_pdf)

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(output_pdf))
        page_count = len(reader.pages)
        print(f"Successfully generated {output_pdf}")
        print(f"  Page Count: {page_count} pages")
        print(f"  File Size:  {output_pdf.stat().st_size} bytes")

        if page_count != 6:
            print(f"ERROR: PDF page count is {page_count}, expected exactly 6 pages!")
            sys.exit(1)
        else:
            print("PERFECT: PDF page count is exactly 6 pages.")

    except ImportError:
        print("Note: pypdf not installed for page verification.")


if __name__ == "__main__":
    main()
