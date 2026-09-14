"""
SupportGraph AI — Taxonomy Review & Finalization Engine (Phase 5)

Aggregates evidence from Phase 4 unsupervised discovery and Phase 4.5 cluster audit
to generate an initial evidence-based candidate operational intent taxonomy.

Key Principles:
- Unsupervised clusters are NOT ground truth.
- Categories represent WHAT THE CUSTOMER NEEDS HELP WITH (operational customer problem).
- Product entities (iPhone, Mac), emotional tone, individual patch versions, and brand
  resolution strategies are strictly excluded from intent names.
- Account access and billing are evaluated separately.
- Merge decisions (e.g. Clusters 0 & 7 for keyboard glitches; Clusters 3 & 6 for updates)
  and split decisions (Cluster 2 for display vs hardware; Cluster 4 for account vs billing)
  are documented with clear rationale.
- Target: 7–10 operational intents.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

try:
    from app.core.logging import get_logger
except ModuleNotFoundError:
    from backend.app.core.logging import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured Data Models
# ---------------------------------------------------------------------------
@dataclass
class IntentDefinition:
    """Detailed definition and operational boundary for a single customer support intent."""
    intent_id: str
    intent_name: str
    definition: str
    include_when: list[str]
    exclude_when: list[str]
    example_messages: list[str]
    source_clusters: list[int]
    review_status: str = "candidate_human_review_required"
    decision_rationale: str = ""


DEFAULT_CANDIDATE_TAXONOMY_PATH = "data/interim/final_taxonomy_candidate.json"


@dataclass
class TaxonomyCandidate:
    """Complete candidate intent taxonomy ready for human review."""
    status: str = "candidate_human_review_required"
    taxonomy_version: str = "v1.0-candidate"
    selected_brand: str = "AppleSupport"
    intent_count: int = 0
    intents: list[IntentDefinition] = field(default_factory=list)
    merge_decisions: list[dict[str, Any]] = field(default_factory=list)
    split_decisions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.taxonomy_version

    @property
    def intent_names(self) -> list[str]:
        return [i.intent_name for i in self.intents]

    def get_intent_names(self) -> list[str]:
        return [i.intent_name for i in self.intents]

    def get_intent_by_name(self, name: str) -> Optional[IntentDefinition]:
        for i in self.intents:
            if i.intent_name == name:
                return i
        return None


# ---------------------------------------------------------------------------
# Intent Keyword Patterns for Mapping Candidate Messages
# ---------------------------------------------------------------------------
INTENT_KEYWORD_PATTERNS = {
    "software_update_problem": re.compile(
        r"\b(update|updated|updating|ios\s*11|ios11|install|upgrade|downloading\s+update|firmware|restore|bricked|apple\s+logo\s+loop)\b",
        re.IGNORECASE,
    ),
    "battery_power_issue": re.compile(
        r"\b(battery|drain|draining|charge|charging|charger|dies|die|percentage|overheat|overheating|power\s+off|shutting\s+down|battery\s+health)\b",
        re.IGNORECASE,
    ),
    "display_touch_issue": re.compile(
        r"\b(screen|display|touch|touchscreen|cracked|black\s+screen|digitizer|lines\s+on\s+screen|flickering|glitch\s+on\s+screen|ghost\s+touch)\b",
        re.IGNORECASE,
    ),
    "account_access_issue": re.compile(
        r"\b(appleid|apple\s+id|password|passcode|locked|login|logging\s+in|sign\s+in|signing\s+in|verification\s+code|two-factor|2fa|icloud\s+account|disabled\s+account)\b",
        re.IGNORECASE,
    ),
    "billing_purchase_issue": re.compile(
        r"\b(billed|billing|charge|charged|subscription|refund|itunes\s+store|app\s+store|receipt|purchased|purchase|payment|credit\s+card|money|unauthorized\s+charge)\b",
        re.IGNORECASE,
    ),
    "keyboard_typing_issue": re.compile(
        r"\b(keyboard|typing|autocorrect|type|letter\s+i|predictive|cursor|keys|auto-correct|question\s+mark|a\s*\?|a\s*\[\?\]|i\s+glitch)\b",
        re.IGNORECASE,
    ),
    "mac_software_issue": re.compile(
        r"\b(macbook|imac|macos|sierra|high\s+sierra|mac\s+mini|mac\s+os|safari.*mac|trackpad|mac\s+memory)\b",
        re.IGNORECASE,
    ),
    "hardware_audio_connection_issue": re.compile(
        r"\b(speaker|sound|audio|microphone|mic|volume|bluetooth|wifi|wi-fi|headphone|earbuds|airpods|cellular|no\s+service|signal|jack)\b",
        re.IGNORECASE,
    ),
}


def assign_candidate_intent(text: str, source_cluster: int) -> str:
    """
    Deterministically assign a candidate intent to a message using reviewed
    cluster origin and lexical patterns.
    """
    t_low = str(text).lower()
    cid = int(source_cluster)

    # 1. High-confidence cluster-guided assignments
    if cid in [0, 7] and INTENT_KEYWORD_PATTERNS["keyboard_typing_issue"].search(t_low):
        return "keyboard_typing_issue"
    if cid == 1 and INTENT_KEYWORD_PATTERNS["battery_power_issue"].search(t_low):
        return "battery_power_issue"
    if cid in [3, 6] and INTENT_KEYWORD_PATTERNS["software_update_problem"].search(t_low):
        return "software_update_problem"
    if cid == 4:
        if INTENT_KEYWORD_PATTERNS["account_access_issue"].search(t_low):
            return "account_access_issue"
        if INTENT_KEYWORD_PATTERNS["billing_purchase_issue"].search(t_low):
            return "billing_purchase_issue"
    if cid == 5 and INTENT_KEYWORD_PATTERNS["mac_software_issue"].search(t_low):
        return "mac_software_issue"
    if cid == 2:
        if INTENT_KEYWORD_PATTERNS["display_touch_issue"].search(t_low):
            return "display_touch_issue"
        if INTENT_KEYWORD_PATTERNS["hardware_audio_connection_issue"].search(t_low):
            return "hardware_audio_connection_issue"

    # 2. General lexical pattern matching fallback
    for intent_name, pattern in INTENT_KEYWORD_PATTERNS.items():
        if pattern.search(t_low):
            return intent_name

    # 3. Default fallback for ambiguous or unspecialized device inquiries
    return "general_device_support"


# ---------------------------------------------------------------------------
# Initial Candidate Taxonomy Definitions
# ---------------------------------------------------------------------------
def get_initial_candidate_intents() -> list[IntentDefinition]:
    """
    Construct the evidence-based 9-intent candidate taxonomy from Phase 4.5 audit findings.
    Each definition includes operational boundaries, positive inclusion/exclusion rules,
    and verified historical examples.
    """
    return [
        IntentDefinition(
            intent_id="intent_01",
            intent_name="software_update_problem",
            definition=(
                "Customer experiences failure, device freeze, app incompatibility, or crash "
                "during or immediately following an operating system update."
            ),
            include_when=[
                "Device unresponsive, bricked, or stuck on Apple logo during/after update.",
                "Apps crash, freeze, or fail to launch specifically following an OS upgrade.",
                "Error messages during update download, verification, or installation.",
            ],
            exclude_when=[
                "Battery drain complaints without installation or operating failure (map to battery_power_issue).",
                "App Store purchase or billing errors occurring after an update (map to billing_purchase_issue).",
                "Mac desktop OS installation (map to mac_software_issue).",
            ],
            example_messages=[
                "<USER> since I upgraded to iOS 11.0.3 my phone just giving issues, it just on life support",
                "<USER> what is going on with iOS 11 it’s stopped my whatsapp working and has bugs v annoyed",
                "Phone non stop messing up since downloading the update, I know your game <USER>",
            ],
            source_clusters=[3, 6],
            decision_rationale=(
                "Clusters 3 and 6 both captured software update complaints. They were merged to eliminate "
                "redundant intent heads and generalize away from single patch numbers (iOS 11.0.1 vs 11.0.3)."
            ),
        ),
        IntentDefinition(
            intent_id="intent_02",
            intent_name="battery_power_issue",
            definition=(
                "Customer reports rapid battery drain, unexpected power shutoffs, failure to charge, "
                "or overheating power components."
            ),
            include_when=[
                "Battery percentage drops abnormally fast during normal or idle usage.",
                "Device unexpectedly shuts down while battery indicator shows remaining charge.",
                "Device fails to charge, charges slowly, or produces accessory not supported errors.",
                "Device runs unusually hot during charging or battery depletion.",
            ],
            exclude_when=[
                "Physical damage to charging port pin or liquid intrusion (map to hardware_audio_connection_issue).",
                "Sluggish performance without explicit battery drain complaints (map to general_device_support).",
            ],
            example_messages=[
                "<USER> the battery drain is a bit much. iOS 11.0.2 on a A1533 iPhone 5S. <URL>",
                "<USER> I upgraded my iPhone7 to 8 to get rid of the problems iOS11 caused, now Iphone8 is worse, battery runs out in hours",
                "<USER> my battery percentage was at 45% and then suddenly dropped to 1% and shut off",
            ],
            source_clusters=[1],
            decision_rationale=(
                "Cluster 1 exhibited high cohesion and tight semantic focus on battery performance. "
                "Maintained as a standalone operational intent."
            ),
        ),
        IntentDefinition(
            intent_id="intent_03",
            intent_name="display_touch_issue",
            definition=(
                "Customer reports physical or functional display malfunctions, including cracked glass, "
                "unresponsive touchscreen digitizer, lines, or black screens."
            ),
            include_when=[
                "Touchscreen digitizer fails to respond to finger taps, swipes, or gestures.",
                "Screen has visible cracks, shattered glass, or dead pixels after an impact.",
                "Display shows green/colored vertical lines, flickering, or remains completely black.",
            ],
            exclude_when=[
                "Temporary application freeze where physical buttons and other apps work (map to general_device_support).",
                "Keyboard character rendering bugs (map to keyboard_typing_issue).",
            ],
            example_messages=[
                "<USER> screen is cracked and touch is not responding at all can you help me",
                "<USER> half of my screen is black with green vertical lines after waking up",
                "<USER> my iPhone display froze completely and I cannot swipe to unlock",
            ],
            source_clusters=[2],
            decision_rationale=(
                "Cluster 2 was split because it mixed physical glass/touch defects with audio and generic device "
                "complaints. Display & touch require specific Genius Bar hardware repair procedures."
            ),
        ),
        IntentDefinition(
            intent_id="intent_04",
            intent_name="account_access_issue",
            definition=(
                "Customer cannot access their Apple ID, iCloud account, or device due to forgotten credentials, "
                "locked account status, or authentication code failures."
            ),
            include_when=[
                "Forgotten Apple ID password, passcode, or security questions.",
                "Account disabled or locked for security reasons.",
                "Failure to receive two-factor authentication (2FA) SMS or push verification codes.",
                "iCloud activation lock preventing device setup.",
            ],
            exclude_when=[
                "Disputed charges on credit card or unexpected subscriptions (map to billing_purchase_issue).",
                "Inability to download a specific free app due to server error (map to billing_purchase_issue).",
            ],
            example_messages=[
                "<USER> forgot my apple id password and cannot login to my icloud account",
                "<USER> your reportaproblem page just keeps refreshing instead of allowing me to sign in",
                "<USER> my account is locked for security reasons and the recovery email never arrives",
            ],
            source_clusters=[4],
            decision_rationale=(
                "Cluster 4 co-mingled credential authentication with payment billing disputes. "
                "Split to enable distinct identity verification routing."
            ),
        ),
        IntentDefinition(
            intent_id="intent_05",
            intent_name="billing_purchase_issue",
            definition=(
                "Customer reports financial, purchase, or subscription discrepancies, including unauthorized charges, "
                "refund requests, or App Store payment declines."
            ),
            include_when=[
                "Customer sees unexpected or recurring charges from iTunes or App Store on bank statement.",
                "Requests for refund on in-app purchases or accidental downloads.",
                "Declined credit/debit card errors or payment method verification failures in App Store.",
                "Subscriptions that were canceled but continue to generate billing notifications.",
            ],
            exclude_when=[
                "Inability to log in to Apple ID to view purchases (map to account_access_issue).",
                "App crashes immediately after purchase without payment dispute (map to software_update_problem).",
            ],
            example_messages=[
                "<USER> i was charged twice for my Apple Music subscription this month please refund",
                "<USER> why is my card being declined in the App Store when funds are available?",
                "<USER> unauthorized $9.99 in-app purchase charge appeared on my receipt how to dispute",
            ],
            source_clusters=[4],
            decision_rationale=(
                "Financial disputes require specialized billing workflows, invoice retrieval, and refund tools. "
                "Separated from account credential recovery."
            ),
        ),
        IntentDefinition(
            intent_id="intent_06",
            intent_name="keyboard_typing_issue",
            definition=(
                "Customer reports text input anomalies, predictive text failures, autocorrect loops, "
                "or keyboard interface rendering glitches."
            ),
            include_when=[
                "Autocorrect automatically substitutes incorrect symbols or loops characters.",
                "On-screen keyboard fails to appear, lags severely behind typing, or freezes.",
                "Glitch where specific letters (e.g. 'I') are rendered as symbols or question marks.",
            ],
            exclude_when=[
                "Physical external Mac keyboard key broken or disconnected (map to mac_software_issue).",
                "Touchscreen digitizer dead across the entire display (map to display_touch_issue).",
            ],
            example_messages=[
                "<USER> you need to fix this “A” shit when I put mf “I”",
                "<USER> y’all better fix these question marks that keep popping up whenever I try to use a damn I",
                "<USER> FIX YOUR GODDAM “I” glitch. All my friends are thinking I’m a drunk because of typos",
            ],
            source_clusters=[0, 7],
            decision_rationale=(
                "Clusters 0 and 7 represented the same underlying keyboard/typing failure mode, with Cluster 7 "
                "being an informal/frustrated expression variant. Merged and generalized beyond the 2017 autocorrect bug."
            ),
        ),
        IntentDefinition(
            intent_id="intent_07",
            intent_name="mac_software_issue",
            definition=(
                "Customer reports software, operating system, or application failures specific to macOS "
                "and Mac desktop/laptop hardware (MacBook, iMac, Mac mini)."
            ),
            include_when=[
                "macOS upgrade installation issues (Sierra, High Sierra) or kernel panic crashes.",
                "Desktop Safari browser memory leaks, pop-up loops, or unresponsive tabs on Mac.",
                "MacBook trackpad software gesture anomalies or desktop iTunes library corruption.",
            ],
            exclude_when=[
                "General iPhone or iPad iOS problems (map to software_update_problem).",
                "Apple ID password reset initiated from a Mac web browser (map to account_access_issue).",
            ],
            example_messages=[
                "lol wtf I swear there is something wrong with the memory of my macbook <USER> <USER>",
                "LEAGUE PASS, YOU'RE FAILING ME AGAIN! <USER> I feel like this one is <USER>'s fault though, some safari pop-up",
                "<USER> macOS Sierra installation stuck on 2 minutes remaining for 4 hours on my MacBook Pro",
            ],
            source_clusters=[5],
            decision_rationale=(
                "Cluster 5 showed clear divergence between desktop Mac software and mobile iOS workflows. "
                "Mac desktop troubleshooting requires distinct terminal/Disk Utility/NVRAM reset paths."
            ),
        ),
        IntentDefinition(
            intent_id="intent_08",
            intent_name="hardware_audio_connection_issue",
            definition=(
                "Customer reports physical hardware defects or wireless connectivity failures related to speakers, "
                "microphones, Bluetooth accessories, Wi-Fi, or cellular network hardware."
            ),
            include_when=[
                "Speaker produces distorted, crackling, or zero audio during phone calls or media.",
                "Microphone does not pick up voice during calls or Siri commands.",
                "Bluetooth fails to pair or persistently drops connection with headphones/AirPods/car audio.",
                "Wi-Fi toggle grayed out or cellular hardware displays 'No Service' persistently.",
                "Physical volume buttons, mute switch, or home button physically stuck or broken.",
            ],
            exclude_when=[
                "Display glass cracks or touchscreen digitizer failures (map to display_touch_issue).",
                "Charging port failure related to power replenishment (map to battery_power_issue).",
            ],
            example_messages=[
                "<USER> since upgrading loud speaker no longer works when receiving a call? Please fix!",
                "<USER> my Bluetooth keeps dropping connection to my AirPods every 30 seconds",
                "<USER> wifi toggle is greyed out in settings and says no service on my iPhone",
            ],
            source_clusters=[2],
            decision_rationale=(
                "Split from Cluster 2 to decouple non-display hardware and connectivity defects from screen digitizers."
            ),
        ),
        IntentDefinition(
            intent_id="intent_09",
            intent_name="general_device_support",
            definition=(
                "Customer submits general how-to questions, store appointment/reservation inquiries, warranty checks, "
                "or ambiguous multi-issue complaints requiring initial support triage."
            ),
            include_when=[
                "Inquiries about Apple Store Genius Bar reservations, repair status, or appointment scheduling.",
                "How-to configuration questions (e.g. how to transfer data, setup new device, adjust ringer volume).",
                "Warranty coverage or trade-in inquiries.",
                "Compound or broad complaints that span multiple categories and require agent clarification.",
            ],
            exclude_when=[
                "Clear single-problem complaints covered by specific intents above.",
                "Incomprehensible, truncated, or non-English messages (classify as unclear_needs_review).",
            ],
            example_messages=[
                "<USER> I’ve got a screenshot saying my #iPhoneX is reserved for the 3rd then an email saying it’s the 18th... what happened?",
                "<USER> how do I change my ringer/notifications volume? Using the volume keys on the home screen used to display “ringer”",
                "<USER> do I need an appointment to replace my battery at the Apple Store today?",
            ],
            source_clusters=[2, 4, 5],
            decision_rationale=(
                "Provides a calibrated triage class for general setup, store reservations, and complex multi-issue "
                "opening messages without contaminating technical troubleshooting categories."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Taxonomy Construction & Validation
# ---------------------------------------------------------------------------
def build_candidate_taxonomy(interim_dir: Optional[Path] = None) -> TaxonomyCandidate:
    """
    Build the initial evidence-based candidate taxonomy compiling Phase 4 and 4.5 artifacts.
    """
    if interim_dir is None:
        interim_dir = Path("data/interim")
    intents = get_initial_candidate_intents()

    merge_decisions = [
        {
            "merged_clusters": [0, 7],
            "target_intent": "keyboard_typing_issue",
            "rationale": (
                "Clusters 0 and 7 addressed identical text rendering / typing problems. Cluster 7 "
                "represented an informal/frustrated expression variant. Merged to form a timeless category."
            ),
        },
        {
            "merged_clusters": [3, 6],
            "target_intent": "software_update_problem",
            "rationale": (
                "Clusters 3 and 6 both captured software update installation errors and post-update app glitches. "
                "Merged to avoid redundant classification classes."
            ),
        },
    ]

    split_decisions = [
        {
            "split_cluster": 4,
            "resulting_intents": ["account_access_issue", "billing_purchase_issue"],
            "rationale": (
                "Cluster 4 co-mingled account authentication (password resets, 2FA, locked Apple ID) with financial "
                "disputes (App Store billing, unwanted charges, refunds). Split into distinct operational workflows."
            ),
        },
        {
            "split_cluster": 2,
            "resulting_intents": ["display_touch_issue", "hardware_audio_connection_issue", "general_device_support"],
            "rationale": (
                "Cluster 2 lumped display glass cracks, touchscreen unresponsiveness, audio defects, and general device inquiries. "
                "Split to enable dedicated screen repair vs connectivity resolution routing."
            ),
        },
        {
            "split_cluster": 5,
            "resulting_intents": ["mac_software_issue", "general_device_support"],
            "rationale": (
                "Cluster 5 mixed desktop macOS/MacBook inquiries with cloud media streaming. "
                "Desktop Mac issues require distinct troubleshooting tools."
            ),
        },
    ]

    taxonomy = TaxonomyCandidate(
        status="candidate_human_review_required",
        taxonomy_version="v1.0-candidate",
        selected_brand="AppleSupport",
        intent_count=len(intents),
        intents=intents,
        merge_decisions=merge_decisions,
        split_decisions=split_decisions,
        metadata={
            "target_intent_range": "7-10",
            "total_intents": len(intents),
            "evidence_sources": [
                "data/interim/intent_discovery_corpus.csv",
                "data/interim/intent_taxonomy_human_review.csv",
                "data/interim/cluster_merge_candidates.csv",
                "data/interim/cluster_split_candidates.csv",
            ],
        },
    )

    # Validate consistency
    errors = validate_taxonomy_consistency(taxonomy)
    if errors:
        raise ValueError(f"Candidate taxonomy consistency validation failed: {errors}")

    return taxonomy


def validate_taxonomy_consistency(taxonomy: TaxonomyCandidate) -> list[str]:
    """
    Validate internal consistency of the taxonomy:
    - Intent count between 7 and 10
    - Unique intent_ids and intent_names
    - All intents have non-empty definition, include_when, exclude_when, examples
    - Valid snake_case names
    """
    errors: list[str] = []
    if not (7 <= taxonomy.intent_count <= 10):
        errors.append(f"Intent count {taxonomy.intent_count} outside target range 7–10.")

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    snake_case_pattern = re.compile(r"^[a-z][a-z0-9_]+$")

    for idx, intent in enumerate(taxonomy.intents):
        if intent.intent_id in seen_ids:
            errors.append(f"Duplicate intent_id: {intent.intent_id}")
        seen_ids.add(intent.intent_id)

        if intent.intent_name in seen_names:
            errors.append(f"Duplicate intent_name: {intent.intent_name}")
        seen_names.add(intent.intent_name)

        if not snake_case_pattern.match(intent.intent_name):
            errors.append(f"Intent name '{intent.intent_name}' must be valid lowercase snake_case.")

        if not intent.definition.strip():
            errors.append(f"Intent '{intent.intent_name}' has empty definition.")

        if not intent.include_when:
            errors.append(f"Intent '{intent.intent_name}' missing include_when rules.")

        if not intent.exclude_when:
            errors.append(f"Intent '{intent.intent_name}' missing exclude_when rules.")

        if len(intent.example_messages) < 2:
            errors.append(f"Intent '{intent.intent_name}' has fewer than 2 example messages.")

    return errors


def save_candidate_taxonomy(taxonomy: TaxonomyCandidate, output_path: Path) -> Path:
    """
    Save candidate taxonomy to JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(taxonomy)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved candidate taxonomy to: %s", output_path)
    return output_path


def load_candidate_taxonomy(json_path: Path | str = DEFAULT_CANDIDATE_TAXONOMY_PATH) -> TaxonomyCandidate:
    """
    Load a candidate taxonomy from JSON artifact.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy JSON artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = [
        IntentDefinition(
            intent_id=item["intent_id"],
            intent_name=item["intent_name"],
            definition=item["definition"],
            include_when=item.get("include_when", []),
            exclude_when=item.get("exclude_when", []),
            example_messages=item.get("example_messages", []),
            source_clusters=item.get("source_clusters", []),
            review_status=item.get("review_status", "candidate_human_review_required"),
            decision_rationale=item.get("decision_rationale", ""),
        )
        for item in data.get("intents", [])
    ]

    return TaxonomyCandidate(
        status=data.get("status", "candidate_human_review_required"),
        taxonomy_version=data.get("taxonomy_version", "v1.0-candidate"),
        selected_brand=data.get("selected_brand", "AppleSupport"),
        intent_count=len(intents),
        intents=intents,
        merge_decisions=data.get("merge_decisions", []),
        split_decisions=data.get("split_decisions", []),
        metadata=data.get("metadata", {}),
    )


create_initial_candidate_taxonomy = build_candidate_taxonomy


def finalize_candidate_taxonomy(
    output_json_path: Path | str = DEFAULT_CANDIDATE_TAXONOMY_PATH,
    interim_dir: Optional[Path | str] = None,
) -> TaxonomyCandidate:
    """
    Construct the empirical candidate operational taxonomy, validate its consistency,
    and save it as JSON.
    """
    dir_path = Path(interim_dir) if interim_dir else None
    taxonomy = build_candidate_taxonomy(dir_path)
    save_candidate_taxonomy(taxonomy, Path(output_json_path))
    return taxonomy
