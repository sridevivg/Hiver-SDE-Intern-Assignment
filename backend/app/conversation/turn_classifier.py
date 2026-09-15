"""
SupportGraph AI — Turn Classifier.

Classifies incoming customer messages into 9 semantic MessageRoleTypes,
extracts stated facts (distinguishing confirmed facts from inferences),
and detects customer-reported action attempts or problem worsening.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from app.conversation.action_catalog import ActionCatalog
    from app.conversation.conversation_state import (
        ConversationState,
        MessageRoleType,
        ResolutionStage,
    )
except ModuleNotFoundError:
    from backend.app.conversation.action_catalog import ActionCatalog  # type: ignore[no-redef]
    from backend.app.conversation.conversation_state import (  # type: ignore[no-redef]
        ConversationState,
        MessageRoleType,
        ResolutionStage,
    )


# Regular expression patterns for turn classification
RESOLUTION_CONFIRMATION_PATTERNS = [
    r"\b(that\s*worked|worked\s*like\s*a\s*charm|it\s*worked)\b",
    r"\b(it'?s\s*(?:working|fixed|resolved|fine\s*now))\b",
    r"\b(fixed\s*it|problem\s*is\s*solved|issue\s*is\s*resolved)\b",
    r"\b(working\s*now|all\s*good\s*now|good\s*to\s*go|back\s*to\s*normal)\b",
    r"\b(solved\s*the\s*problem|thank\s*you\s*so\s*much\s*it\s*works)\b",
    r"\b(thank\s*you,?\s*that\s*(?:did\s*it|helped|worked))\b",
]

NEGATIVE_RESULT_PATTERNS = [
    r"\b(didn'?t\s*work|still\s*not\s*working|doesn'?t\s*work)\b",
    r"\b(still\s*(?:broken|fails|the\s*same|having\s*the\s*issue|not\s*connecting))\b",
    r"\b(same\s*(?:issue|problem|thing|result))\b",
    r"\b(no\s*luck|nothing\s*changed|hasn'?t\s*helped|made\s*no\s*difference)\b",
    r"\b(still\s*won'?t\s*(?:turn\s*on|connect|charge|update|open))\b",
]

WORSENING_PATTERNS = [
    r"\b(now\s*it'?s?\s*(?:worse|completely\s*dead|black\s*screen|won'?t\s*turn\s*on\s*at\s*all))\b",
    r"\b(it\s*got\s*worse|getting\s*worse|made\s*it\s*worse)\b",
    r"\b(now\s*(?:nothing\s*works|it\s*shuts\s*off\s*immediately|smoke|sparking))\b",
    r"\b(completely\s*unresponsive\s*now|won'?t\s*even\s*power\s*on\s*now)\b",
    r"\b(smoking|smoke|sparking|sparks|burning|burnt|fire|flames)\b",
    r"\b(swelling|swollen|bulging|expanded\s*battery)\b",
    r"\b((?:extremely|very|burning)\s*hot(?:\s*to\s*touch)?)\b",
    r"\b((?:connect|transfer|speak|talk|pass|escalate)\s*(?:me|us)?\s*(?:with|to)?\s*(?:a\s*)?(?:human|specialist|agent|representative|person|advisor|technical\s*support))\b",
    r"\b(human\s*(?:specialist|agent|representative|support|help))\b",
]

UNCLEAR_PATTERNS = [
    r"^(?:\?+|\.+|!+|huh\??|what\??|idk|dunno|ok|okay|k|yes|no)$",
    r"^(?:my\s+phone|my\s+device|phone|device|it|help|help\s*me|broken|fix\s*it|idk|dunno)$",
    r"^[a-zA-Z]{1,3}$",
]

# Fact extraction regex patterns
DEVICE_MODEL_PATTERNS = [
    r"\b(iphone\s*(?:1[1-6]|x[s|r]?|[6-8]|se)(?:\s*(?:pro\s*max|pro|plus|mini))?)\b",
    r"\b(ipad\s*(?:pro|air|mini)?(?:\s*(?:1[0-2]|gen|[0-9]+))?)\b",
    r"\b(macbook\s*(?:pro|air)?)\b",
    r"\b(apple\s*watch(?:\s*series\s*[0-9]+)?)\b",
    r"\b(airpods(?:\s*pro|\s*max)?)\b",
]

OS_VERSION_PATTERNS = [
    r"\b(ios\s*(?:1[4-8](?:\.[0-9]+)*))\b",
    r"\b(ipados\s*(?:1[4-8](?:\.[0-9]+)*))\b",
    r"\b(macos\s*(?:1[2-5](?:\.[0-9]+)*|sonoma|ventura|monterey|sequoia))\b",
    r"\b(watchos\s*(?:9|1[0-1](?:\.[0-9]+)*))\b",
]

CARRIER_PATTERNS = [
    r"\b(verizon|at&t|t-mobile|sprint|vodafone|ee|o2|airtel|jio)\b",
]


class TurnClassifier:
    """
    Classifies customer messages into dialogue roles and extracts confirmed facts.
    """

    @classmethod
    def extract_facts(cls, text: str) -> Dict[str, Any]:
        """
        Extract explicit facts stated by customer in the text.
        Returns a dict of identified facts.
        """
        facts: Dict[str, Any] = {}
        text_lower = text.lower()

        # Extract device model
        for pattern in DEVICE_MODEL_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                facts["device_model"] = match.group(1).title()
                break

        # Extract OS version
        for pattern in OS_VERSION_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                facts["os_version"] = match.group(1).upper()
                break

        # Extract Carrier
        for pattern in CARRIER_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                facts["carrier"] = match.group(1).title()
                break

        return facts

    @classmethod
    def classify_turn(
        cls,
        message: str,
        state: ConversationState,
    ) -> Tuple[MessageRoleType, Dict[str, Any], Optional[str]]:
        """
        Classify the role of the message, extract confirmed facts, and identify referenced action.
        Returns (role_type, extracted_facts, referenced_action).
        """
        text = message.strip()
        text_lower = text.lower()
        extracted_facts = cls.extract_facts(text)
        referenced_action = ActionCatalog.identify_action(text)

        # 1. Check for UNCLEAR_MESSAGE
        if not text or len(text) < 4 or any(re.match(p, text_lower) for p in UNCLEAR_PATTERNS):
            # If length is very short and no facts or actions extracted
            if not extracted_facts and not referenced_action:
                return MessageRoleType.UNCLEAR_MESSAGE, extracted_facts, referenced_action

        # 2. Check for PROBLEM_WORSENING
        for pat in WORSENING_PATTERNS:
            if re.search(pat, text_lower):
                return MessageRoleType.PROBLEM_WORSENING, extracted_facts, referenced_action

        # 3. Check for RESOLUTION_CONFIRMATION
        for pat in RESOLUTION_CONFIRMATION_PATTERNS:
            if re.search(pat, text_lower):
                return MessageRoleType.RESOLUTION_CONFIRMATION, extracted_facts, referenced_action

        # 4. Check for NEGATIVE_RESOLUTION_RESULT
        is_negative = any(re.search(pat, text_lower) for pat in NEGATIVE_RESULT_PATTERNS)
        if is_negative:
            return MessageRoleType.NEGATIVE_RESOLUTION_RESULT, extracted_facts, referenced_action

        # 5. Check for ACTION_ATTEMPT_RESULT
        # If user explicitly mentions doing an action or past tense action
        if referenced_action is not None:
            # Check for past attempt cues
            past_attempt_cues = [
                r"\b(already|tried|did|done|performed|tested|attempted)\b",
                r"\b(i\s*(?:restarted|rebooted|toggled|unpaired|reset|cleaned|updated|checked|cleared))\b",
                r"\b(after\s*(?:restarting|rebooting|resetting|updating))\b",
            ]
            if any(re.search(cue, text_lower) for cue in past_attempt_cues) or is_negative:
                return MessageRoleType.ACTION_ATTEMPT_RESULT, extracted_facts, referenced_action

        # 6. Check for CLARIFICATION_ANSWER
        # If the conversation was awaiting clarification or the previous agent message asked a question
        if state.stage == ResolutionStage.CLARIFYING or (
            state.turns and state.turns[-1].speaker == "agent" and "?" in state.turns[-1].content
        ):
            # If user provided facts or answered a question
            if extracted_facts or len(text.split()) <= 15:
                # Check if it's not a new problem statement
                if not (len(text.split()) > 15 and "won't" in text_lower):
                    return MessageRoleType.CLARIFICATION_ANSWER, extracted_facts, referenced_action

        # 7. Check for NEW_PROBLEM_IN_EXISTING_CONVERSATION
        if len(state.turns) >= 2 and state.problem_family:
            # Check if user message introduces a completely different problem family
            try:
                from app.understanding.problem_extractor import ProblemExtractor
            except ModuleNotFoundError:
                from backend.app.understanding.problem_extractor import ProblemExtractor  # type: ignore[no-redef]
            new_profile = ProblemExtractor().extract(text)
            new_fam = new_profile.primary_problem_family.value if new_profile.primary_problem_family else None
            if new_fam and new_fam != state.problem_family:
                if new_profile.family_confidence >= 0.7:
                    return MessageRoleType.NEW_PROBLEM_IN_EXISTING_CONVERSATION, extracted_facts, referenced_action

        # 8. Check for NEW_PROBLEM (First customer turn)
        if len(state.turns) == 0:
            return MessageRoleType.NEW_PROBLEM, extracted_facts, referenced_action

        # 9. Default to CONTINUATION
        return MessageRoleType.CONTINUATION, extracted_facts, referenced_action
